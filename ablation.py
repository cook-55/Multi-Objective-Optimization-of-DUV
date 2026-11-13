"""Run the paired material-library and candidate-screening ablation."""

import argparse
import hashlib
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ablation_figures import save_ablation_figures
from demo import (
    CALC_VERSION,
    MAX_TOTAL_NM,
    SURROGATE_TARGETS,
    THICKNESS_MAX_NM,
    THICKNESS_MIN_NM,
    TRANSMISSION_PENALTY_WEIGHT,
    TRANSMISSION_TARGET_PERCENT,
    CoatingDataset,
    CurvedSubstrate,
    DatasetGenerator,
    MaterialDatabase,
    Surrogate,
    TMMSolver,
    generate_all_strict_templates,
    is_valid_material_sequence,
    load_torch_weights,
    performance_cost,
    refine_thickness,
    screen_candidates,
)


HERE = Path(__file__).resolve().parent
GROUPS = {
    "A": ("full", "dnn"),
    "B": ("full", "random"),
    "C": ("binary", "dnn"),
    "D": ("binary", "random"),
}
BINARY = tuple(generate_all_strict_templates([])[0])
COMPARISONS = {"A-B": ("A", "B"), "C-D": ("C", "D"),
               "B-D": ("B", "D"), "A-C": ("A", "C")}
REPORT_METRICS = (
    "comprehensive_cost", "avg_R", "avg_T", "std_R", "avg_stress",
    "refinement_gain", "tmm_evals", "online_seconds",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-size", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1001,
                        help="First candidate seed. The following seeds are consecutive.")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--per-start-evals", type=int, default=400,
                        help="Maximum extra TMM evaluations per start after its initial score.")
    parser.add_argument("--output-dir", type=Path,
                        help="A new directory for this experiment.")
    parser.add_argument("--stack-study", nargs="?", type=Path,
                        const=HERE / "deep_ar_optimization_results" / "best_design.csv",
                        help="Also compare YF3/AlF3 substitutions in the saved seven-layer design.")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Also run 1000 paired thickness perturbations per selected design.")
    args = parser.parse_args()
    if not 1 <= args.top_k <= args.pool_size:
        parser.error("Require 1 <= --top-k <= --pool-size.")
    if args.seed < 0 or args.repeats < 1 or args.per_start_evals < 1:
        parser.error("Require a nonnegative seed and positive repeats and per-start budget.")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def design_key(materials, thickness_nm):
    # Match the resolution of the training CSV when checking for overlap.
    return tuple(materials), tuple(f"{float(value):.2f}" for value in thickness_nm)


def load_frozen_surrogate(device):
    data_path = HERE / "coating_dataset.csv"
    weight_path = HERE / "best_surrogate.pth"
    scaler_path = HERE / "surrogate_scalers.pkl"
    for path in (data_path, weight_path, scaler_path):
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Prepare the current demo.py data and model first.")

    dataset = CoatingDataset(data_path, max_layers=9)
    if ("ver" not in dataset.df or not dataset.df["ver"].eq(CALC_VERSION).all()
            or len(dataset) < 2):
        raise SystemExit("The cached dataset must match the current demo.py calculation version.")

    # These are the local scaler files produced by demo.py.
    with scaler_path.open("rb") as stream:
        bundle = pickle.load(stream)
    scaler_t, scaler_y = bundle["thickness"], bundle["target"]
    train_size = int(0.9 * len(dataset))
    split = torch.Generator().manual_seed(42)
    fit_ids = torch.randperm(len(dataset), generator=split)[:train_size].numpy()
    for saved, values in ((scaler_t, dataset.Xt), (scaler_y, dataset.raw_y)):
        fitted = StandardScaler().fit(values[fit_ids])
        if (saved.n_features_in_ != fitted.n_features_in_
                or not np.array_equal(saved.n_samples_seen_, fitted.n_samples_seen_)
                or not all(np.allclose(getattr(saved, name), getattr(fitted, name),
                                       rtol=1e-6, atol=1e-8)
                           for name in ("mean_", "var_", "scale_"))):
            raise SystemExit("The saved scalers do not match the cached dataset's demo.py split.")

    model = Surrogate(dataset.maxL * (dataset.nmat + 1))
    model.load_state_dict(load_torch_weights(weight_path, map_location="cpu"), strict=True)
    model.to(device).eval()
    known = {
        design_key(materials.split("|"), thickness.split(","))
        for materials, thickness in zip(dataset.df["materials"], dataset.df["thickness_nm"])
    }
    source_paths = (data_path, weight_path, scaler_path, HERE / "demo.py",
                    HERE / "ablation.py", HERE / "ablation_figures.py")
    provenance = {
        "source": "existing frozen checkpoint",
        "sha256": {path.name: file_sha256(path) for path in source_paths},
        "offline_label_tmm_evals": len(dataset),
        "training_rows": train_size,
        "validation_rows": len(dataset) - train_size,
        "offline_label_seconds": None,
        "offline_training_seconds": None,
    }
    screen_args = (model, scaler_t, scaler_y, device, dataset.maxL,
                   dataset.idx, dataset.ys_names)
    return screen_args, known, provenance


def make_paired_pools(substrate, db, count, rng, known):
    generator = DatasetGenerator(substrate, db, rng=rng)
    full, binary = [], []
    pool_keys = set()
    for _ in range(20 * count):
        d, materials = generator.random_valid_design()
        thickness_nm = d * 1e9
        keys = {design_key(materials, thickness_nm), design_key(BINARY, thickness_nm)}
        if keys & known or keys & pool_keys:
            continue
        pool_keys.update(keys)
        full.append((tuple(materials), thickness_nm))
        binary.append((BINARY, thickness_nm))
        if len(full) == count:
            return {"full": full, "binary": binary}
    raise RuntimeError("Could not build the requested paired pool without data overlap.")


class EvaluationLimit(Exception):
    """Stop before another exact solver call would exceed the evaluation cap."""


class ExactEvaluator:
    def __init__(self, tmm, budget):
        self.tmm = tmm
        self.budget = budget
        self.calls = 0
        self.best = None
        self.trace = []

    def evaluate(self, materials, thickness_nm, candidate_id):
        d = np.asarray(thickness_nm, dtype=float)
        # The total-thickness cutoff matches the exact objective in demo.py.
        if (not np.isfinite(d).all() or np.any(d < THICKNESS_MIN_NM - 1e-9)
                or np.any(d > THICKNESS_MAX_NM + 1e-9) or d.sum() > MAX_TOTAL_NM):
            return 1e9
        if self.calls >= self.budget:
            raise EvaluationLimit
        self.calls += 1
        perf = self.tmm.calculate_objectives(d * 1e-9, list(materials))
        score = float(performance_cost(perf))
        if not np.isfinite(score):
            raise RuntimeError("The exact TMM objective returned a non-finite score.")
        if self.best is None or score < self.best["comprehensive_cost"]:
            # Invert the existing relative uniformity definition to get percentage points.
            std_R = perf["uniformity"] * max(perf["avg_R"] / 100.0, 1e-6)
            self.best = {
                **perf,
                "std_R": std_R,
                "avg_A": 100.0 - perf["avg_R"] - perf["avg_T"],
                "comprehensive_cost": score,
                "candidate_id": candidate_id,
                "materials": "|".join(materials),
                "thickness_nm": json.dumps(d.tolist()),
            }
            self.trace.append({"tmm_evals": self.calls, "best_cost": score})
        return score

    def finish_trace(self):
        if self.trace[-1]["tmm_evals"] != self.calls:
            self.trace.append({"tmm_evals": self.calls,
                               "best_cost": self.best["comprehensive_cost"]})


def run_refinement(pool, selected, tmm, per_start_evals, label):
    evaluator = ExactEvaluator(tmm, len(selected) * (1 + per_start_evals))
    initial_scores = {}
    for candidate_id in selected:
        materials, d_init = pool[candidate_id]
        initial_scores[candidate_id] = evaluator.evaluate(materials, d_init, candidate_id)
    if evaluator.calls != len(selected):
        raise ValueError("All selected initial designs must satisfy the common thickness limits.")
    initial = evaluator.best.copy()
    successful, budget_stops = 0, 0
    for candidate_id in tqdm(selected, desc=label, leave=False):
        materials, d_init = pool[candidate_id]
        start_calls = evaluator.calls

        def evaluate_cost(d):
            if np.array_equal(d, d_init):
                return initial_scores[candidate_id]
            if evaluator.calls - start_calls >= per_start_evals:
                raise EvaluationLimit
            return evaluator.evaluate(materials, d, candidate_id)

        try:
            result = refine_thickness(d_init, evaluate_cost)
        except EvaluationLimit:
            budget_stops += 1
        else:
            successful += int(result.success)
    evaluator.finish_trace()
    return initial, evaluator, successful, budget_stops


def write_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_comparisons(runs, traces, out):
    write_csv(runs, out / "ablation_runs.csv")
    write_csv(traces, out / "ablation_trace.csv")
    stats = []
    for group, table in runs.groupby("group"):
        for metric in REPORT_METRICS:
            values = table[metric]
            stats.append({"group": group, "metric": metric, "n": len(values),
                          "median": values.median(),
                          "q1": values.quantile(0.25), "q3": values.quantile(0.75)})
    write_csv(pd.DataFrame(stats), out / "comparison_summary.csv")

    paired = []
    for comparison, (left, right) in COMPARISONS.items():
        left_data = runs[runs["group"] == left].set_index("seed")
        right_data = runs[runs["group"] == right].set_index("seed")
        diff = left_data[list(REPORT_METRICS)] - right_data[list(REPORT_METRICS)]
        for seed, values in diff.dropna().iterrows():
            paired.append({"comparison": comparison, "seed": seed, **values.to_dict()})
    paired = pd.DataFrame(paired)
    write_csv(paired, out / "paired_differences.csv")
    return paired


def load_stack_reference(path):
    table = pd.read_csv(path)
    if len(table) != 1 or "layer8_material" in table:
        raise SystemExit("--stack-study requires a single saved seven-layer best_design.csv.")
    row = table.iloc[0]
    materials = tuple(row[f"layer{i}_material"] for i in range(1, 8))
    expected = BINARY[:2] + ("YF3",) + BINARY[3:6] + ("AlF3",)
    if materials != expected:
        raise SystemExit("--stack-study requires the YF3 layer 3 / AlF3 layer 7 reference stack.")
    thickness = np.array([row[f"layer{i}_thickness_nm"] for i in range(1, 8)], dtype=float)
    if (not np.isfinite(thickness).all() or thickness.min() < THICKNESS_MIN_NM
            or thickness.max() > THICKNESS_MAX_NM or thickness.sum() > MAX_TOTAL_NM):
        raise SystemExit("The saved reference thicknesses are outside the common search bounds.")
    return materials, thickness


def run_stack_study(reference, tmm, per_start_evals):
    reference_materials, thickness = reference
    variants = {"both": (True, True), "yf3_only": (True, False),
                "alf3_only": (False, True), "neither": (False, False)}
    rows = []
    for variant, (keep_yf3, keep_alf3) in variants.items():
        mats = list(reference_materials)
        if not keep_yf3:
            mats[2] = BINARY[2]
        if not keep_alf3:
            mats[6] = BINARY[6]
        if not is_valid_material_sequence(mats):
            raise RuntimeError(f"Invalid reference substitution: {variant}")
        fixed, evaluator, successful, budget_stops = run_refinement(
            [(tuple(mats), thickness)], [0], tmm, per_start_evals, f"Stack {variant}")
        rows.append({"variant": variant, "stage": "fixed", "tmm_evals": 1, **fixed})
        rows.append({"variant": variant, "stage": "refined", "tmm_evals": evaluator.calls,
                     "successful_starts": successful, "budget_stops": budget_stops,
                     **evaluator.best})
    return pd.DataFrame(rows)


def robustness_row(tmm, design, group, seed):
    d = np.array(json.loads(design["thickness_nm"])) * 1e-9
    samples = tmm.monte_carlo_robustness(
        d, design["materials"].split("|"), n_samples=1000,
        thickness_sigma_relative=0.015, seed=42)
    row = {"group": group, "seed": seed, "samples": 1000, "mc_seed": 42,
           "thickness_sigma_relative": 0.015}
    for metric, values in samples.items():
        row.update({
            f"{metric}_mean": np.mean(values), f"{metric}_std": np.std(values),
            f"{metric}_p5": np.percentile(values, 5), f"{metric}_p95": np.percentile(values, 95),
        })
    return row


def main():
    args = parse_args()
    out = args.output_dir or HERE / "ablation_results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out.resolve()
    if out.exists():
        raise SystemExit(f"Choose a new output directory. Already exists: {out}")
    reference = load_stack_reference(args.stack_study) if args.stack_study else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading frozen model and checking dataset overlap keys on {device}...")
    screen_args, known, provenance = load_frozen_surrogate(device)
    db = MaterialDatabase()
    substrate = CurvedSubstrate(radius=100e-3, aperture=105e-3, num_zones=20)
    tmm = TMMSolver(substrate, db)
    budget = args.top_k * (1 + args.per_start_evals)
    print(f"Offline labels: {provenance['offline_label_tmm_evals']:,} TMM evaluations.")
    print(f"Online cap: {budget:,} TMM evaluations per group and seed.")
    pool_dir = out / "candidate_pools"
    pool_dir.mkdir(parents=True)
    settings = {
        "calculation_version": CALC_VERSION, "device": str(device),
        "groups": GROUPS, "binary_materials": BINARY, "layers": 7,
        "pool_size": args.pool_size, "top_k": args.top_k,
        "seeds": list(range(args.seed, args.seed + args.repeats)),
        "per_start_extra_tmm_evals": args.per_start_evals, "online_tmm_budget": budget,
        "thickness_min_nm": THICKNESS_MIN_NM, "thickness_max_nm": THICKNESS_MAX_NM,
        "max_total_nm": MAX_TOTAL_NM, "wavelength_nm": 193,
        "curvature_radius_mm": 100, "aperture_diameter_mm": 105, "radial_zones": 20,
        "transmission_target_percent": TRANSMISSION_TARGET_PERCENT,
        "transmission_penalty_weight": TRANSMISSION_PENALTY_WEIGHT,
        "stack_reference": str(args.stack_study.resolve()) if args.stack_study else None,
        "monte_carlo": args.monte_carlo, "model": provenance,
        "trace_format": "incumbent improvements and final evaluation",
        "incumbent": "best feasible exact evaluation, including initial points",
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    run_rows, trace_rows, mc_rows = [], [], []
    for seed in settings["seeds"]:
        pool_start = perf_counter()
        pool_stream, selection_stream = np.random.SeedSequence(seed).spawn(2)
        pool_rng = random.Random(int(pool_stream.generate_state(1, dtype=np.uint64)[0]))
        pools = make_paired_pools(substrate, db, args.pool_size, pool_rng, known)
        pool_seconds = perf_counter() - pool_start
        write_csv(pd.DataFrame([
            {"candidate_id": i, "full_materials": "|".join(materials),
             "thickness_nm": json.dumps(d.tolist())}
            for i, (materials, d) in enumerate(pools["full"])
        ]), pool_dir / f"seed_{seed}.csv")

        for group, (material_group, method) in GROUPS.items():
            pool = pools[material_group]
            started = perf_counter()
            if method == "dnn":
                scores = screen_candidates(pool, *screen_args)
                if not np.isfinite(scores).all():
                    raise RuntimeError("The surrogate produced non-finite screening scores.")
                selected = sorted(range(len(pool)), key=scores.__getitem__)[:args.top_k]
            else:
                # Reusing this stream's seed pairs the initial thicknesses in B and D.
                selected = np.random.default_rng(selection_stream).choice(
                    len(pool), size=args.top_k, replace=False).tolist()
            screening_seconds = perf_counter() - started
            label = f"Seed {seed} / {group}"
            initial, evaluator, successful, budget_stops = run_refinement(
                pool, selected, tmm, args.per_start_evals, label)
            online_seconds = perf_counter() - started
            best = evaluator.best
            run_rows.append({
                "group": group, "seed": seed, "material_group": material_group,
                "screening": method, "initial_candidate_ids": json.dumps(selected),
                "best_initial_candidate_id": initial["candidate_id"],
                "initial_cost": initial["comprehensive_cost"], "initial_avg_R": initial["avg_R"],
                "initial_avg_T": initial["avg_T"],
                "refinement_gain": initial["comprehensive_cost"] - best["comprehensive_cost"],
                "successful_starts": successful, "budget_stops": budget_stops,
                "budget": budget, "tmm_evals": evaluator.calls, "pool_seconds": pool_seconds,
                "screening_seconds": screening_seconds, "online_seconds": online_seconds,
                "transmission_shortfall_pp": max(0.0, TRANSMISSION_TARGET_PERCENT - best["avg_T"]),
                **best,
            })
            trace_rows.extend({"group": group, "seed": seed, **point} for point in evaluator.trace)
            print(f"{label} | initial {initial['comprehensive_cost']:.4f} "
                  f"-> best {best['comprehensive_cost']:.4f} | "
                  f"R {best['avg_R']:.4f}% | T {best['avg_T']:.4f}% | "
                  f"TMM {evaluator.calls}/{budget}")
            if args.monte_carlo:
                mc_rows.append(robustness_row(tmm, best, group, seed))

        runs, traces = pd.DataFrame(run_rows), pd.DataFrame(trace_rows)
        paired = write_comparisons(runs, traces, out)
        if mc_rows:
            write_csv(pd.DataFrame(mc_rows), out / "robustness_comparison.csv")

    stack_runs = None
    if reference is not None:
        stack_runs = run_stack_study(reference, tmm, args.per_start_evals)
        write_csv(stack_runs, out / "material_stack_comparison.csv")
    figures = save_ablation_figures(runs, traces, out / "figures", stack_runs, settings=settings)
    medians = runs.groupby("group")[list(REPORT_METRICS)].median()
    print("\nGroup medians across paired seeds")
    print(medians.to_string(float_format=lambda value: f"{value:.4f}"))
    deltas = paired.groupby("comparison")[list(REPORT_METRICS)].median()
    print("\nMedian paired differences (left group minus right group)")
    print(deltas.to_string(float_format=lambda value: f"{value:+.4f}"))
    print(f"\nTables and {len(figures)} figure files saved to {out}")


if __name__ == "__main__":
    main()
