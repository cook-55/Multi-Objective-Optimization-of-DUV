"""Reproduce the manuscript's conditional, material-wise stress sensitivity.

Uses the saved (0.1 nm rounded) thicknesses, not the unavailable full-precision
optimizer vector. The input distributions are assumptions, not measured yields.
Run: python code/stress_uncertainty.py
"""
import ast
import csv
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    tree = ast.parse((root / "demo.py").read_text(encoding="utf-8-sig"))
    database = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "MaterialDatabase")
    materials = {}
    for node in ast.walk(database):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in ("mainstream", "auxiliary"):
                    materials.update(ast.literal_eval(node.value))
    source = root / "deep_ar_optimization_results" / "best_design.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        design = next(csv.DictReader(handle))
    layers = [design[f"layer{i}_material"] for i in range(1, 8)]
    thickness = np.array([float(design[f"layer{i}_thickness_nm"]) for i in range(1, 8)])
    names = sorted(set(layers))
    weights = np.array([sum(d for m, d in zip(layers, thickness) if m == name)
                        for name in names]) / thickness.sum()
    stress = np.array([materials[name]["stress"] for name in names])
    contributions = weights * stress
    mean = float(contributions.sum())
    seed, samples, threshold = 2026, 1_000_000, 50.0
    cases = []
    for amplitude in (0.2, 0.3):
        # All occurrences of a material share its draw; different materials
        # are independent. Reset the seed to pair the two amplitude scenarios.
        rng = np.random.default_rng(seed)
        errors = rng.uniform(-amplitude, amplitude, size=(samples, len(names)))
        values = mean + errors @ contributions
        half_width = float(amplitude * np.abs(contributions).sum())
        cases.append({
            "relative_half_width": amplitude,
            "analytic_mean_MPa": mean,
            "analytic_sd_MPa": float(amplitude * np.linalg.norm(contributions) / np.sqrt(3)),
            "support_MPa": [mean - half_width, mean + half_width],
            "sample_mean_MPa": float(values.mean()),
            "sample_sd_MPa": float(values.std()),
            "fraction_abs_stress_above_threshold": float(np.mean(np.abs(values) > threshold)),
        })
    result = {
        "design_source": "deep_ar_optimization_results/best_design.csv",
        "thickness_precision_nm": 0.1,
        "materials": names,
        "material_thickness_weights": weights.tolist(),
        "nominal_material_stress_MPa": stress.tolist(),
        "saved_full_precision_design_stress_MPa": float(design["avg_stress"]),
        "seed": seed, "samples_per_case": samples, "threshold_MPa": threshold,
        "numpy_version": np.__version__, "cases": cases,
    }
    output = source.parent / "stress_uncertainty.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
