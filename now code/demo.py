import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Patch
from scipy.optimize import minimize
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 材料数据库与曲面基底
# ============================================================
class MaterialDatabase:
    def __init__(self):
        self.substrates = {
            "SiO2": {"n": 1.560, "k": 1.5e-4, "dn_dT": 11.0, "alpha": 0.55, "stress": 0.0}
        }
        self.mainstream = {
            "MgF2": {"n": 1.44, "k": 7.8e-4, "dn_dT": -2.0, "alpha": 13.5, "stress": 300.0},
            "LaF3": {"n": 1.72, "k": 2.0e-3, "dn_dT": -1.0, "alpha": 17.0, "stress": -200.0}
        }
        self.auxiliary = {
            "AlF3": {"n": 1.40, "k": 3.0e-4, "dn_dT": -1.5, "alpha": 13.0, "stress": 100.0},
            "YF3": {"n": 1.58, "k": 8.0e-4, "dn_dT": -1.2, "alpha": 14.0, "stress": -100.0},
            "GdF3": {"n": 1.70, "k": 2.2e-3, "dn_dT": -1.0, "alpha": 14.5, "stress": -80.0},
            "Al2O3": {"n": 1.85, "k": 2.5e-3, "dn_dT": 13.0, "alpha": 8.1, "stress": 500.0}
        }
        self.all_materials = {}
        self.all_materials.update(self.mainstream)
        self.all_materials.update(self.auxiliary)
        self.lambda0 = 193e-9
        self.substrate_name = "SiO2"

    def get_refractive_index(self, mat_name: str, temperature: float = 20.0) -> complex:
        if mat_name == "air":
            return complex(1.0, 0.0)
        elif mat_name in self.substrates:
            mat = self.substrates[mat_name]
        elif mat_name in self.all_materials:
            mat = self.all_materials[mat_name]
        else:
            raise ValueError(f"Unknown material: {mat_name}")
        n_real = mat["n"] + mat["dn_dT"] * 1e-6 * (temperature - 20.0)
        return complex(n_real, mat["k"])

    def get_stress(self, mat_name: str) -> float:
        if mat_name in self.all_materials:
            return self.all_materials[mat_name]["stress"]
        return 0.0


class CurvedSubstrate:
    def __init__(self, radius: float = 100e-3, aperture: float = 105e-3, num_zones: int = 20):
        self.radius = radius
        self.aperture_diameter = aperture
        self.aperture_radius = aperture / 2.0
        self.num_zones = num_zones
        self.zones = self._discretize_zones()

    def _discretize_zones(self) -> list[dict]:
        n_inner = self.num_zones // 2
        n_edge = self.num_zones - n_inner
        r_outer_start = 0.8 * self.aperture_radius
        r_inner = np.linspace(0.0, r_outer_start, n_inner, endpoint=False)
        r_edge = np.linspace(r_outer_start, self.aperture_radius, n_edge, endpoint=True)
        r_all = np.concatenate([r_inner, r_edge])
        zones = []
        for r in r_all:
            sin_val = min(r / self.radius, 1.0 - 1e-9)
            theta_rad = np.arcsin(sin_val)
            theta_dep = np.radians(30)
            thickness_bias = np.cos(theta_dep) / (np.cos(theta_rad + theta_dep) + 1e-30)
            thickness_bias = float(np.clip(thickness_bias, 1.0, 1.1))
            zones.append({
                "r": r,
                "r_mm": r * 1e3,
                "theta_rad": float(theta_rad),
                "theta_deg": float(np.rad2deg(theta_rad)),
                "thickness_bias": thickness_bias
            })
        return zones


# ============================================================
# 2. TMM 传输矩阵求解器
# ============================================================
class TMMSolver:
    def __init__(self, substrate: CurvedSubstrate, db: MaterialDatabase):
        self.substrate = substrate
        self.db = db
        self.lambda0 = db.lambda0

    def reflectance(self, d_list: np.ndarray, material_list: list[str],
                    theta_rad: float, temperature: float = 20.0) -> float:
        n_inc = self.db.get_refractive_index("air", temperature)
        n_sub = self.db.get_refractive_index(self.db.substrate_name, temperature)
        n_layers = [self.db.get_refractive_index(mat, temperature) for mat in material_list]
        n_layers_air_to_sub = list(reversed(n_layers))
        d_air_to_sub = np.asarray(d_list, dtype=float)[::-1]

        sin_theta_inc = float(n_inc.real) * np.sin(theta_rad)
        cos_theta_inc = np.sqrt(1.0 - sin_theta_inc ** 2 + 0j)
        if cos_theta_inc.imag < 0:
            cos_theta_inc = np.conj(cos_theta_inc)

        eta_inc_s = n_inc * cos_theta_inc
        eta_inc_p = n_inc / (cos_theta_inc + 1e-30)

        sin_theta_sub = sin_theta_inc / n_sub
        cos_theta_sub = np.sqrt(1.0 - sin_theta_sub ** 2 + 0j)
        if cos_theta_sub.imag < 0:
            cos_theta_sub = np.conj(cos_theta_sub)
        eta_sub_s = n_sub * cos_theta_sub
        eta_sub_p = n_sub / (cos_theta_sub + 1e-30)

        M_s = np.eye(2, dtype=complex)
        M_p = np.eye(2, dtype=complex)
        k0 = 2 * np.pi / self.lambda0

        for n_mat, d in zip(n_layers_air_to_sub, d_air_to_sub):
            sin_theta_layer = sin_theta_inc / n_mat
            cos_theta_layer = np.sqrt(1.0 - sin_theta_layer ** 2 + 0j)
            if cos_theta_layer.imag < 0:
                cos_theta_layer = np.conj(cos_theta_layer)
            delta = k0 * n_mat * d * cos_theta_layer

            eta_s = n_mat * cos_theta_layer
            eta_p = n_mat / (cos_theta_layer + 1e-30)
            c, s = np.cos(delta), np.sin(delta)

            m_s = [[c, -1j * s / (eta_s + 1e-30)], [-1j * eta_s * s, c]]
            m_p = [[c, -1j * s / (eta_p + 1e-30)], [-1j * eta_p * s, c]]
            M_s = M_s @ m_s
            M_p = M_p @ m_p

        Y_s = (M_s[1, 0] + M_s[1, 1] * eta_sub_s) / (M_s[0, 0] + M_s[0, 1] * eta_sub_s + 1e-30)
        Y_p = (M_p[1, 0] + M_p[1, 1] * eta_sub_p) / (M_p[0, 0] + M_p[0, 1] * eta_sub_p + 1e-30)
        R_s = abs((eta_inc_s - Y_s) / (eta_inc_s + Y_s + 1e-30)) ** 2
        R_p = abs((eta_inc_p - Y_p) / (eta_inc_p + Y_p + 1e-30)) ** 2
        return (R_s + R_p) / 2.0

    # 在 TMMSolver 类中，删除原来的 _reflectance_full 方法，替换为：
    def _reflectance_polarized(self, d_list: np.ndarray, material_list: list[str],
                               theta_rad: float, temperature: float = 20.0) -> dict:
        """返回平均反射率、s/p 偏振反射率，不涉及透过率/吸收。"""
        n_inc = self.db.get_refractive_index("air", temperature)
        n_sub = self.db.get_refractive_index(self.db.substrate_name, temperature)
        n_layers = [self.db.get_refractive_index(mat, temperature) for mat in material_list]
        n_layers_air_to_sub = list(reversed(n_layers))
        d_air_to_sub = np.asarray(d_list, dtype=float)[::-1]

        sin_theta_inc = float(n_inc.real) * np.sin(theta_rad)
        cos_theta_inc = np.sqrt(1.0 - sin_theta_inc ** 2 + 0j)
        if cos_theta_inc.imag < 0:
            cos_theta_inc = np.conj(cos_theta_inc)

        eta_inc_s = n_inc * cos_theta_inc
        eta_inc_p = n_inc / (cos_theta_inc + 1e-30)

        sin_theta_sub = sin_theta_inc / n_sub
        cos_theta_sub = np.sqrt(1.0 - sin_theta_sub ** 2 + 0j)
        if cos_theta_sub.imag < 0:
            cos_theta_sub = np.conj(cos_theta_sub)
        eta_sub_s = n_sub * cos_theta_sub
        eta_sub_p = n_sub / (cos_theta_sub + 1e-30)

        M_s = np.eye(2, dtype=complex)
        M_p = np.eye(2, dtype=complex)
        k0 = 2 * np.pi / self.lambda0

        for n_mat, d in zip(n_layers_air_to_sub, d_air_to_sub):
            sin_theta_layer = sin_theta_inc / n_mat
            cos_theta_layer = np.sqrt(1.0 - sin_theta_layer ** 2 + 0j)
            if cos_theta_layer.imag < 0:
                cos_theta_layer = np.conj(cos_theta_layer)
            delta = k0 * n_mat * d * cos_theta_layer

            eta_s = n_mat * cos_theta_layer
            eta_p = n_mat / (cos_theta_layer + 1e-30)
            c, s = np.cos(delta), np.sin(delta)

            m_s = [[c, -1j * s / (eta_s + 1e-30)], [-1j * eta_s * s, c]]
            m_p = [[c, -1j * s / (eta_p + 1e-30)], [-1j * eta_p * s, c]]
            M_s = M_s @ m_s
            M_p = M_p @ m_p

        # 反射系数（沿用原代码的计算方式，保证与之前一致）
        Y_s = (M_s[1, 0] + M_s[1, 1] * eta_sub_s) / (M_s[0, 0] + M_s[0, 1] * eta_sub_s + 1e-30)
        Y_p = (M_p[1, 0] + M_p[1, 1] * eta_sub_p) / (M_p[0, 0] + M_p[0, 1] * eta_sub_p + 1e-30)
        R_s = abs((eta_inc_s - Y_s) / (eta_inc_s + Y_s + 1e-30)) ** 2
        R_p = abs((eta_inc_p - Y_p) / (eta_inc_p + Y_p + 1e-30)) ** 2
        R_avg = (R_s + R_p) / 2.0

        return {
            'R_avg': float(R_avg),
            'R_s': float(R_s),
            'R_p': float(R_p),
        }

    def evaluate_radial(self, d_list: np.ndarray, material_list: list[str],
                        temperature: float = 20.0) -> pd.DataFrame:
        rows = []
        for zone in self.substrate.zones:
            d_corr = d_list * zone["thickness_bias"]
            res = self._reflectance_polarized(d_corr, material_list, zone["theta_rad"], temperature)
            rows.append({
                "r_mm": zone["r_mm"],
                "theta_deg": zone["theta_deg"],
                "R": res['R_avg'],
                "R_s": res['R_s'],
                "R_p": res['R_p'],
            })
        return pd.DataFrame(rows)

    def radial_area_weights(self, r_mm) -> np.ndarray:
        """同心环面积权重，用于非均匀径向采样的全口径平均。"""
        r = np.asarray(r_mm, dtype=float) * 1e-3
        n = len(r)
        if n == 0:
            return np.array([], dtype=float)
        if n == 1:
            return np.ones(1, dtype=float)

        order = np.argsort(r)
        r_sorted = r[order]
        bounds = np.empty(n + 1, dtype=float)
        bounds[0] = 0.0
        bounds[-1] = self.substrate.aperture_radius
        bounds[1:-1] = 0.5 * (r_sorted[:-1] + r_sorted[1:])
        bounds = np.clip(bounds, 0.0, self.substrate.aperture_radius)

        sorted_weights = np.diff(bounds ** 2)
        sorted_weights = np.maximum(sorted_weights, 0.0)
        total = float(np.sum(sorted_weights))
        if total <= 0:
            sorted_weights = np.full(n, 1.0 / n, dtype=float)
        else:
            sorted_weights /= total

        weights = np.empty_like(sorted_weights)
        weights[order] = sorted_weights
        return weights

    @staticmethod
    def weighted_mean(values, weights) -> float:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if len(values) == 0:
            return 0.0
        return float(np.sum(values * weights) / max(np.sum(weights), 1e-30))

    @classmethod
    def weighted_std(cls, values, weights) -> float:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if len(values) == 0:
            return 0.0
        mean = cls.weighted_mean(values, weights)
        return float(np.sqrt(np.sum(weights * (values - mean) ** 2) /
                             max(np.sum(weights), 1e-30)))

    def calculate_objectives(self, d_list: np.ndarray, material_list: list[str],
                             temperature: float = 20.0) -> dict[str, float]:
        df = self.evaluate_radial(d_list, material_list, temperature)
        R = df["R"].values
        R_s = df["R_s"].values
        R_p = df["R_p"].values
        weights = self.radial_area_weights(df["r_mm"].values)

        mean_R = self.weighted_mean(R, weights)
        avg_R = mean_R * 100
        uniformity = self.weighted_std(R, weights) / max(mean_R, 1e-6) * 100
        theta = df["theta_deg"].values
        mask = theta >= 0.8 * np.max(theta) if len(theta) else []
        high_R = np.max(R[mask]) * 100 if np.any(mask) else np.max(R) * 100
        stress = self.calculate_stress_metrics(d_list, material_list)
        dR = self.calculate_delta_R(d_list, material_list)

        pol_diff = self.weighted_mean(np.abs(R_s - R_p), weights) * 100
        max_pol_diff = np.max(np.abs(R_s - R_p)) * 100  # 最大偏振差

        return {
            "avg_R": avg_R,
            "uniformity": uniformity,
            "high_R": high_R,
            "delta_R": dR,
            "avg_stress": stress.get("avg_stress", 0),
            "max_stress": stress.get("max_stress", 0),
            "pol_diff": pol_diff,
            "max_pol_diff": max_pol_diff,
        }

    def calculate_stress_metrics(self, d_list, mats):
        t = np.array(d_list) * 1e9
        s = np.array([self.db.get_stress(m) for m in mats])
        if len(t) == 0 or np.sum(t) <= 0:
            return {"avg_stress": 0, "max_stress": 0}
        w = t * s
        avg = np.sum(w) / np.sum(t)
        cum = np.cumsum(w) / np.maximum(np.cumsum(t), 1e-30)
        return {"avg_stress": float(avg), "max_stress": float(np.max(np.abs(cum)))}

    def calculate_delta_R(self, d_list, mats):
        df20 = self.evaluate_radial(d_list, mats, 20)
        df40 = self.evaluate_radial(d_list, mats, 40)
        weights = self.radial_area_weights(df20["r_mm"].values)
        return self.weighted_mean(np.abs(df40["R"].values - df20["R"].values),
                                  weights) * 100

    def scan_wavelength_angle_map(self, d_list, material_list,
                                  wavelengths, temperature=20.0):
        """
        返回二维反射率矩阵：行为波长，列为径向环带（或入射角）。
        wavelengths : 1D array (m,)
        返回值 :
            R_map : 2D array (m, K)  反射率, 0~1
            theta_deg : 1D array (K,) 各环带对应的入射角
        """
        K = len(self.substrate.zones)
        m = len(wavelengths)
        R_map = np.empty((m, K), dtype=float)
        orig_lambda0 = self.lambda0  # 备份原始工作波长

        for i, lam in enumerate(wavelengths):
            self.lambda0 = lam  # 临时修改求解器波长
            df = self.evaluate_radial(d_list, material_list, temperature)
            R_map[i, :] = df['R'].values  # 0~1

        self.lambda0 = orig_lambda0  # 恢复
        return R_map, np.array([z['theta_deg'] for z in self.substrate.zones])

    def monte_carlo_robustness(self, d_list, material_list, n_samples=1000,
                               thickness_sigma_relative=0.01,
                               reflectance_threshold=0.5,
                               std_threshold=0.12,
                               seed=123):
        rng = np.random.default_rng(seed)
        d_nominal = np.array(d_list, dtype=float)
        samples = {
            'avg_R': [],
            'uniformity': [],  # 保留但不用于良率
            'high_R': [],
            'delta_R': [],
            'avg_stress': [],
            'max_stress': [],
            'std_R': [],  # 绝对标准差 (%)
            'PV': []  # 峰谷值 (%)
        }
        for _ in tqdm(range(n_samples), desc="Monte Carlo Robustness"):
            delta = rng.normal(0, thickness_sigma_relative * d_nominal)
            d_perturbed = np.maximum(d_nominal + delta, 1e-11)
            # 原有指标
            perf = self.calculate_objectives(d_perturbed, material_list)
            for key in ['avg_R', 'uniformity', 'high_R', 'delta_R', 'avg_stress', 'max_stress']:
                samples[key].append(perf[key])
            # 新增绝对均匀性
            df = self.evaluate_radial(d_perturbed, material_list, 20.0)
            R_vals = df['R'].values * 100  # 转为 %
            weights = self.radial_area_weights(df["r_mm"].values)
            samples['std_R'].append(self.weighted_std(R_vals, weights))
            samples['PV'].append(np.max(R_vals) - np.min(R_vals))
        # 转为数组
        for key in samples:
            samples[key] = np.array(samples[key])
        # 良率：平均反射率和绝对标准差均低于工艺阈值
        yield_mask = ((samples['avg_R'] < reflectance_threshold) &
                      (samples['std_R'] < std_threshold))
        yield_rate = np.mean(yield_mask) * 100
        return samples, yield_rate


# ============================================================
# 3. 数据集生成器与数据类
# ============================================================
CALC_VERSION = "2026-05-30-area-weighted-v4"
THICKNESS_MIN_NM = 20.0
THICKNESS_MAX_NM = 140.0
MIN_ADJACENT_DIFF = 4.0
MAX_TOTAL_NM = 400.0


def is_valid_material_sequence(mats, aux=None):
    if aux is None: aux = ["AlF3", "YF3", "GdF3", "Al2O3"]
    if not (7 <= len(mats) <= 9): return False
    for a, b in zip(mats, mats[1:]):
        if a == b: return False
    if mats.count("AlF3") > 1 or (mats.count("AlF3") == 1 and mats[-1] != "AlF3"):
        return False
    if mats.count("Al2O3") > 1 or (mats.count("Al2O3") == 1 and mats[0] != "Al2O3"):
        return False
    return sum(1 for m in mats if m in aux) <= 2


def is_valid_thicknesses(t, max_total=MAX_TOTAL_NM):
    t = np.asarray(t)
    if not (7 <= len(t) <= 9): return False
    if np.any(t < 20 - 1e-9) or np.any(t > 140 + 1e-9): return False
    if len(t) > 1 and np.any(np.abs(np.diff(t)) < 4 - 1e-9): return False
    return np.sum(t) <= max_total + 1


def is_valid_design(t, mats, aux=None):
    return len(t) == len(mats) and is_valid_material_sequence(mats, aux) and is_valid_thicknesses(t)


def dataset_needs_regeneration(path, calc_version):
    if not os.path.exists(path):
        return True, 0
    try:
        version_df = pd.read_csv(path, usecols=["ver"])
    except Exception:
        try:
            return True, len(pd.read_csv(path))
        except Exception:
            return True, 0
    if version_df.empty:
        return True, 0
    stale = not (version_df["ver"].astype(str) == calc_version).all()
    return stale, len(version_df)


def generate_all_strict_templates(aux):
    base = ["LaF3", "MgF2", "LaF3", "MgF2", "LaF3", "MgF2", "LaF3"]
    valid = {tuple(base)}
    for i in range(7):
        for m1 in aux:
            s = base.copy()
            s[i] = m1
            if is_valid_material_sequence(s, aux): valid.add(tuple(s))
            for j in range(i + 1, 7):
                for m2 in aux:
                    s2 = s.copy()
                    s2[j] = m2
                    if is_valid_material_sequence(s2, aux): valid.add(tuple(s2))
    return [list(x) for x in sorted(valid)]


class DatasetGenerator:
    def __init__(self, substrate, db):
        self.sub = substrate
        self.db = db
        self.solver = TMMSolver(substrate, db)
        self.aux = list(db.auxiliary.keys())
        self.templates = generate_all_strict_templates(self.aux)

    def random_valid_design(self, max_attempts=2000):
        for _ in range(max_attempts):
            mats = random.choice(self.templates).copy()
            L = len(mats)
            t = []
            ok = True
            for i in range(L):
                if i == 0:
                    low, high = 20, 140
                else:
                    last = t[-1]
                    pool = []
                    if last - 4 >= 20: pool.append((20, last - 4))
                    if last + 4 <= 140: pool.append((last + 4, 140))
                    if not pool:
                        ok = False
                        break
                    low, high = random.choice(pool)
                t.append(random.uniform(low, high))
            if ok and np.sum(t) <= MAX_TOTAL_NM:
                return np.array(t) * 1e-9, mats
        d = np.array([34.6, 54.5, 34.6, 54.5, 34.6, 54.5, 34.6]) * 1e-9
        return d, ["LaF3", "MgF2", "LaF3", "MgF2", "LaF3", "MgF2", "LaF3"]

    def compute(self, d, mats):
        return self.solver.calculate_objectives(d, mats)

    def generate(self, N, path="coating_dataset.csv"):
        data = []
        for _ in tqdm(range(N), desc="Generating Dataset"):
            d, m = self.random_valid_design()
            p = self.compute(d, m)
            data.append({
                "L": len(m), "materials": "|".join(m),
                "thickness_nm": ",".join([f"{x:.2f}" for x in d * 1e9]),
                "ver": CALC_VERSION, **p
            })
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        return df


class CoatingDataset(Dataset):
    def __init__(self, path, max_layers=9):
        self.df = pd.read_csv(path)
        self.maxL = max_layers
        self.mats = ["MgF2", "LaF3", "AlF3", "YF3", "Al2O3", "GdF3"]
        self.idx = {m: i for i, m in enumerate(self.mats)}
        self.nmat = len(self.mats)
        self.ys_names = ["avg_R", "uniformity", "high_R", "delta_R", "avg_stress", "pol_diff"]

        Xm, Xt = [], []
        self.raw_y = self.df[self.ys_names].values.astype(np.float32)

        for _, r in self.df.iterrows():
            ms = str(r["materials"]).split("|")
            ts = str(r["thickness_nm"]).split(",")
            mvec = np.zeros(self.maxL * self.nmat, np.float32)
            tvec = np.zeros(self.maxL, np.float32)
            for j in range(min(len(ms), self.maxL)):
                if ms[j] in self.idx:
                    mvec[j * self.nmat + self.idx[ms[j]]] = 1
                    tvec[j] = float(ts[j]) / 200
            Xm.append(mvec)
            Xt.append(tvec)

        self.Xm = np.array(Xm)
        self.Xt = np.array(Xt)

        # 运行时由外部映射控制
        self.Xts = np.zeros_like(self.Xt)
        self.ys = np.zeros_like(self.raw_y)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x = np.empty(self.maxL * (self.nmat + 1), np.float32)
        for j in range(self.maxL):
            o = j * (self.nmat + 1)
            x[o:o + self.nmat] = self.Xm[i, j * self.nmat:(j + 1) * self.nmat]
            x[o + self.nmat] = self.Xts[i, j]
        return torch.tensor(x), torch.tensor(self.ys[i])


class Surrogate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x): return self.net(x)


def load_torch_weights(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train_surrogate(model, train_loader, val_loader, epochs=100, device="cpu",
                    model_path="best_surrogate.pth"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    loss_fn = nn.MSELoss()
    best, pat = 1e9, 0

    train_history = []
    val_history = []

    for e in range(epochs):
        model.train()
        tl = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            l = loss_fn(model(x), y)
            l.backward()
            opt.step()
            tl += l.item() * x.shape[0]
        tl /= len(train_loader.dataset)
        train_history.append(tl)

        model.eval()
        vl = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vl += loss_fn(model(x), y).item() * x.shape[0]
        vl /= len(val_loader.dataset)
        val_history.append(vl)
        sch.step(vl)

        if vl < best:
            best = vl
            pat = 0
            torch.save(model.state_dict(), model_path)
        else:
            pat += 1
            if pat >= 20: break
        print(f"Epoch {e + 1:3d} | Train Loss: {tl:.2e} | Val Loss: {vl:.2e}")

    model.load_state_dict(load_torch_weights(model_path, map_location=device))
    return train_history, val_history


# ============================================================
# 4. 科学级可视化模块（work.py 风格改造）
# ============================================================
class JournalFigureConfig:
    """图片尺寸与全局学术风格参数。"""
    FIGSIZES = {
        "single": (9.6, 5.8),
        "1.5col": (12.0, 6.8),
        "double": (15.5, 8.2),
        "panel": (15.0, 9.2),
    }

    @staticmethod
    def apply_style():
        """应用统一的论文图风格。"""
        import matplotlib.font_manager as fm

        available_fonts = {f.name for f in fm.fontManager.ttflist}
        font_stack = [
            f for f in ["Times New Roman", "Arial", "Microsoft YaHei",
                        "SimHei", "DejaVu Sans"]
            if f in available_fonts
        ] or ["DejaVu Sans"]

        plt.rcParams.update({
            'text.usetex': False,
            'font.family': font_stack,
            'mathtext.fontset': 'stix',
            'axes.labelsize': 13,
            'axes.titlesize': 15,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'bold',
            'axes.edgecolor': '#334155',
            'axes.linewidth': 1.0,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'xtick.color': '#334155',
            'ytick.color': '#334155',
            'legend.fontsize': 10.5,
            'figure.dpi': 320,
            'savefig.dpi': 320,
            'savefig.bbox': 'tight',
            'savefig.format': 'png',
            'savefig.facecolor': 'white',
            'axes.unicode_minus': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'figure.figsize': JournalFigureConfig.FIGSIZES["single"],
            'axes.grid': False,
            'lines.solid_capstyle': 'round',
            'lines.solid_joinstyle': 'round',
        })

    @staticmethod
    def get_figsize(layout="single"):
        """返回宽 > 高的图形尺寸（单位：英寸）。"""
        return JournalFigureConfig.FIGSIZES.get(
            layout, JournalFigureConfig.FIGSIZES["single"]
        )


class ScientificPalette:
    """色盲友好、适合论文展示的低饱和配色。"""
    COLORS = {
        'paper': '#FFFFFF',
        'panel': '#F6F8FB',
        'ink': '#202124',
        'muted_ink': '#64748B',
        'axis': '#334155',
        'grid': '#D8DEE9',
        'light_gray': '#E7EBF2',
        'primary_blue': '#2F6BFF',
        'dark_blue': '#243B53',
        'soft_blue': '#EAF2FF',
        'cyan': '#14A3C7',
        'sage_green': '#2E7D6B',
        'mint': '#DFF3EE',
        'burnt_orange': '#D76B2A',
        'gold': '#D99A2B',
        'soft_red': '#C43B4A',
        'wine_red': '#8B2D43',
        'violet': '#6D5BD0',
        'white': '#FFFFFF',
        'black': '#202124',
    }

    MATERIAL_COLORS = {
        "MgF2": "#259D8E",
        "LaF3": "#315E7E",
        "AlF3": "#D9A441",
        "YF3": "#C86F4A",
        "GdF3": "#7467C8",
        "Al2O3": "#6D8A3A",
    }

    REFLECTANCE_CMAP = LinearSegmentedColormap.from_list(
        "reflectance_modern",
        ["#23395B", "#1F7A8C", "#2EA58D", "#F2C14E", "#C43B4A"]
    )


class ScienceVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        JournalFigureConfig.apply_style()
        self.c = ScientificPalette.COLORS
        self.mat_color = ScientificPalette.MATERIAL_COLORS

    def _save_figure(self, fig, name, formats=('png', 'pdf')):
        for fmt in formats:
            path = os.path.join(self.output_dir, f"{name}.{fmt}")
            fig.savefig(path, dpi=320, bbox_inches='tight', pad_inches=0.06,
                        facecolor=fig.get_facecolor())
        plt.close(fig)

    def _style_axis(self, ax, grid='y'):
        ax.set_axisbelow(True)
        if grid:
            ax.grid(True, axis=grid, color=self.c['grid'], linestyle='-',
                    linewidth=0.85, alpha=0.85)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(self.c['axis'])
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(axis='both', which='major', length=4, width=1.0,
                       color=self.c['axis'], labelcolor=self.c['axis'])

    def _legend(self, ax, **kwargs):
        leg = ax.legend(frameon=True, fancybox=False, framealpha=1.0,
                        edgecolor=self.c['grid'], facecolor=self.c['paper'],
                        borderpad=0.7, handlelength=2.2, **kwargs)
        if leg:
            leg.get_frame().set_linewidth(0.8)
        return leg

    def _stats_box(self, ax, text, x=0.03, y=0.95, ha='left', va='top'):
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                fontsize=10.5, color=self.c['ink'],
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor=self.c['grid'], linewidth=0.8, alpha=0.95))

    def _signed_percent(self, value, digits=4):
        return f"{value:+.{digits}f}%"

    def _text_color_for(self, fill_color):
        red, green, blue = to_rgb(fill_color)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return self.c['ink'] if luminance > 0.58 else self.c['white']

    def _draw_coating_stack(self, ax, d_list, mats, compact=False,
                            show_legend=True):
        thickness_nm = np.asarray(d_list, dtype=float) * 1e9
        total_nm = float(np.sum(thickness_nm))
        side_width = max(34.0, total_nm * 0.11)
        strip_h = 0.82 if compact else 0.96
        y0 = -strip_h / 2
        current_z = 0.0

        ax.add_patch(plt.Rectangle(
            (-side_width, y0), side_width, strip_h,
            facecolor='#EDF1F7', edgecolor=self.c['axis'], linewidth=0.9
        ))
        ax.text(-side_width / 2, 0, "SiO$_2$\nSubstrate",
                ha='center', va='center', fontsize=9 if compact else 10,
                color=self.c['axis'], fontweight='bold')

        for idx, (d_nm, mat) in enumerate(zip(thickness_nm, mats), start=1):
            color = self.mat_color.get(mat, self.c['light_gray'])
            ax.add_patch(plt.Rectangle(
                (current_z, y0), d_nm, strip_h,
                facecolor=color, edgecolor='white', linewidth=1.4
            ))
            label = f"L{idx}\n{mat}" if compact else f"L{idx}\n{mat}\n{d_nm:.1f} nm"
            font_size = 7.0 if d_nm < 34 else 7.6 if compact else 8.5
            ax.text(current_z + d_nm / 2, 0, label,
                    ha='center', va='center', rotation=0,
                    fontsize=font_size,
                    color=self._text_color_for(color), fontweight='bold',
                    linespacing=0.95)
            current_z += d_nm

        ax.add_patch(plt.Rectangle(
            (current_z, y0), side_width, strip_h,
            facecolor='white', edgecolor=self.c['axis'], linewidth=0.9
        ))
        ax.text(current_z + side_width / 2, 0, "Air",
                ha='center', va='center', fontsize=9 if compact else 10,
                color=self.c['axis'], fontweight='bold')

        ax.set_xlim(-side_width * 1.08, current_z + side_width * 1.08)
        ax.set_ylim(-0.82 if show_legend else -0.66,
                    0.98 if show_legend else 0.72)
        ax.axis('off')
        ax.text(0.5, 0.92, f"Total optical stack: {total_nm:.1f} nm",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10.5 if compact else 12, color=self.c['muted_ink'],
                fontweight='bold')

        if show_legend:
            handles = [
                Patch(facecolor=self.mat_color[m], edgecolor='none', label=m)
                for m in dict.fromkeys(mats)
            ]
            leg = ax.legend(handles=handles, loc='lower center',
                            bbox_to_anchor=(0.5, -0.08), ncol=len(handles),
                            frameon=True, fancybox=False, framealpha=1.0,
                            edgecolor=self.c['grid'], facecolor='white',
                            fontsize=10)
            leg.get_frame().set_linewidth(0.8)

    # ---------- 新增：基底几何与沉积偏差图 ----------
    def plot_substrate_incidence_angle(self, substrate: CurvedSubstrate,
                                       save_name="substrate_incidence"):
        """径向入射角分布图。"""
        r_mm = np.array([z["r_mm"] for z in substrate.zones])
        theta_deg = np.array([z["theta_deg"] for z in substrate.zones])

        fig, ax = plt.subplots(figsize=JournalFigureConfig.get_figsize("single"))
        ax.fill_between(r_mm, 0, theta_deg, color=self.c['soft_blue'],
                        alpha=0.85, zorder=1)
        ax.plot(r_mm, theta_deg, color=self.c['primary_blue'], linewidth=2.8,
                zorder=3, label=r"Local incidence angle $\theta(r)$")
        ax.scatter(r_mm, theta_deg, s=46, facecolor='white',
                   edgecolor=self.c['primary_blue'], linewidth=1.6, zorder=4)
        ax.annotate(f"Edge: {theta_deg[-1]:.1f}°",
                    xy=(r_mm[-1], theta_deg[-1]), xytext=(-64, 26),
                    textcoords='offset points', color=self.c['axis'],
                    fontsize=10.5, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=self.c['axis'],
                                    linewidth=1.0))
        ax.set_xlabel("Radial Position (mm)", fontweight="bold")
        ax.set_ylabel("Incident Angle (°)", fontweight="bold")
        ax.set_title("Incident Angle Distribution on Spherical Substrate",
                     fontweight="bold")
        ax.set_ylim(bottom=0)
        self._style_axis(ax, grid='both')
        self._legend(ax, loc="upper left")
        self._save_figure(fig, save_name)

    def plot_thickness_deviation(self, substrate: CurvedSubstrate,
                                 save_name="thickness_deviation"):
        """行星沉积膜厚偏差系数图"""
        r_mm = np.array([z["r_mm"] for z in substrate.zones])
        bias = np.array([z["thickness_bias"] for z in substrate.zones])

        fig, ax = plt.subplots(figsize=JournalFigureConfig.get_figsize("single"))
        ax.axhline(y=1.0, color=self.c['muted_ink'], linewidth=1.2,
                   linestyle='--', label="Center reference ($\\alpha=1$)")
        ax.fill_between(r_mm, 1.0, bias, color=self.c['mint'], alpha=0.95)
        ax.plot(r_mm, bias, color=self.c['sage_green'], linewidth=2.8,
                label="Thickness bias $\\alpha(r)$")
        ax.scatter(r_mm, bias, s=44, facecolor='white',
                   edgecolor=self.c['sage_green'], linewidth=1.5, zorder=4)
        ax.set_xlabel("Radial Position (mm)", fontweight="bold")
        ax.set_ylabel("Thickness Bias Factor $\\alpha(r)$", fontweight="bold")
        ax.set_title("Deposition Thickness Deviation on Curved Substrate",
                     fontweight="bold")
        self._style_axis(ax, grid='both')
        self._legend(ax, loc="upper left")
        self._stats_box(
            ax,
            f"Min: {bias.min():.3f}\nMax: {bias.max():.3f}\n"
            f"Edge gain: {(bias[-1] - 1) * 100:.1f}%",
            x=0.97, y=0.07, ha='right', va='bottom'
        )
        self._save_figure(fig, save_name)

    # ---------- 原有绘图函数（work.py 风格改造） ----------
    def plot_loss_curve(self, train_history, val_history, save_name="loss_curve"):
        fig, ax = plt.subplots(figsize=JournalFigureConfig.get_figsize("single"))
        epochs = np.arange(1, len(train_history) + 1)
        ax.plot(epochs, train_history, label='Train Loss',
                color=self.c['primary_blue'], lw=2.7)
        ax.plot(epochs, val_history, label='Validation Loss',
                color=self.c['burnt_orange'], lw=2.7, ls='--')
        ax.set_yscale('log')
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("MSE Loss (Log Scale)", fontweight="bold")
        ax.set_title("Surrogate Model Training History", fontweight="bold")
        self._style_axis(ax, grid='both')
        self._legend(ax, loc="upper right")
        self._save_figure(fig, save_name)

    def plot_training_diagnostics(self, train_history, val_history,
                                  save_name="training_diagnostics"):
        train_history = np.asarray(train_history, dtype=float)
        val_history = np.asarray(val_history, dtype=float)
        n = min(len(train_history), len(val_history))
        train_history = train_history[:n]
        val_history = val_history[:n]
        epochs = np.arange(1, n + 1)
        if n == 0:
            return

        best_idx = int(np.argmin(val_history))
        gap = val_history - train_history
        gap_ratio = gap / np.maximum(train_history, 1e-12) * 100
        val_improvement = np.r_[0.0, -np.diff(val_history)]
        relative_improvement = (
            val_improvement / np.maximum(np.r_[val_history[0], val_history[:-1]], 1e-12) * 100
        )

        fig, axes = plt.subplots(
            2, 2, figsize=JournalFigureConfig.get_figsize("panel"),
            constrained_layout=True
        )
        ax1, ax2, ax3, ax4 = axes.flat

        ax1.plot(epochs, train_history, color=self.c['primary_blue'],
                 lw=2.6, label="Train")
        ax1.plot(epochs, val_history, color=self.c['burnt_orange'],
                 lw=2.6, label="Validation")
        ax1.scatter(epochs[best_idx], val_history[best_idx], s=74,
                    facecolor='white', edgecolor=self.c['soft_red'],
                    linewidth=1.7, zorder=4, label="Best validation")
        ax1.set_yscale("log")
        ax1.set_title("(a) Loss Trajectory", fontweight="bold")
        ax1.set_xlabel("Epoch", fontweight="bold")
        ax1.set_ylabel("MSE Loss", fontweight="bold")
        self._style_axis(ax1, grid='both')
        self._legend(ax1, loc="upper right")

        ax2.plot(epochs, val_history, color=self.c['burnt_orange'], lw=2.4)
        ax2.fill_between(epochs, val_history.min(), val_history,
                         color='#FDE9D8', alpha=0.95)
        ax2.axvline(epochs[best_idx], color=self.c['soft_red'],
                    lw=1.4, ls='--')
        self._stats_box(
            ax2,
            f"Best epoch: {epochs[best_idx]}\n"
            f"Best val: {val_history[best_idx]:.2e}",
            x=0.97, y=0.95, ha='right', va='top'
        )
        ax2.set_title("(b) Validation Minimum", fontweight="bold")
        ax2.set_xlabel("Epoch", fontweight="bold")
        ax2.set_ylabel("Validation Loss", fontweight="bold")
        self._style_axis(ax2, grid='both')

        colors = [self.c['soft_red'] if g > 0 else self.c['sage_green']
                  for g in gap_ratio]
        ax3.bar(epochs, gap_ratio, color=colors, edgecolor='white',
                linewidth=0.8, alpha=0.92)
        ax3.axhline(0, color=self.c['axis'], lw=1.1)
        ax3.set_title("(c) Generalization Gap", fontweight="bold")
        ax3.set_xlabel("Epoch", fontweight="bold")
        ax3.set_ylabel(r"$(Val - Train) / Train$ (%)", fontweight="bold")
        self._style_axis(ax3, grid='y')

        ax4.bar(epochs, relative_improvement, color=self.c['violet'],
                alpha=0.88, edgecolor='white', linewidth=0.8)
        ax4.axhline(0, color=self.c['axis'], lw=1.1)
        self._stats_box(
            ax4,
            f"Final train: {train_history[-1]:.2e}\n"
            f"Final val: {val_history[-1]:.2e}",
            x=0.97, y=0.95, ha='right', va='top'
        )
        ax4.set_title("(d) Validation Improvement Rate", fontweight="bold")
        ax4.set_xlabel("Epoch", fontweight="bold")
        ax4.set_ylabel("Loss reduction (%)", fontweight="bold")
        self._style_axis(ax4, grid='y')

        plt.suptitle("Surrogate Neural Network Training Diagnostics",
                     fontweight='bold', fontsize=18)
        self._save_figure(fig, save_name)

    def plot_coating_structure(self, d_list, mats, save_name="coating_structure"):
        fig, ax = plt.subplots(figsize=(12.0, 3.9))
        self._draw_coating_stack(ax, d_list, mats, compact=False,
                                 show_legend=True)
        ax.set_title("Multilayer Coating Structure", fontweight='bold', pad=12)
        self._save_figure(fig, save_name)

    def plot_radial_reflectance(self, tmm_solver, d_list, mats,
                                save_name="radial_reflectance"):
        df = tmm_solver.evaluate_radial(d_list, mats, 20.0)
        r = df['r_mm'].values
        R = df['R'].values * 100

        fig, ax = plt.subplots(figsize=JournalFigureConfig.get_figsize("single"))
        weights = tmm_solver.radial_area_weights(r)
        mean_R = tmm_solver.weighted_mean(R, weights)
        std_R = tmm_solver.weighted_std(R, weights)
        ax.axhspan(mean_R - std_R, mean_R + std_R,
                   color=self.c['soft_blue'], alpha=0.95,
                   label=r"Area-weighted $\mu \pm \sigma$")
        ax.axhline(mean_R, color=self.c['dark_blue'], lw=1.4,
                   linestyle='--', label=f"Weighted mean = {mean_R:.3f}%")
        ax.plot(r, R, color=self.c['primary_blue'], lw=2.8, zorder=3)
        ax.scatter(r, R, s=48, facecolor='white',
                   edgecolor=self.c['primary_blue'], linewidth=1.5, zorder=4)
        ax.set_xlabel("Radial Position $r$ (mm)", fontweight="bold")
        ax.set_ylabel("Reflectance $R$ (%)", fontweight="bold")
        ax.set_title("Radial Reflectance at 193 nm", fontweight="bold")
        self._style_axis(ax, grid='both')
        self._legend(ax, loc="lower right")
        stats = (f"$n$={len(r)}\nArea $\\mu$={mean_R:.3f}%\n"
                 f"$\\sigma$={std_R:.4f}%\nPV={np.ptp(R):.4f}%")
        self._stats_box(ax, stats, x=0.97, y=0.20, ha='right', va='bottom')
        self._save_figure(fig, save_name)

    def plot_polarization_radial(self, tmm_solver, d_list, mats,
                                 save_name="polarization_radial"):
        df = tmm_solver.evaluate_radial(d_list, mats, 20.0)
        r = df['r_mm'].values
        R_s = df['R_s'].values * 100
        R_p = df['R_p'].values * 100
        pol_diff = np.abs(R_s - R_p)

        fig, ax1 = plt.subplots(figsize=JournalFigureConfig.get_figsize("single"))
        ax1.fill_between(r, R_s, R_p, color=self.c['violet'], alpha=0.12,
                         label='s/p separation')
        ax1.plot(r, R_s, color=self.c['primary_blue'], lw=2.7,
                 label='$R_s$ (s-pol)')
        ax1.plot(r, R_p, color=self.c['soft_red'], lw=2.7,
                 label='$R_p$ (p-pol)')
        ax1.scatter(r, R_s, s=34, facecolor='white',
                    edgecolor=self.c['primary_blue'], linewidth=1.2, zorder=4)
        ax1.scatter(r, R_p, s=34, facecolor='white',
                    edgecolor=self.c['soft_red'], linewidth=1.2, zorder=4)
        ax1.set_xlabel("Radial Position $r$ (mm)", fontweight="bold")
        ax1.set_ylabel("Reflectance (%)", fontweight="bold")
        self._style_axis(ax1, grid='both')

        ax2 = ax1.twinx()
        ax2.fill_between(r, 0, pol_diff, color=self.c['mint'], alpha=0.85)
        ax2.plot(r, pol_diff, color=self.c['sage_green'], lw=2.1,
                 linestyle='-.', label='$|R_s - R_p|$')
        ax2.set_ylabel(r'$|R_s - R_p|$ (%)', color=self.c['sage_green'],
                       fontweight="bold")
        ax2.tick_params(axis='y', labelcolor=self.c['sage_green'])
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color(self.c['sage_green'])
        ax2.spines['top'].set_visible(False)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                   frameon=True, fancybox=False, framealpha=1.0,
                   edgecolor=self.c['grid'], facecolor='white')
        ax1.set_title("Radial Polarization-Resolved Reflectance",
                      fontweight="bold")
        self._save_figure(fig, save_name)

    def plot_tornado_sensitivity(self, tmm_solver, d_list, mats,
                                 save_name="sensitivity_tornado"):
        base = tmm_solver.calculate_objectives(d_list, mats, 20)['avg_R']
        res = []
        for i, (d, mat) in enumerate(zip(d_list, mats)):
            d_neg, d_pos = d * 0.95, d * 1.05
            d_mod_neg = np.copy(d_list)
            d_mod_neg[i] = d_neg
            r_neg = tmm_solver.calculate_objectives(d_mod_neg, mats)['avg_R']
            d_mod_pos = np.copy(d_list)
            d_mod_pos[i] = d_pos
            r_pos = tmm_solver.calculate_objectives(d_mod_pos, mats)['avg_R']
            res.append({"layer": i + 1, "mat": mat, "min": r_neg - base,
                        "max": r_pos - base})

        df = pd.DataFrame(res)
        df['range'] = abs(df['max'] - df['min'])
        df = df.sort_values('range', ascending=True)

        fig, ax = plt.subplots(figsize=JournalFigureConfig.get_figsize("1.5col"))
        y_pos = np.arange(len(df))
        labels = [f"L{r['layer']} {r['mat']}" for _, r in df.iterrows()]

        min_val, max_val = df[['min', 'max']].min().min(), df[['min', 'max']].max().max()
        val_range = max_val - min_val if max_val != min_val else 1.0
        ax.set_xlim(min_val - 0.48 * val_range, max_val + 0.48 * val_range)
        x_left, x_right = ax.get_xlim()
        label_left = x_left + 0.035 * (x_right - x_left)
        label_right = x_right - 0.035 * (x_right - x_left)

        for i, (_, row) in enumerate(df.iterrows()):
            low, high = sorted([row['min'], row['max']])
            if low < 0:
                ax.barh(i, min(high, 0) - low, left=low, height=0.58,
                        color=self.c['sage_green'], edgecolor='white',
                        linewidth=1.0)
            if high > 0:
                ax.barh(i, high - max(low, 0), left=max(low, 0), height=0.58,
                        color=self.c['soft_red'], edgecolor='white',
                        linewidth=1.0)
            ax.scatter([row['min'], row['max']], [i, i], s=36,
                       facecolor='white', edgecolor=self.c['axis'],
                       linewidth=1.0, zorder=4)
            left_value = float(low)
            right_value = float(high)
            ax.text(label_left, i, self._signed_percent(left_value),
                    ha='left', va='center', fontsize=9.3,
                    color=self.c['axis'], fontweight='bold')
            ax.text(label_right, i, self._signed_percent(right_value),
                    ha='right', va='center', fontsize=9.3,
                    color=self.c['axis'], fontweight='bold')

        ax.axvline(0, color=self.c['axis'], ls='--', lw=1.2, alpha=0.7)
        ax.text(label_left, len(df) - 0.32, "Lower $\\Delta R$",
                ha='left', va='bottom', fontsize=9.4,
                color=self.c['muted_ink'], fontweight='bold')
        ax.text(label_right, len(df) - 0.32, "Higher $\\Delta R$",
                ha='right', va='bottom', fontsize=9.4,
                color=self.c['muted_ink'], fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10.5, fontweight='bold')
        ax.set_ylim(-0.65, len(df) - 0.02)
        ax.set_xlabel(r"$\Delta R_{avg}$ (%)", fontweight="bold")
        ax.set_title("Thickness Sensitivity (±5%)", fontweight="bold")
        self._style_axis(ax, grid='x')
        self._save_figure(fig, save_name)

    def plot_reflectance_angle_map(self, tmm_solver, d_list, mats,
                                   wavelengths=None,
                                   save_name="angle_reflectance_map"):
        if wavelengths is None:
            # 默认扫描 191 nm ~ 195 nm，步长 0.1 nm
            wavelengths = np.linspace(191e-9, 195e-9, 41)  # 41 个波长点

        R_map, theta_deg = tmm_solver.scan_wavelength_angle_map(
            d_list, mats, wavelengths, temperature=20.0
        )
        R_map_pct = R_map * 100.0  # 转换为 %

        # 创建网格 (X 轴为入射角，Y 轴为波长)
        theta_grid, lam_grid = np.meshgrid(theta_deg, wavelengths * 1e9)  # 波长单位 nm

        fig, ax = plt.subplots(figsize=JournalFigureConfig.get_figsize("1.5col"))
        cf = ax.contourf(theta_grid, lam_grid, R_map_pct,
                         levels=30, cmap=ScientificPalette.REFLECTANCE_CMAP,
                         alpha=0.92)
        # 叠加几条等值线
        contours = ax.contour(theta_grid, lam_grid, R_map_pct,
                              levels=6, colors=self.c['ink'], linewidths=0.8)
        ax.clabel(contours, inline=True, fontsize=10, fmt='%.2f%%')
        cbar = fig.colorbar(cf, ax=ax, pad=0.025)
        cbar.set_label("Reflectance (%)", fontsize=12, fontweight="bold")

        ax.set_xlabel("Incident Angle $\\theta$ (deg)", fontweight="bold")
        ax.set_ylabel("Wavelength $\\lambda$ (nm)", fontweight="bold")
        ax.set_title("Angle‑Wavelength 2D Reflectance Map", fontweight="bold")
        ax.set_xlim(theta_deg.min(), theta_deg.max())
        ax.set_ylim(wavelengths[0] * 1e9, wavelengths[-1] * 1e9)
        self._style_axis(ax, grid='both')
        self._save_figure(fig, save_name)



    def plot_monte_carlo_yield(self, samples, nominal_perf, yield_rate,
                               thickness_sigma_relative=0.015,
                               reflectance_threshold=0.5,
                               std_threshold=0.12,
                               save_name="monte_carlo_yield"):
        fig, ax = plt.subplots(figsize=JournalFigureConfig.get_figsize("single"))
        avg_R_vals = samples['avg_R']
        n_samples = len(avg_R_vals)

        ax.hist(avg_R_vals, bins=40, color=self.c['primary_blue'],
                alpha=0.88, edgecolor='white', linewidth=0.9)
        nominal = nominal_perf['avg_R']
        ax.axvline(nominal, color=self.c['soft_red'], lw=2.5, ls='--',
                   label=f'Nominal $R_{{avg}}$ = {nominal:.3f}%')
        ax.axvline(reflectance_threshold, color=self.c['wine_red'], lw=2.0,
                   ls=':', label=fr'Yield threshold ($R<{reflectance_threshold:g}$%)')

        mean_val = np.mean(avg_R_vals)
        std_val = np.std(avg_R_vals)
        stats_str = (f"Samples: {n_samples}\n"
                     f"Mean $\\pm$ $\\sigma$: {mean_val:.3f} $\\pm$ {std_val:.3f}%\n"
                     fr"Yield ($R<{reflectance_threshold:g}$%, "
                     fr"$\sigma_R<{std_threshold:g}$%): {yield_rate:.1f}%")
        self._stats_box(ax, stats_str, x=0.97, y=0.95, ha='right', va='top')

        ax.set_xlabel('Average Reflectance $R_{avg}$ (%)', fontweight="bold")
        ax.set_ylabel('Frequency', fontweight="bold")
        ax.set_title('Monte Carlo Robustness: Layer Thickness Perturbation '
                     f'($\\sigma={thickness_sigma_relative * 100:.1f}\\%$)',
                     fontweight="bold")
        self._style_axis(ax, grid='y')
        self._legend(ax, loc='upper left')
        self._save_figure(fig, save_name)

    def plot_validation_refractive_index(self,
                                         save_name="validation_refractive_index"):
        c_paper = '#1A5276'
        c_range = '#E6B0AA'
        c_lit_mid = '#B03A2E'
        c_grid = '#EBEDEF'

        fig, ax = plt.subplots(figsize=(6.8, 4.4))

        n_data = {
            r'$\mathrm{LaF_3}$': {
                'paper': 1.72, 'lit_lo': 1.670, 'lit_hi': 1.720,
                'src': 'Ristau 2002; Bischoff 2008; C. Liu 2016',
                'dx': -0.015
            },
            r'$\mathrm{MgF_2}$': {
                'paper': 1.44, 'lit_lo': 1.415, 'lit_hi': 1.440,
                'src': 'Ristau 2002; Guo 2013',
                'dx': 0.010
            },
            r'$\mathrm{GdF_3}$': {
                'paper': 1.70, 'lit_lo': 1.650, 'lit_hi': 1.710,
                'src': 'Liu 2006; Xue 2009; B. Liu 2016',
                'dx': 0.000
            },
            r'$\mathrm{AlF_3}$': {
                'paper': 1.40, 'lit_lo': 1.380, 'lit_hi': 1.430,
                'src': 'Xue 2010; B. Liu 2016',
                'dx': 0.025
            },
        }

        mats = list(n_data.keys())
        y = np.arange(len(mats))

        for i, m in enumerate(mats):
            d = n_data[m]
            ax.plot([d['lit_lo'], d['lit_hi']], [i, i],
                    color=c_range, linewidth=4.5, solid_capstyle='round',
                    zorder=2, label='Literature range' if i == 0 else "")
            mid = (d['lit_lo'] + d['lit_hi']) / 2
            ax.plot(mid, i, 'o', color=c_lit_mid, markersize=5.5,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=3,
                    label='Literature midpoint' if i == 0 else "")

        paper_vals = [n_data[m]['paper'] for m in mats]
        ax.plot(paper_vals, y, 'D', color=c_paper, markersize=8.5,
                markeredgecolor='white', markeredgewidth=1.2, zorder=4,
                label='This work (adopted)')

        for i, m in enumerate(mats):
            d = n_data[m]
            ax.text(d['paper'], i + 0.16, f"{d['paper']:.2f}",
                    fontsize=8.5, color=c_paper, fontweight='bold',
                    ha='center', va='bottom')

            lit_str = f"Lit: {d['lit_lo']:.3f}–{d['lit_hi']:.3f} ({d['src']})"
            lit_x = (d['lit_lo'] + d['lit_hi']) / 2 + d.get('dx', 0.0)
            ax.text(lit_x, i - 0.18, lit_str,
                    fontsize=7.8, color=c_lit_mid, ha='center', va='top')

        ax.set_yticks(y)
        ax.set_yticklabels(mats, fontsize=11, fontweight='bold')
        ax.set_xlabel('Refractive index $n$ at 193 nm',
                      fontsize=11, fontweight='bold')
        ax.set_xlim(1.34, 1.78)
        ax.set_ylim(-0.6, len(mats) + 0.15)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax.grid(axis='x', color=c_grid, linewidth=0.8,
                linestyle='--', zorder=0)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(self.c['axis'])
        ax.spines['bottom'].set_color(self.c['axis'])

        ax.legend(loc='upper right', fontsize=8.2,
                  framealpha=0.95, edgecolor='#CCCCCC')
        fig.tight_layout()
        self._save_figure(fig, save_name)

    def plot_validation_stress(self, save_name="validation_stress"):
        c_paper = '#1A5276'
        c_range = '#E6B0AA'
        c_lit_mid = '#B03A2E'
        c_grid = '#EBEDEF'

        fig, ax = plt.subplots(figsize=(7.2, 5.0))

        s_data = {
            r'$\mathrm{Al_2O_3}$': {
                'paper': 500, 'lit_lo': 400, 'lit_hi': 550,
                'src': 'Abadias 2018'
            },
            r'$\mathrm{MgF_2}$': {
                'paper': 300, 'lit_lo': 300, 'lit_hi': 400,
                'src': 'Ristau 2002'
            },
            r'$\mathrm{AlF_3}$': {
                'paper': 100, 'lit_lo': 23, 'lit_hi': 150,
                'src': 'B. Liu 2016; Xue 2010'
            },
            r'$\mathrm{GdF_3}$': {
                'paper': -80, 'lit_lo': -120, 'lit_hi': -35,
                'src': 'B. Liu 2016; Abadias 2018'
            },
            r'$\mathrm{YF_3}$': {
                'paper': -100, 'lit_lo': -150, 'lit_hi': -60,
                'src': 'Abadias 2018'
            },
            r'$\mathrm{LaF_3}$': {
                'paper': -200, 'lit_lo': -300, 'lit_hi': -150,
                'src': 'Abadias 2018'
            },
        }

        mats_s = list(s_data.keys())
        y_s = np.arange(len(mats_s))

        ax.axvspan(-360, 0, color='#EBF5FB', alpha=0.6, zorder=0)
        ax.axvspan(0, 620, color='#FDEDEC', alpha=0.5, zorder=0)
        ax.text(-180, len(mats_s) - 0.35, "Compressive Stress (–)",
                ha='center', fontsize=9, color='#2980B9',
                fontweight='bold', alpha=0.85)
        ax.text(160, len(mats_s) - 0.35, "Tensile Stress (+)",
                ha='center', fontsize=9, color='#C0392B',
                fontweight='bold', alpha=0.85)

        for i, m in enumerate(mats_s):
            d = s_data[m]
            ax.plot([d['lit_lo'], d['lit_hi']], [i, i],
                    color=c_range, linewidth=4.5, solid_capstyle='round',
                    zorder=2, label='Literature range' if i == 0 else "")
            mid = (d['lit_lo'] + d['lit_hi']) / 2
            ax.plot(mid, i, 'o', color=c_lit_mid, markersize=5.5,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=3,
                    label='Literature midpoint' if i == 0 else "")

        paper_s = [s_data[m]['paper'] for m in mats_s]
        ax.plot(paper_s, y_s, 'D', color=c_paper, markersize=8.5,
                markeredgecolor='white', markeredgewidth=1.2, zorder=4,
                label='This work (adopted)')

        for i, m in enumerate(mats_s):
            d = s_data[m]
            sign_p = '+' if d['paper'] > 0 else ''
            ax.text(d['paper'], i + 0.16, f"{sign_p}{d['paper']}",
                    fontsize=8.5, color=c_paper, fontweight='bold',
                    ha='center', va='bottom')

            sign_lo = '+' if d['lit_lo'] > 0 else ''
            sign_hi = '+' if d['lit_hi'] > 0 else ''
            lit_str = (f"Lit: {sign_lo}{d['lit_lo']} to "
                       f"{sign_hi}{d['lit_hi']} ({d['src']})")
            lit_x = (d['lit_lo'] + d['lit_hi']) / 2
            ax.text(lit_x, i - 0.18, lit_str,
                    fontsize=7.6, color=c_lit_mid, ha='center', va='top')

        ax.axvline(0, color=self.c['axis'], linewidth=0.9,
                   linestyle='--', zorder=1)
        ax.set_yticks(y_s)
        ax.set_yticklabels(mats_s, fontsize=11, fontweight='bold')
        ax.set_xlabel('Residual stress (MPa)', fontsize=11, fontweight='bold')
        ax.set_xlim(-350, 600)
        ax.set_ylim(-0.6, len(mats_s) + 0.15)
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))
        ax.grid(axis='x', color=c_grid, linewidth=0.8,
                linestyle='--', zorder=0)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(self.c['axis'])
        ax.spines['bottom'].set_color(self.c['axis'])

        ax.legend(loc='upper right', fontsize=8.2,
                  framealpha=0.95, edgecolor='#CCCCCC')
        fig.tight_layout()
        self._save_figure(fig, save_name)

# ============================================================
# 5. 主优化流程
# ============================================================
def performance_cost(perf_dict):
    score = (perf_dict['avg_R'] * 7.0 +
             perf_dict['uniformity'] * 2.0 +
             perf_dict['high_R'] * 0.5 +
             perf_dict.get('delta_R', 0) * 0.2 +
             perf_dict.get('pol_diff', 0) * 0.2)
    if abs(perf_dict['avg_stress']) > 50.0:
        score += 50.0
    return score


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "deep_ar_optimization_results")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    db = MaterialDatabase()
    substrate = CurvedSubstrate(radius=100e-3, aperture=105e-3, num_zones=20)
    tmm = TMMSolver(substrate, db)
    gen = DatasetGenerator(substrate, db)

    dataset_csv = os.path.join(base_dir, "coating_dataset.csv")
    model_path = os.path.join(base_dir, "best_surrogate.pth")
    history_csv = os.path.join(base_dir, "loss_history.csv")
    scaler_path = os.path.join(base_dir, "surrogate_scalers.pkl")
    dataset_samples = int(os.getenv("COATING_DATASET_SAMPLES", "80000"))  # 默认与正文一致的 8e4 训练样本
    regenerate_dataset, existing_samples = dataset_needs_regeneration(dataset_csv, CALC_VERSION)
    if regenerate_dataset:
        samples_to_generate = max(dataset_samples, existing_samples)
        print(f"\nGenerating {samples_to_generate} samples dataset "
              f"for {CALC_VERSION}...")
        gen.generate(samples_to_generate, dataset_csv)
        for stale_path in [model_path, history_csv, scaler_path]:
            if os.path.exists(stale_path):
                os.remove(stale_path)

    max_layers = 9
    n_materials = 6
    dataset = CoatingDataset(dataset_csv, max_layers=max_layers)

    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        dataset, [train_len, val_len], generator=split_generator
    )

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler_bundle = pickle.load(f)
        scaler_t = scaler_bundle["thickness"]
        scaler_y = scaler_bundle["target"]
    elif os.path.exists(model_path):
        print("\nExisting surrogate has no saved scaler bundle; "
              "using full-dataset scalers for compatibility. "
              "Delete best_surrogate.pth to retrain with train-only scalers.")
        scaler_t = StandardScaler().fit(dataset.Xt)
        scaler_y = StandardScaler().fit(dataset.raw_y)
    else:
        train_indices = np.asarray(train_set.indices, dtype=int)
        scaler_t = StandardScaler().fit(dataset.Xt[train_indices])
        scaler_y = StandardScaler().fit(dataset.raw_y[train_indices])

    # 映射同步回 dataset，供 DataLoader 在运行期正常取用
    dataset.Xts = scaler_t.transform(dataset.Xt)
    dataset.ys = scaler_y.transform(dataset.raw_y)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)

    surrogate = Surrogate(max_layers * (n_materials + 1))

    if not os.path.exists(model_path):
        print("\nTraining surrogate model...")
        if os.path.exists(history_csv):
            os.remove(history_csv)
        train_hist, val_hist = train_surrogate(
            surrogate, train_loader, val_loader, epochs=100, device=device,
            model_path=model_path
        )
        pd.DataFrame({"train_loss": train_hist, "val_loss": val_hist}).to_csv(history_csv, index=False)
        with open(scaler_path, "wb") as f:
            pickle.dump({"thickness": scaler_t, "target": scaler_y}, f)
    else:
        print("\nSurrogate model loaded.")
    surrogate.load_state_dict(load_torch_weights(model_path, map_location=device))
    surrogate.to(device).eval()

    n_candidates = int(os.getenv("COATING_CANDIDATES", "10000"))
    candidates = []
    for _ in tqdm(range(n_candidates), desc="Generating Candidates"):
        d, m = gen.random_valid_design()
        candidates.append((m, d * 1e9))

    print("\nScreening candidates with surrogate model...")
    X = []
    midx = {"MgF2": 0, "LaF3": 1, "AlF3": 2, "YF3": 3, "Al2O3": 4, "GdF3": 5}

    for ms, tn in candidates:
        mv = np.zeros(max_layers * n_materials, np.float32)
        tv = np.zeros(max_layers, np.float32)
        for j in range(len(ms)):
            if ms[j] in midx:
                mv[j * n_materials + midx[ms[j]]] = 1
                tv[j] = tn[j] / 200

        tvs = scaler_t.transform(tv.reshape(1, -1)).flatten()

        x = np.empty(max_layers * (n_materials + 1), np.float32)
        for j in range(max_layers):
            o = j * (n_materials + 1)
            x[o:o + n_materials] = mv[j * n_materials:(j + 1) * n_materials]
            x[o + n_materials] = tvs[j]
        X.append(x)
    X = np.array(X)

    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            bx = torch.tensor(X[i:i + 512]).to(device)
            pred = scaler_y.inverse_transform(surrogate(bx).cpu().numpy())
            for p in pred:
                s = p[0] * 7 + p[1] * 2 + p[2] * 0.5 + p[3] * 0.2 + p[5] * 0.2
                if abs(p[4]) > 50: s += 50
                scores.append(s)

    top_candidates = sorted(zip(candidates, scores), key=lambda x: x[1])[:200]

    print("\nRefining top candidates with TMM + L-BFGS-B...")
    best_design = None
    best_score = 1e9

    for (mats, d_init_nm), _ in tqdm(top_candidates, desc="TMM Refinement"):
        bounds = [(THICKNESS_MIN_NM, THICKNESS_MAX_NM)] * len(d_init_nm)

        def objective(x):
            x = np.asarray(x)
            if np.any(np.abs(np.diff(x)) < MIN_ADJACENT_DIFF - 1e-6): return 1e9
            if np.sum(x) > MAX_TOTAL_NM: return 1e9
            return performance_cost(tmm.calculate_objectives(x * 1e-9, mats))

        res = minimize(objective, d_init_nm, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})

        if res.fun < best_score and is_valid_design(res.x, mats, gen.aux):
            best_score = res.fun
            best_design = (mats, res.x * 1e-9, tmm.calculate_objectives(res.x * 1e-9, mats))

    if best_design is None:
        print("❌ No valid design found during refinement. Falling back to default baseline.")
        best_mats = ["LaF3", "MgF2", "LaF3", "MgF2", "LaF3", "MgF2", "LaF3"]
        best_d = np.array([34.6, 54.5, 34.6, 54.5, 34.6, 54.5, 34.6]) * 1e-9
        best_perf = tmm.calculate_objectives(best_d, best_mats)
    else:
        best_mats, best_d, best_perf = best_design

    # ---- 🔧 新增：计算名义设计的绝对均匀性指标 ----
    df_nom = tmm.evaluate_radial(best_d, best_mats, 20.0)
    R_nom = df_nom['R'].values * 100  # 转换为 %
    nominal_weights = tmm.radial_area_weights(df_nom["r_mm"].values)
    best_perf['std_R'] = tmm.weighted_std(R_nom, nominal_weights)
    best_perf['PV'] = np.max(R_nom) - np.min(R_nom)

    result_df = pd.DataFrame([{
        **{f"layer{i + 1}_material": m for i, m in enumerate(best_mats)},
        **{f"layer{i + 1}_thickness_nm": round(t * 1e9, 1) for i, t in enumerate(best_d)},
        **best_perf, "comprehensive_cost": best_score
    }])
    result_df.to_csv(os.path.join(output_dir, "best_design.csv"), index=False)

    print("\nGenerating publication-quality figures...")
    vis = ScienceVisualizer(figures_dir)
    # ---- 基底模型可视化 ----
    vis.plot_substrate_incidence_angle(substrate, save_name="substrate_incidence")
    vis.plot_thickness_deviation(substrate, save_name="thickness_deviation")
    vis.plot_validation_refractive_index(save_name="validation_refractive_index")
    vis.plot_validation_stress(save_name="validation_stress")
    # ---- 打印最优设计总览 ----
    print("\n" + "=" * 60)
    print("OPTIMAL COATING DESIGN RESULTS")
    print("=" * 60)
    print(f"{'Parameter':<30} {'Value':>15}")
    print("-" * 60)
    for i, (mat, t) in enumerate(zip(best_mats, best_d * 1e9)):
        print(f"  Layer {i+1} ({mat:<6}) thickness       {t:8.2f} nm")
    total_nm = np.sum(best_d * 1e9)
    print(f"  Total thickness                     {total_nm:8.2f} nm")
    print("-" * 60)
    print(f"  avg_R                               {best_perf['avg_R']:8.4f} %")
    print(f"  uniformity (relative)              {best_perf['uniformity']:8.4f} %")
    print(f"  std_R (absolute)                    {best_perf['std_R']:8.4f} %")
    print(f"  PV (absolute)                       {best_perf['PV']:8.4f} %")
    print(f"  high_R (edge)                       {best_perf['high_R']:8.4f} %")
    print(f"  delta_R (20→40°C)                   {best_perf['delta_R']:8.4f} %")
    print(f"  avg_stress                        {best_perf['avg_stress']:8.2f} MPa")
    print(f"  max_stress                        {best_perf['max_stress']:8.2f} MPa")
    print(f"  pol_diff (avg |Rs-Rp|)              {best_perf['pol_diff']:8.4f} %")
    print(f"  max_pol_diff                        {best_perf['max_pol_diff']:8.4f} %")
    print("=" * 60 + "\n")

    # 偏振可视化
    vis.plot_polarization_radial(tmm, best_d, best_mats, save_name="polarization_radial")

    if os.path.exists(history_csv):
        h_df = pd.read_csv(history_csv)
        train_loss = h_df["train_loss"].values
        val_loss = h_df["val_loss"].values
        vis.plot_loss_curve(train_loss, val_loss)
        vis.plot_training_diagnostics(train_loss, val_loss)
    elif 'train_hist' in locals():
        vis.plot_loss_curve(train_hist, val_hist)
        vis.plot_training_diagnostics(train_hist, val_hist)

    vis.plot_coating_structure(best_d, best_mats)
    vis.plot_radial_reflectance(tmm, best_d, best_mats)
    vis.plot_tornado_sensitivity(tmm, best_d, best_mats)
    vis.plot_reflectance_angle_map(tmm, best_d, best_mats,
                                   wavelengths=np.linspace(191e-9, 195e-9, 41))

    # ---- 蒙特卡洛鲁棒性评估 ----
    print("\nPerforming Monte Carlo robustness analysis...")
    mc_sigma = 0.015
    reflectance_threshold = 0.5
    std_threshold = 0.12  # 与正文 Sec.3.4.1 验收规则一致（sigma_R < 0.12%）
    mc_results, mc_yield = tmm.monte_carlo_robustness(
        best_d, best_mats, n_samples=1000,
        thickness_sigma_relative=mc_sigma,
        reflectance_threshold=reflectance_threshold,
        std_threshold=std_threshold,
        seed=42
    )
    vis.plot_monte_carlo_yield(
        mc_results, best_perf, mc_yield,
        thickness_sigma_relative=mc_sigma,
        reflectance_threshold=reflectance_threshold,
        std_threshold=std_threshold,
        save_name="monte_carlo_yield"
    )

    # ---- 完整的统计输出（包含绝对均匀性） ----
    stats_summary = {}
    # 调整显示顺序：核心指标靠前，均匀性放后面参考
    metric_names = ['avg_R', 'std_R', 'PV', 'high_R', 'delta_R', 'avg_stress']
    print("\n" + "=" * 60)
    print("Monte Carlo Robustness Statistics (N=1000)")
    print("=" * 60)
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'P5':>10} {'P95':>10} | Nominal")
    print("-" * 60)

    for key in metric_names:
        if key not in mc_results:
            continue
        data = mc_results[key]
        mean_v = np.mean(data)
        std_v = np.std(data)
        p5 = np.percentile(data, 5)
        p95 = np.percentile(data, 95)
        nom = best_perf.get(key, np.nan)
        stats_summary[f'{key}_mean'] = mean_v
        stats_summary[f'{key}_std'] = std_v
        stats_summary[f'{key}_p5'] = p5
        stats_summary[f'{key}_p95'] = p95
        stats_summary[f'{key}_nominal'] = nom
        print(f"{key:<20} {mean_v:10.4f} {std_v:10.4f} {p5:10.4f} {p95:10.4f} | {nom:10.4f}")

    stats_summary['yield_rate_percent'] = mc_yield
    print("-" * 60)
    print(f"Yield (avg_R<{reflectance_threshold:g}%, "
          f"std_R<{std_threshold:g}%): {mc_yield:.1f}%")
    print("=" * 60 + "\n")

    pd.DataFrame([stats_summary]).to_csv(
        os.path.join(output_dir, "robustness_stats.csv"), index=False
    )
    print("Detailed robustness statistics saved to robustness_stats.csv")



if __name__ == "__main__":
    main()
