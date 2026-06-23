

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
except Exception:  # pragma: no cover - optional dependency
    SklearnDBSCAN = None

try:
    from scipy.interpolate import CubicSpline
except Exception:  # pragma: no cover - optional dependency
    CubicSpline = None


Array = np.ndarray
VoxelKey = Tuple[int, int, int]


@dataclass
class LiangU3DTEConfig:
    """Configuration for the global-local clustering trajectory estimator."""

    voxel_size: float = 0.30
    dbscan_eps: float = 0.75
    dbscan_min_samples: int = 4
    min_cluster_points: int = 5
    lambda_iou: float = 0.8
    iou_epsilon: float = 1.0e-6
    density_exp_clip: float = 8.0
    pair_strategy: str = "consecutive"  # "consecutive" or "all"
    spline_samples_per_frame: int = 1
    max_points_for_fallback_dbscan: int = 12000


@dataclass
class ClusterScore:
    label: int
    score: float
    psi_rho: float
    psi_iou: float
    global_density: float
    local_density_ratios: List[float]
    frame_indices: List[int]
    num_points: int


def _as_xyz(points: Array) -> Array:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Each frame must be an array with at least 3 columns: x,y,z")
    return arr[:, :3]


def voxel_keys(points: Array, voxel_size: float) -> Array:
    """Map xyz points into integer voxel coordinates."""

    xyz = _as_xyz(points)
    if len(xyz) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.floor(xyz / float(voxel_size)).astype(np.int64)


def unique_voxel_count(points: Array, voxel_size: float) -> int:
    keys = voxel_keys(points, voxel_size)
    if len(keys) == 0:
        return 0
    return int(len({tuple(v) for v in keys}))


def density(points: Array, voxel_size: float) -> float:
    """rho_C(P|F) = |C_k(P|F)| / V(C_k(P|F))."""

    xyz = _as_xyz(points)
    occupied = unique_voxel_count(xyz, voxel_size)
    return float(len(xyz)) / max(1, occupied)


def voxel_iou(a: Array, b: Array, voxel_size: float) -> float:
    """IoU between occupied voxel sets."""

    va = {tuple(v) for v in voxel_keys(a, voxel_size)}
    vb = {tuple(v) for v in voxel_keys(b, voxel_size)}
    if not va and not vb:
        return 0.0
    inter = len(va & vb)
    union = len(va | vb)
    return float(inter) / max(1, union)


def _fallback_dbscan(points: Array, eps: float, min_samples: int, max_points: int) -> Array:
    """Small dependency-free DBSCAN fallback for environments without sklearn."""

    xyz = _as_xyz(points)
    n = len(xyz)
    if n > max_points:
        raise RuntimeError(
            "sklearn is not installed and fallback DBSCAN would be too slow for "
            f"{n} points. Install scikit-learn or raise max_points_for_fallback_dbscan."
        )
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    eps2 = eps * eps

    def region_query(i: int) -> np.ndarray:
        diff = xyz - xyz[i]
        return np.where(np.einsum("ij,ij->i", diff, diff) <= eps2)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = list(region_query(i))
        if len(neighbors) < min_samples:
            continue
        labels[i] = cluster_id
        cursor = 0
        while cursor < len(neighbors):
            j = neighbors[cursor]
            if not visited[j]:
                visited[j] = True
                j_neighbors = list(region_query(j))
                if len(j_neighbors) >= min_samples:
                    for candidate in j_neighbors:
                        if candidate not in neighbors:
                            neighbors.append(candidate)
            if labels[j] == -1:
                labels[j] = cluster_id
            cursor += 1
        cluster_id += 1
    return labels


def dbscan_labels(points: Array, config: LiangU3DTEConfig) -> Array:
    if SklearnDBSCAN is not None:
        return SklearnDBSCAN(
            eps=config.dbscan_eps,
            min_samples=config.dbscan_min_samples,
        ).fit_predict(_as_xyz(points))
    return _fallback_dbscan(
        points,
        config.dbscan_eps,
        config.dbscan_min_samples,
        config.max_points_for_fallback_dbscan,
    )


class LiangU3DTrajectoryEstimator:
    """Global-local unsupervised UAV trajectory estimator."""

    def __init__(self, config: Optional[LiangU3DTEConfig] = None) -> None:
        self.config = config or LiangU3DTEConfig()

    def fit(
        self,
        frames: Sequence[Array],
        timestamps: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """Estimate UAV trajectory from a sequence of LiDAR frames.

        Args:
            frames: list of Nx3/NxM point arrays, one array per LiDAR scan.
            timestamps: optional frame timestamps. Defaults to 0..T-1.

        Returns:
            Dictionary containing selected cluster, sampled UAV points, centroids,
            fitted trajectory, and scoring diagnostics.
        """

        if not frames:
            raise ValueError("frames cannot be empty")
        xyz_frames = [_as_xyz(frame) for frame in frames]
        if timestamps is None:
            t = np.arange(len(xyz_frames), dtype=float)
        else:
            t = np.asarray(timestamps, dtype=float)
            if len(t) != len(xyz_frames):
                raise ValueError("timestamps length must match frames length")

        merged, frame_index = self._temporal_concat(xyz_frames)
        labels = dbscan_labels(merged, self.config)
        scores = self._score_all_clusters(merged, labels, frame_index, len(xyz_frames))
        if not scores:
            return {
                "selected_label": None,
                "scores": [],
                "uav_points": np.empty((0, 4), dtype=float),
                "centroids": np.empty((0, 4), dtype=float),
                "trajectory": np.empty((0, 4), dtype=float),
                "config": asdict(self.config),
            }

        best = max(scores, key=lambda item: item.score)
        uav_points, per_frame_points = self._collect_cluster_points(
            merged, labels, frame_index, best.label, len(xyz_frames)
        )
        centroids = self._frame_centroids(per_frame_points, t)
        trajectory = self._fit_spline(centroids)
        return {
            "selected_label": best.label,
            "scores": [asdict(s) for s in sorted(scores, key=lambda item: item.score, reverse=True)],
            "uav_points": uav_points,
            "centroids": centroids,
            "trajectory": trajectory,
            "config": asdict(self.config),
        }

    @staticmethod
    def _temporal_concat(frames: Sequence[Array]) -> Tuple[Array, Array]:
        merged = []
        frame_ids = []
        for i, frame in enumerate(frames):
            xyz = _as_xyz(frame)
            merged.append(xyz)
            frame_ids.append(np.full(len(xyz), i, dtype=int))
        return np.vstack(merged), np.concatenate(frame_ids)

    def _collect_cluster_points(
        self,
        merged: Array,
        labels: Array,
        frame_index: Array,
        label: int,
        num_frames: int,
    ) -> Tuple[Array, List[Array]]:
        mask = labels == label
        per_frame = []
        with_frame_col = []
        for frame_id in range(num_frames):
            pts = merged[mask & (frame_index == frame_id)]
            per_frame.append(pts)
            if len(pts):
                ids = np.full((len(pts), 1), frame_id, dtype=float)
                with_frame_col.append(np.hstack([pts, ids]))
        stacked = np.vstack(with_frame_col) if with_frame_col else np.empty((0, 4), dtype=float)
        return stacked, per_frame

    def _score_all_clusters(
        self,
        merged: Array,
        labels: Array,
        frame_index: Array,
        num_frames: int,
    ) -> List[ClusterScore]:
        scores = []
        for label in sorted(int(v) for v in set(labels) if int(v) >= 0):
            mask = labels == label
            if int(mask.sum()) < self.config.min_cluster_points:
                continue
            _, per_frame_points = self._collect_cluster_points(
                merged, labels, frame_index, label, num_frames
            )
            scores.append(self.score_cluster(label, per_frame_points))
        return scores

    def score_cluster(self, label: int, per_frame_points: Sequence[Array]) -> ClusterScore:
        """Implement the paper's density and voxel-IoU scoring equations.

        rho_global = |C_k(P|sum F)| / V(C_k(P|sum F))
        R_k^{i,j} is approximated as local_density / global_density.
        psi_rho = sum exp(R_k)
        psi_iou = sum log(1 / IoU_k^{i,j})
        psi = psi_rho + lambda * psi_iou
        """

        non_empty = [(i, _as_xyz(p)) for i, p in enumerate(per_frame_points) if len(p)]
        if not non_empty:
            return ClusterScore(label, -np.inf, 0.0, 0.0, 0.0, [], [], 0)

        all_points = np.vstack([p for _, p in non_empty])
        rho_global = density(all_points, self.config.voxel_size)
        ratios = []
        psi_rho = 0.0
        for _, pts in non_empty:
            rho_local = density(pts, self.config.voxel_size)
            ratio = rho_local / max(rho_global, self.config.iou_epsilon)
            ratios.append(float(ratio))
            psi_rho += float(np.exp(np.clip(ratio, -self.config.density_exp_clip, self.config.density_exp_clip)))

        pairs = self._frame_pairs(non_empty)
        psi_iou = 0.0
        for (_, a), (_, b) in pairs:
            iou = voxel_iou(a, b, self.config.voxel_size)
            psi_iou += float(np.log(1.0 / max(iou, self.config.iou_epsilon)))

        score = psi_rho + self.config.lambda_iou * psi_iou
        return ClusterScore(
            label=label,
            score=float(score),
            psi_rho=float(psi_rho),
            psi_iou=float(psi_iou),
            global_density=float(rho_global),
            local_density_ratios=ratios,
            frame_indices=[i for i, _ in non_empty],
            num_points=int(len(all_points)),
        )

    def _frame_pairs(self, non_empty: Sequence[Tuple[int, Array]]) -> List[Tuple[Tuple[int, Array], Tuple[int, Array]]]:
        if len(non_empty) < 2:
            return []
        if self.config.pair_strategy == "all":
            return list(combinations(non_empty, 2))
        return list(zip(non_empty[:-1], non_empty[1:]))

    @staticmethod
    def _frame_centroids(per_frame_points: Sequence[Array], timestamps: Array) -> Array:
        rows = []
        for idx, pts in enumerate(per_frame_points):
            if len(pts):
                c = np.mean(_as_xyz(pts), axis=0)
                rows.append([timestamps[idx], c[0], c[1], c[2]])
        return np.asarray(rows, dtype=float) if rows else np.empty((0, 4), dtype=float)

    def _fit_spline(self, centroids: Array) -> Array:
        if len(centroids) <= 1:
            return centroids.copy()
        t = centroids[:, 0]
        xyz = centroids[:, 1:4]
        samples = max(len(t), int((len(t) - 1) * self.config.spline_samples_per_frame + 1))
        new_t = np.linspace(float(t[0]), float(t[-1]), samples)

        if CubicSpline is not None and len(np.unique(t)) >= 4:
            fitted = np.column_stack([CubicSpline(t, xyz[:, dim], bc_type="natural")(new_t) for dim in range(3)])
        else:
            fitted = np.column_stack([np.interp(new_t, t, xyz[:, dim]) for dim in range(3)])
        return np.column_stack([new_t, fitted])


def frames_from_npz(path: Path) -> Tuple[List[Array], Optional[Array]]:
    """Load frames from .npz.

    Supported layouts:
    - frames: object array/list of per-frame Nx3 arrays
    - points: Nx4 array with x,y,z,t_or_frame_id columns
    """

    data = np.load(path, allow_pickle=True)
    if "frames" in data:
        frames = [np.asarray(frame, dtype=float)[:, :3] for frame in data["frames"]]
        timestamps = np.asarray(data["timestamps"], dtype=float) if "timestamps" in data else None
        return frames, timestamps
    if "points" in data:
        points = np.asarray(data["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError("points must have at least four columns: x,y,z,t_or_frame")
        frames = []
        ts = []
        for value in np.unique(points[:, 3]):
            mask = points[:, 3] == value
            frames.append(points[mask, :3])
            ts.append(value)
        return frames, np.asarray(ts, dtype=float)
    raise ValueError(".npz must contain either 'frames' or 'points'")


def synthetic_demo() -> Tuple[List[Array], Array]:
    rng = np.random.default_rng(7)
    frames = []
    timestamps = np.arange(24, dtype=float)
    for i, t in enumerate(timestamps):
        background = rng.normal(0.0, 8.0, size=(300, 3))
        background[:, 2] = np.abs(background[:, 2])
        center = np.array([0.35 * t, 4.0 * np.sin(t / 5.0), 8.0 + 0.08 * t])
        uav = center + rng.normal(0.0, 0.12, size=(10, 3))
        frames.append(np.vstack([background, uav]))
    return frames, timestamps


def save_result(result: Dict[str, object], output_prefix: Path) -> None:
    trajectory = np.asarray(result["trajectory"], dtype=float)
    centroids = np.asarray(result["centroids"], dtype=float)
    uav_points = np.asarray(result["uav_points"], dtype=float)

    for suffix, array, header in [
        ("trajectory.csv", trajectory, ["t", "x", "y", "z"]),
        ("centroids.csv", centroids, ["t", "x", "y", "z"]),
        ("uav_points.csv", uav_points, ["x", "y", "z", "frame"]),
    ]:
        out = output_prefix.with_name(output_prefix.name + "_" + suffix)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(array.tolist())

    meta = dict(result)
    for key in ["trajectory", "centroids", "uav_points"]:
        meta.pop(key, None)
    output_prefix.with_name(output_prefix.name + "_diagnostics.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liang U3DTE sparse LiDAR UAV trajectory estimator")
    parser.add_argument("--input", type=Path, default=None, help=".npz file with frames or points")
    parser.add_argument("--output-prefix", type=Path, default=Path("liang_u3dte_result"))
    parser.add_argument("--voxel-size", type=float, default=LiangU3DTEConfig.voxel_size)
    parser.add_argument("--dbscan-eps", type=float, default=LiangU3DTEConfig.dbscan_eps)
    parser.add_argument("--dbscan-min-samples", type=int, default=LiangU3DTEConfig.dbscan_min_samples)
    parser.add_argument("--lambda-iou", type=float, default=LiangU3DTEConfig.lambda_iou)
    parser.add_argument("--pair-strategy", choices=["consecutive", "all"], default=LiangU3DTEConfig.pair_strategy)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is None:
        frames, timestamps = synthetic_demo()
    else:
        frames, timestamps = frames_from_npz(args.input)
    config = LiangU3DTEConfig(
        voxel_size=args.voxel_size,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        lambda_iou=args.lambda_iou,
        pair_strategy=args.pair_strategy,
    )
    estimator = LiangU3DTrajectoryEstimator(config)
    result = estimator.fit(frames, timestamps)
    save_result(result, args.output_prefix)
    print(json.dumps({k: result[k] for k in ["selected_label", "config"]}, indent=2))


if __name__ == "__main__":
    main()
