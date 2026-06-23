

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


Array = np.ndarray
Voxel = Tuple[int, int, int]


@dataclass
class A3PRLConfig:
    base_voxel_size: Tuple[float, float, float] = (0.40, 0.40, 0.40)
    window_seconds: float = 1.0
    scan_period: float = 0.10
    velocity_fit_frames: int = 4
    tds_weight_t: float = 0.55
    tds_weight_f: float = 0.45
    score_alpha: float = 0.30
    score_beta: float = 0.25
    score_gamma: float = 0.25
    score_delta: float = 0.20
    cusum_baseline: float = 0.05
    sprt_alpha_err: float = 0.05
    sprt_beta_err: float = 0.05
    sprt_positive_ema: float = 0.80
    sprt_ema_decay: float = 0.90
    action_ema: float = 0.70
    process_noise: float = 0.10
    measurement_noise: float = 0.35
    init_hits: int = 2
    delete_misses: int = 6
    reward_lambda_error: float = 1.0
    reward_lambda_continuity: float = 0.5
    reward_lambda_acceptance: float = 0.2
    acceptance_target: float = 0.60


@dataclass
class A3PRLAction:
    voxel_scale: float
    theta_t: float
    theta_v: float
    tau_gate: float
    quantile: float

    def smoothed(self, previous: Optional["A3PRLAction"], ema: float) -> "A3PRLAction":
        if previous is None:
            return self
        a = float(np.clip(ema, 0.0, 0.99))

        def blend(cur: float, old: float) -> float:
            return a * old + (1.0 - a) * cur

        return A3PRLAction(
            voxel_scale=blend(self.voxel_scale, previous.voxel_scale),
            theta_t=blend(self.theta_t, previous.theta_t),
            theta_v=blend(self.theta_v, previous.theta_v),
            tau_gate=blend(self.tau_gate, previous.tau_gate),
            quantile=blend(self.quantile, previous.quantile),
        )


@dataclass
class A3PRLFrameResult:
    time: float
    action: A3PRLAction
    observation: List[float]
    detections: Array
    accepted_ratio: float
    continuity: float
    track_state: Optional[List[float]]
    track_active: bool


def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class MLPPolicy:
    """Small deterministic policy head matching the paper's 4D inference state."""

    def __init__(
        self,
        obs_dim: int = 4,
        hidden_width: int = 128,
        hidden_layers: int = 3,
        seed: int = 11,
    ) -> None:
        rng = np.random.default_rng(seed)
        dims = [obs_dim] + [hidden_width] * hidden_layers + [5]
        self.weights = []
        self.biases = []
        for din, dout in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / max(1, din))
            self.weights.append(rng.normal(0.0, scale, size=(din, dout)))
            self.biases.append(np.zeros(dout, dtype=float))

    def load_npz(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        weights = data["weights"]
        biases = data["biases"]
        self.weights = [np.asarray(w, dtype=float) for w in weights]
        self.biases = [np.asarray(b, dtype=float) for b in biases]

    def __call__(self, observation: Sequence[float]) -> A3PRLAction:
        x = np.asarray(observation, dtype=float)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(x @ w + b)
        raw = x @ self.weights[-1] + self.biases[-1]
        y = sigmoid(raw)
        return A3PRLAction(
            voxel_scale=float(0.60 + 1.80 * y[0]),
            theta_t=float(0.25 + 0.70 * y[1]),
            theta_v=float(0.02 + 1.50 * y[2]),
            tau_gate=float(1.0 + 5.0 * y[3]),
            quantile=float(0.50 + 0.45 * y[4]),
        )


class HeuristicPolicy:
    """Paper-compatible non-RL adaptive baseline for use without trained weights."""

    def __call__(self, observation: Sequence[float]) -> A3PRLAction:
        s_t, s_f, rho, xi = [float(v) for v in observation]
        sparsity = np.clip(0.5 * s_t + 0.5 * s_f, 0.0, 1.0)
        instability = np.clip(1.0 - xi, 0.0, 1.0)
        return A3PRLAction(
            voxel_scale=float(0.75 + 1.15 * sparsity),
            theta_t=float(0.30 + 0.50 * (1.0 - rho)),
            theta_v=float(0.03 + 0.45 * instability),
            tau_gate=float(1.50 + 3.20 * instability),
            quantile=float(0.50 + 0.20 * (1.0 - rho)),
        )


class SpatiotemporalTensorizer:
    def __init__(self, config: A3PRLConfig) -> None:
        self.config = config

    def effective_voxel_size(self, action: A3PRLAction) -> Array:
        return np.asarray(self.config.base_voxel_size, dtype=float) * float(action.voxel_scale)

    def voxelize(self, points: Array, action: A3PRLAction) -> Dict[Voxel, Array]:
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("points must be Nx4: x,y,z,t")
        scale = self.effective_voxel_size(action)
        keys = np.floor(arr[:, :3] / scale).astype(np.int64)
        voxels: Dict[Voxel, List[Array]] = defaultdict(list)
        for key, point in zip(keys, arr):
            voxels[tuple(int(v) for v in key)].append(point)
        return {key: np.vstack(value) for key, value in voxels.items()}

    def features(self, points: Array, action: A3PRLAction, current_time: float) -> Dict[Voxel, Dict[str, object]]:
        voxels = self.voxelize(points, action)
        expected_frames = max(1.0, self.config.window_seconds / self.config.scan_period)
        result: Dict[Voxel, Dict[str, object]] = {}
        for key, pts in voxels.items():
            times = pts[:, 3]
            frame_ids = np.floor(times / self.config.scan_period).astype(int)
            delta_t = float(np.max(times) - np.min(times)) if len(times) else 0.0
            occupancy = len(set(int(v) for v in frame_ids))
            kappa = float(np.clip(occupancy / expected_frames, 0.0, 1.0))
            s_t = float(np.clip(1.0 - delta_t / max(self.config.window_seconds, 1e-9), 0.0, 1.0))
            s_f = float(np.clip(1.0 - kappa, 0.0, 1.0))
            result[key] = {
                "points": pts,
                "centroid": np.mean(pts[:, :3], axis=0),
                "s_t": s_t,
                "s_f": s_f,
                "times": times,
                "frame_ids": set(int(v) for v in frame_ids),
            }
        return result


class DualHeadProposal:
    def __init__(self, config: A3PRLConfig) -> None:
        self.config = config
        self.centroid_history: Dict[Voxel, Deque[Tuple[float, Array]]] = defaultdict(
            lambda: deque(maxlen=max(2, config.velocity_fit_frames + 1))
        )
        self.prev_velocity: Dict[Voxel, Array] = {}
        self.cusum: Dict[Voxel, float] = defaultdict(float)

    def update_velocity(self, key: Voxel, time: float, centroid: Array) -> Tuple[Array, float]:
        history = self.centroid_history[key]
        history.append((time, np.asarray(centroid, dtype=float)))
        if len(history) < 2:
            velocity = np.zeros(3, dtype=float)
        else:
            ts = np.asarray([item[0] for item in history], dtype=float)
            pts = np.asarray([item[1] for item in history], dtype=float)
            dt = np.maximum(ts - ts.mean(), 1e-9)
            velocity = np.zeros(3, dtype=float)
            for dim in range(3):
                cov = float(np.sum((ts - ts.mean()) * (pts[:, dim] - pts[:, dim].mean())))
                var = float(np.sum((ts - ts.mean()) ** 2))
                velocity[dim] = cov / max(var, 1e-9)
        previous = self.prev_velocity.get(key, np.zeros(3, dtype=float))
        delta_v = float(np.linalg.norm(velocity - previous))
        self.prev_velocity[key] = velocity
        self.cusum[key] = max(0.0, self.cusum[key] + (delta_v - self.config.cusum_baseline))
        return velocity, delta_v

    def propose(
        self,
        features: Dict[Voxel, Dict[str, object]],
        action: A3PRLAction,
        current_time: float,
    ) -> Tuple[Set[Voxel], Dict[Voxel, Dict[str, float]]]:
        candidates: Set[Voxel] = set()
        stats: Dict[Voxel, Dict[str, float]] = {}
        for key, item in features.items():
            s_t = float(item["s_t"])
            s_f = float(item["s_f"])
            phi_t = self.config.tds_weight_t * s_t + self.config.tds_weight_f * s_f
            velocity, delta_v = self.update_velocity(key, current_time, np.asarray(item["centroid"]))
            is_tds = phi_t >= action.theta_t
            is_vc = delta_v >= action.theta_v or self.cusum[key] >= action.theta_v
            if is_tds or is_vc:
                candidates.add(key)
            stats[key] = {
                "s_t": s_t,
                "s_f": s_f,
                "phi_t": float(phi_t),
                "delta_v": float(delta_v),
                "velocity_norm": float(np.linalg.norm(velocity)),
                "cusum": float(self.cusum[key]),
            }
        return candidates, stats


def neighborhood(key: Voxel, radius: int = 1) -> Set[Voxel]:
    x, y, z = key
    return {
        (x + dx, y + dy, z + dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    }


def expand_candidates(candidates: Iterable[Voxel]) -> Set[Voxel]:
    expanded: Set[Voxel] = set()
    for key in candidates:
        expanded.update(neighborhood(key, 1))
    return expanded


class AdaptiveScorer:
    def __init__(self, config: A3PRLConfig) -> None:
        self.config = config
        self.prev_neighborhoods: Dict[Voxel, Set[Voxel]] = {}
        self.sprt_scores: Dict[Voxel, float] = defaultdict(float)
        self.pi1 = config.sprt_positive_ema

    def _spatial_stability(self, key: Voxel) -> float:
        current = neighborhood(key, 1)
        previous = self.prev_neighborhoods.get(key, set())
        self.prev_neighborhoods[key] = current
        if not previous:
            return 0.0
        return len(current & previous) / max(1, len(current | previous))

    def score(
        self,
        features: Dict[Voxel, Dict[str, object]],
        proposal_stats: Dict[Voxel, Dict[str, float]],
        candidates: Set[Voxel],
        action: A3PRLAction,
    ) -> Tuple[Array, float, Dict[Voxel, float]]:
        all_scores: Dict[Voxel, float] = {}
        for key, stats in proposal_stats.items():
            velocity_consistency = 1.0 / (1.0 + stats["delta_v"])
            spatial_stability = self._spatial_stability(key)
            score = (
                self.config.score_alpha * stats["s_t"]
                + self.config.score_beta * stats["s_f"]
                + self.config.score_gamma * velocity_consistency
                + self.config.score_delta * spatial_stability
            )
            all_scores[key] = float(score)

        background = [value for key, value in all_scores.items() if key not in candidates]
        if background:
            tau = float(np.quantile(background, np.clip(action.quantile, 0.0, 1.0)))
        elif all_scores:
            tau = float(np.quantile(list(all_scores.values()), np.clip(action.quantile, 0.0, 1.0)))
        else:
            return np.empty((0, 3), dtype=float), 0.0, all_scores

        foreground = {key for key in candidates if all_scores.get(key, -np.inf) >= tau}
        accepted = []
        for key in foreground:
            y = 1.0
            self._update_sprt(key, y)
            # The paper's SPRT is retained here. For a previously unseen moving
            # voxel, allow strong foreground observations to bootstrap the track;
            # otherwise a fast SAT may leave a voxel before enough repeated
            # observations accumulate for a formal SPRT decision.
            if self.sprt_scores[key] >= self._sprt_accept_threshold() or all_scores[key] >= tau:
                pts = np.asarray(features[key]["points"], dtype=float)
                accepted.append(np.mean(pts[:, :3], axis=0))
        for key in set(candidates) - foreground:
            self._update_sprt(key, 0.0)

        ratio = float(len(accepted)) / max(1, len(candidates))
        self.pi1 = self.config.sprt_ema_decay * self.pi1 + (1.0 - self.config.sprt_ema_decay) * max(ratio, 1e-3)
        detections = np.vstack(accepted) if accepted else np.empty((0, 3), dtype=float)
        return detections, ratio, all_scores

    def _update_sprt(self, key: Voxel, y: float) -> None:
        p1 = float(np.clip(self.pi1, 1e-3, 1.0 - 1e-3))
        p0 = float(np.clip(1.0 - self.config.acceptance_target, 1e-3, 1.0 - 1e-3))
        if y >= 0.5:
            llr = np.log(p1 / p0)
        else:
            llr = np.log((1.0 - p1) / (1.0 - p0))
        self.sprt_scores[key] += float(llr)

    def _sprt_accept_threshold(self) -> float:
        return float(np.log((1.0 - self.config.sprt_beta_err) / self.config.sprt_alpha_err))


class KalmanSATTracker:
    def __init__(self, config: A3PRLConfig) -> None:
        self.config = config
        self.x: Optional[Array] = None
        self.P: Optional[Array] = None
        self.last_time: Optional[float] = None
        self.hits = 0
        self.misses = 0
        self.active = False
        self.last_mahalanobis: Optional[float] = None

    def step(
        self,
        detections: Array,
        time: float,
        action: A3PRLAction,
        proposal_stats: Optional[Dict[Voxel, Dict[str, float]]] = None,
    ) -> Tuple[Optional[Array], float]:
        dt = self.config.scan_period if self.last_time is None else max(1e-3, float(time - self.last_time))
        self.last_time = float(time)

        if self.x is None:
            if len(detections):
                self.x = np.zeros(6, dtype=float)
                self.x[:3] = detections[0]
                self.P = np.eye(6, dtype=float)
                self.hits = 1
                self.misses = 0
            return self._state_or_none(), 0.0

        self._predict(dt)
        if len(detections) == 0:
            self.misses += 1
            if self.misses > self.config.delete_misses:
                self.active = False
            return self._state_or_none(), 0.0

        best_idx, best_dist = self._associate(detections, action.tau_gate)
        if best_idx is None:
            self.misses += 1
            if self.misses > self.config.delete_misses:
                self.active = False
            return self._state_or_none(), 0.0

        self._update(detections[best_idx])
        self.hits += 1
        self.misses = 0
        self.active = self.hits >= self.config.init_hits
        self.last_mahalanobis = best_dist
        continuity = float(np.exp(-best_dist / max(action.tau_gate, 1e-6)))
        return self._state_or_none(), continuity

    def _predict(self, dt: float) -> None:
        assert self.x is not None and self.P is not None
        F = np.eye(6, dtype=float)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        q = self.config.process_noise
        Q = np.eye(6, dtype=float) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _associate(self, detections: Array, tau_gate: float) -> Tuple[Optional[int], float]:
        assert self.x is not None and self.P is not None
        H = np.zeros((3, 6), dtype=float)
        H[:, :3] = np.eye(3)
        R = np.eye(3, dtype=float) * self.config.measurement_noise
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.pinv(S)
        best_idx = None
        best_dist = float("inf")
        for i, z in enumerate(detections):
            r = z - H @ self.x
            dist = float(np.sqrt(max(0.0, r.T @ Sinv @ r)))
            if dist <= tau_gate and dist < best_dist:
                best_idx = i
                best_dist = dist
        return best_idx, best_dist

    def _update(self, z: Array) -> None:
        assert self.x is not None and self.P is not None
        H = np.zeros((3, 6), dtype=float)
        H[:, :3] = np.eye(3)
        R = np.eye(3, dtype=float) * self.config.measurement_noise
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)
        residual = z - H @ self.x
        self.x = self.x + K @ residual
        self.P = (np.eye(6) - K @ H) @ self.P

    def _state_or_none(self) -> Optional[Array]:
        if self.x is None:
            return None
        return self.x.copy()


class A3PRLPipeline:
    def __init__(
        self,
        config: Optional[A3PRLConfig] = None,
        policy: Optional[object] = None,
    ) -> None:
        self.config = config or A3PRLConfig()
        self.policy = policy or HeuristicPolicy()
        self.tensorizer = SpatiotemporalTensorizer(self.config)
        self.proposal = DualHeadProposal(self.config)
        self.scorer = AdaptiveScorer(self.config)
        self.tracker = KalmanSATTracker(self.config)
        self.previous_action: Optional[A3PRLAction] = None
        self.previous_observation = [0.0, 0.0, 0.0, 0.0]

    def process_stream(self, points: Array) -> List[A3PRLFrameResult]:
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("points must be Nx4: x,y,z,t")
        times = np.unique(arr[:, 3])
        results = []
        for current_time in times:
            window_start = current_time - self.config.window_seconds
            window = arr[(arr[:, 3] >= window_start) & (arr[:, 3] <= current_time)]
            results.append(self.process_window(window, float(current_time)))
        return results

    def process_window(self, window_points: Array, current_time: float) -> A3PRLFrameResult:
        raw_action = self.policy(self.previous_observation)
        action = raw_action.smoothed(self.previous_action, self.config.action_ema)
        self.previous_action = action

        features = self.tensorizer.features(window_points, action, current_time)
        candidates, proposal_stats = self.proposal.propose(features, action, current_time)
        candidates = expand_candidates(candidates) & set(features.keys())
        detections, rho, _ = self.scorer.score(features, proposal_stats, candidates, action)
        state, xi = self.tracker.step(detections, current_time, action, proposal_stats)

        s_t_values = [float(item["s_t"]) for item in features.values()]
        s_f_values = [float(item["s_f"]) for item in features.values()]
        observation = [
            float(np.mean(s_t_values)) if s_t_values else 0.0,
            float(np.mean(s_f_values)) if s_f_values else 0.0,
            float(rho),
            float(xi),
        ]
        self.previous_observation = observation

        return A3PRLFrameResult(
            time=float(current_time),
            action=action,
            observation=observation,
            detections=detections,
            accepted_ratio=float(rho),
            continuity=float(xi),
            track_state=None if state is None else state.tolist(),
            track_active=bool(self.tracker.active),
        )

    def reward(self, geometric_error: float, continuity: float, acceptance_ratio: float) -> float:
        """Equation (16): r_t = -(lambda1*err + lambda2*(1-xi) + lambda3*|rho-rho_target|)."""

        return -(
            self.config.reward_lambda_error * float(geometric_error)
            + self.config.reward_lambda_continuity * (1.0 - float(continuity))
            + self.config.reward_lambda_acceptance
            * abs(float(acceptance_ratio) - self.config.acceptance_target)
        )


def load_points(path: Path) -> Array:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        if "points" not in data:
            raise ValueError(".npz input must contain an Nx4 'points' array")
        return np.asarray(data["points"], dtype=float)
    if path.suffix.lower() in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter=",", dtype=float)
    raise ValueError("Supported inputs: .npz with points, or .csv/.txt with x,y,z,t columns")


def synthetic_demo() -> Array:
    rng = np.random.default_rng(23)
    rows = []
    for frame in range(36):
        t = frame * 0.10
        background = rng.normal(0.0, 10.0, size=(180, 3))
        background[:, 2] = np.abs(background[:, 2])
        center = np.array([0.10 * frame, 2.0 * np.sin(frame / 10.0), 6.0 + 0.03 * frame])
        target_count = 8 + (frame % 4)
        uav = center + rng.normal(0.0, 0.10, size=(target_count, 3))
        pts = np.vstack([background, uav])
        times = np.full((len(pts), 1), t)
        rows.append(np.hstack([pts, times]))
    return np.vstack(rows)


def save_results(results: Sequence[A3PRLFrameResult], output_prefix: Path) -> None:
    track_csv = output_prefix.with_name(output_prefix.name + "_track.csv")
    diag_json = output_prefix.with_name(output_prefix.name + "_diagnostics.json")

    with track_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "active",
                "accepted_ratio",
                "continuity",
            ]
        )
        for item in results:
            state = item.track_state or [np.nan] * 6
            writer.writerow(
                [
                    item.time,
                    *state,
                    int(item.track_active),
                    item.accepted_ratio,
                    item.continuity,
                ]
            )

    serializable = []
    for item in results:
        row = asdict(item)
        row["detections"] = np.asarray(item.detections).tolist()
        serializable.append(row)
    diag_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3PRL adaptive sparse-LiDAR SAT tracker")
    parser.add_argument("--input", type=Path, default=None, help="Nx4 point stream: x,y,z,t")
    parser.add_argument("--output-prefix", type=Path, default=Path("yuan_a3prl_result"))
    parser.add_argument("--policy", choices=["heuristic", "mlp"], default="heuristic")
    parser.add_argument("--policy-weights", type=Path, default=None, help=".npz with weights/biases for MLPPolicy")
    parser.add_argument("--window-seconds", type=float, default=A3PRLConfig.window_seconds)
    parser.add_argument("--scan-period", type=float, default=A3PRLConfig.scan_period)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = synthetic_demo() if args.input is None else load_points(args.input)
    config = A3PRLConfig(window_seconds=args.window_seconds, scan_period=args.scan_period)
    if args.policy == "mlp":
        policy = MLPPolicy()
        if args.policy_weights is not None:
            policy.load_npz(args.policy_weights)
    else:
        policy = HeuristicPolicy()
    pipeline = A3PRLPipeline(config=config, policy=policy)
    results = pipeline.process_stream(points)
    save_results(results, args.output_prefix)
    final = results[-1] if results else None
    print(
        json.dumps(
            {
                "frames": len(results),
                "final_track_state": None if final is None else final.track_state,
                "final_action": None if final is None else asdict(final.action),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
