from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import compare_eegdenoisenet as bench
from signal_preprocessing.asr_subspace import ASRConfig
from signal_preprocessing.filtering import FilterConfig
from signal_preprocessing.fuzzy_artifact import FuzzyArtifactConfig
from signal_preprocessing.preprocess import clean_raw_signal


_WORKER_DATA: dict[str, Any] = {}


def _init_profile_worker(data_root: str) -> None:
    root = Path(data_root)
    _WORKER_DATA["eeg"] = np.load(root / "EEG_all_epochs.npy", mmap_mode="r")
    _WORKER_DATA["eog"] = np.load(root / "EOG_all_epochs.npy", mmap_mode="r")
    _WORKER_DATA["emg"] = np.load(root / "EMG_all_epochs.npy", mmap_mode="r")


def _evaluate_profile_worker(task: dict[str, Any]) -> dict[str, Any]:
    name = str(task["name"])
    params = _canonicalize_params(dict(task["params"]))
    n_trials = int(task["n_trials"])
    seeds = [int(s) for s in task["seeds"]]
    snr_levels_by_noise = {
        "EOG": [float(v) for v in task["snr_levels_by_noise"]["EOG"]],
        "EMG": [float(v) for v in task["snr_levels_by_noise"]["EMG"]],
    }
    include_per_seed = bool(task.get("include_per_seed", False))

    profile = _build_profile(name, params)
    filter_cfg = profile["filter"]
    asr_cfg = profile["asr"]
    fuzzy_cfg = profile["fuzzy"]

    eeg = _WORKER_DATA["eeg"]
    eog = _WORKER_DATA["eog"]
    emg = _WORKER_DATA["emg"]

    per_seed_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, float]] = []

    def _denoise_cfg(raw_noisy, cfg, _noise_label=None):
        return clean_raw_signal(raw_noisy, fuzzy_config=cfg, asr_config=asr_cfg, filter_config=filter_cfg)

    for seed in seeds:
        profile_rng = np.random.default_rng(int(seed))
        reports = [
            bench.evaluate_noise_type(
                eeg,
                eog,
                "EOG",
                snr_levels_by_noise["EOG"],
                n_trials,
                256.0,
                fuzzy_cfg,
                _denoise_cfg,
                True,
                profile_rng,
                target_len=512,
                filter_config=filter_cfg,
            ),
            bench.evaluate_noise_type(
                eeg,
                emg,
                "EMG",
                snr_levels_by_noise["EMG"],
                n_trials,
                512.0,
                fuzzy_cfg,
                _denoise_cfg,
                True,
                profile_rng,
                target_len=1024,
                filter_config=filter_cfg,
            ),
        ]
        summary = _summarize_reports(reports)
        seed_summaries.append(summary)
        if include_per_seed:
            per_seed_rows.append({"seed": int(seed), "summary": summary})

    summary_mean, summary_std = _aggregate_seed_summaries(seed_summaries)
    score, score_parts = _robust_accuracy_score(summary_mean, summary_std)

    return {
        "name": name,
        "params": params,
        "score": float(score),
        "score_parts": score_parts,
        "summary_mean": summary_mean,
        "summary_std": summary_std,
        "filter_config": asdict(filter_cfg),
        "asr_config": asdict(asr_cfg),
        "fuzzy_config": asdict(fuzzy_cfg),
        "n_trials": n_trials,
        "seeds": seeds,
        "seed_count": len(seeds),
        "evaluated_seed_count": len(seeds),
        "pruned_early": False,
        "per_seed": per_seed_rows,
    }


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or not val.strip():
        return int(default)
    try:
        return int(val)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or not val.strip():
        return float(default)
    try:
        return float(val)
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or not val.strip():
        return bool(default)
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("in_depth_sweep")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in logger.handlers:
        h.close()
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _summary_brief(summary: dict[str, float]) -> str:
    return (
        f"Acc={summary.get('accuracy_like_mean', 0.0):.4f}, "
        f"CC={summary.get('cc_mean', 0.0):.4f}, "
        f"RRMSE_t={summary.get('rrmse_temporal_mean', 0.0):.4f}, "
        f"RRMSE_s={summary.get('rrmse_spectral_mean', 0.0):.4f}, "
        f"SNR={summary.get('snr_improvement_db_mean', 0.0):.4f}, "
        f"Acc_low={summary.get('accuracy_like_low_snr_mean', 0.0):.4f}"
    )


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _round_step(v: float, step: float) -> float:
    return round(round(float(v) / step) * step, 6)


PARAM_SPECS: dict[str, dict[str, float]] = {
    "severe_drift_ratio_threshold": {"min": 3.2, "max": 4.8, "step": 0.2},
    "severe_emg_ratio_threshold": {"min": 0.36, "max": 0.56, "step": 0.02},
    "severe_ocular_hpf_hz": {"min": 4.0, "max": 7.0, "step": 0.5},
    "severe_myogenic_hpf_hz": {"min": 9.0, "max": 12.0, "step": 1.0},
    "severe_myogenic_lpf_hz": {"min": 40.0, "max": 50.0, "step": 1.0},
    "asr_window_seconds": {"min": 0.8, "max": 1.4, "step": 0.1},
    "asr_step_seconds": {"min": 0.4, "max": 0.8, "step": 0.1},
    "asr_calibration_percentile": {"min": 20.0, "max": 45.0, "step": 5.0},
    "asr_variance_cutoff": {"min": 4.0, "max": 6.5, "step": 0.5},
    "asr_max_attenuation": {"min": 0.65, "max": 0.85, "step": 0.05},
    "artifact_score_threshold": {"min": 0.56, "max": 0.66, "step": 0.02},
    "min_denoise_strength": {"min": 0.08, "max": 0.20, "step": 0.02},
    "max_denoise_strength": {"min": 0.66, "max": 0.82, "step": 0.02},
    "severe_epoch_ptp_ratio": {"min": 5.0, "max": 7.0, "step": 0.5},
    "severe_hf_ratio": {"min": 0.45, "max": 0.65, "step": 0.05},
    "severe_lf_ratio": {"min": 0.75, "max": 1.10, "step": 0.05},
    "severe_strength_boost": {"min": 0.04, "max": 0.14, "step": 0.02},
}


def _canonicalize_params(params: dict[str, float | int]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    notch = int(params.get("notch_harmonics", 2))
    out["notch_harmonics"] = int(max(1, min(3, notch)))

    for key, spec in PARAM_SPECS.items():
        val = float(params.get(key, spec["min"]))
        val = _clip(val, spec["min"], spec["max"])
        out[key] = float(_round_step(val, spec["step"]))

    # Keep ASR overlap sensible.
    out["asr_step_seconds"] = float(
        min(float(out["asr_step_seconds"]), max(0.4, 0.85 * float(out["asr_window_seconds"])))
    )
    out["asr_step_seconds"] = float(_round_step(float(out["asr_step_seconds"]), PARAM_SPECS["asr_step_seconds"]["step"]))

    # Keep fuzzy denoise bounds sensible.
    min_strength = float(out["min_denoise_strength"])
    max_strength = float(out["max_denoise_strength"])
    if max_strength < (min_strength + 0.10):
        max_strength = min(0.82, min_strength + 0.10)
    out["max_denoise_strength"] = float(_round_step(max_strength, PARAM_SPECS["max_denoise_strength"]["step"]))

    # Ensure myogenic band has enough width.
    myo_hpf = float(out["severe_myogenic_hpf_hz"])
    myo_lpf = float(out["severe_myogenic_lpf_hz"])
    min_lpf = myo_hpf + 20.0
    if myo_lpf < min_lpf:
        myo_lpf = min(50.0, min_lpf)
    out["severe_myogenic_lpf_hz"] = float(_round_step(myo_lpf, PARAM_SPECS["severe_myogenic_lpf_hz"]["step"]))

    return out


def _params_key(params: dict[str, float | int]) -> str:
    return json.dumps(_canonicalize_params(params), sort_keys=True)


def _summarize_reports(reports: list[dict[str, object]]) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for rep in reports:
        for row in rep["results"]:
            rows.append(row)

    low_rows = [r for r in rows if float(r.get("target_snr_db", 999.0)) <= -3.0]

    def _mean_for(metric: str, source: list[dict[str, float]]) -> float:
        if not source:
            return 0.0
        return float(np.mean([float(r[metric]) for r in source]))

    if not rows:
        return {
            "snr_improvement_db_mean": 0.0,
            "rrmse_temporal_mean": 0.0,
            "rrmse_spectral_mean": 0.0,
            "cc_mean": 0.0,
            "accuracy_like_mean": 0.0,
            "snr_improvement_db_low_snr_mean": 0.0,
            "rrmse_temporal_low_snr_mean": 0.0,
            "rrmse_spectral_low_snr_mean": 0.0,
            "cc_low_snr_mean": 0.0,
            "accuracy_like_low_snr_mean": 0.0,
        }

    return {
        "snr_improvement_db_mean": _mean_for("snr_improvement_db", rows),
        "rrmse_temporal_mean": _mean_for("rrmse_temporal_mean", rows),
        "rrmse_spectral_mean": _mean_for("rrmse_spectral_mean", rows),
        "cc_mean": _mean_for("cc_mean", rows),
        "accuracy_like_mean": _mean_for("accuracy_like_mean", rows),
        "snr_improvement_db_low_snr_mean": _mean_for("snr_improvement_db", low_rows),
        "rrmse_temporal_low_snr_mean": _mean_for("rrmse_temporal_mean", low_rows),
        "rrmse_spectral_low_snr_mean": _mean_for("rrmse_spectral_mean", low_rows),
        "cc_low_snr_mean": _mean_for("cc_mean", low_rows),
        "accuracy_like_low_snr_mean": _mean_for("accuracy_like_mean", low_rows),
    }


def _aggregate_seed_summaries(seed_summaries: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    if not seed_summaries:
        return {}, {}

    keys = sorted(seed_summaries[0].keys())
    mean_out: dict[str, float] = {}
    std_out: dict[str, float] = {}
    for key in keys:
        vals = np.asarray([float(s[key]) for s in seed_summaries], dtype=float)
        mean_out[key] = float(np.mean(vals))
        std_out[key] = float(np.std(vals))
    return mean_out, std_out


def _robust_accuracy_score(summary_mean: dict[str, float], summary_std: dict[str, float]) -> tuple[float, dict[str, float]]:
    # Balanced objective with low-SNR emphasis and stability penalty.
    base = (
        0.34 * float(summary_mean.get("accuracy_like_mean", 0.0))
        + 0.22 * float(summary_mean.get("cc_mean", 0.0))
        - 0.18 * float(summary_mean.get("rrmse_temporal_mean", 0.0))
        - 0.08 * float(summary_mean.get("rrmse_spectral_mean", 0.0))
        + 0.04 * float(summary_mean.get("snr_improvement_db_mean", 0.0))
    )
    low_snr = (
        0.14 * float(summary_mean.get("accuracy_like_low_snr_mean", 0.0))
        + 0.10 * float(summary_mean.get("cc_low_snr_mean", 0.0))
        - 0.08 * float(summary_mean.get("rrmse_temporal_low_snr_mean", 0.0))
        - 0.04 * float(summary_mean.get("rrmse_spectral_low_snr_mean", 0.0))
    )
    stability_penalty = (
        0.10 * float(summary_std.get("accuracy_like_mean", 0.0))
        + 0.08 * float(summary_std.get("cc_mean", 0.0))
        + 0.08 * float(summary_std.get("rrmse_temporal_mean", 0.0))
        + 0.04 * float(summary_std.get("rrmse_spectral_mean", 0.0))
    )

    total = base + low_snr - stability_penalty
    return float(total), {
        "base_component": float(base),
        "low_snr_component": float(low_snr),
        "stability_penalty": float(stability_penalty),
    }


def _build_profile(name: str, params: dict[str, float | int]) -> dict[str, object]:
    p = _canonicalize_params(params)
    return {
        "name": name,
        "filter": FilterConfig(
            notch_harmonics=int(p["notch_harmonics"]),
            severe_drift_ratio_threshold=float(p["severe_drift_ratio_threshold"]),
            severe_emg_ratio_threshold=float(p["severe_emg_ratio_threshold"]),
            severe_ocular_hpf_hz=float(p["severe_ocular_hpf_hz"]),
            severe_myogenic_hpf_hz=float(p["severe_myogenic_hpf_hz"]),
            severe_myogenic_lpf_hz=float(p["severe_myogenic_lpf_hz"]),
        ),
        "asr": ASRConfig(
            enabled=True,
            window_seconds=float(p["asr_window_seconds"]),
            step_seconds=float(p["asr_step_seconds"]),
            calibration_percentile=float(p["asr_calibration_percentile"]),
            variance_cutoff=float(p["asr_variance_cutoff"]),
            max_attenuation=float(p["asr_max_attenuation"]),
        ),
        "fuzzy": FuzzyArtifactConfig(
            artifact_score_threshold=float(p["artifact_score_threshold"]),
            min_denoise_strength=float(p["min_denoise_strength"]),
            max_denoise_strength=float(p["max_denoise_strength"]),
            severe_epoch_ptp_ratio=float(p["severe_epoch_ptp_ratio"]),
            severe_hf_ratio=float(p["severe_hf_ratio"]),
            severe_lf_ratio=float(p["severe_lf_ratio"]),
            severe_strength_boost=float(p["severe_strength_boost"]),
            severe_swt_level_boost=0,
            severe_approx_denoise=False,
            parallel_channels=False,
        ),
        "params": p,
    }


def _sample_candidate_params(rng: random.Random) -> dict[str, float | int]:
    sampled: dict[str, float | int] = {"notch_harmonics": rng.choice([1, 2, 3])}
    for key, spec in PARAM_SPECS.items():
        raw = rng.uniform(spec["min"], spec["max"])
        sampled[key] = _round_step(raw, spec["step"])
    return _canonicalize_params(sampled)


def _mutate_candidate_params(base: dict[str, float | int], rng: random.Random, scale: float) -> dict[str, float | int]:
    mutated = dict(base)

    if rng.random() < 0.35:
        mutated["notch_harmonics"] = int(rng.choice([1, 2, 3]))

    for key, spec in PARAM_SPECS.items():
        step = spec["step"]
        span = spec["max"] - spec["min"]
        sigma = max(step, scale * span)
        new_v = float(mutated.get(key, spec["min"])) + rng.gauss(0.0, sigma)
        mutated[key] = _round_step(_clip(new_v, spec["min"], spec["max"]), step)

    return _canonicalize_params(mutated)


def _seed_list(base_seed: int, count: int, stride: int) -> list[int]:
    return [int(base_seed + i * stride) for i in range(max(1, count))]


def _subsample_snr_levels(levels: list[float], stride: int) -> list[float]:
    if not levels:
        return []
    step = max(1, int(stride))
    picked = [float(levels[i]) for i in range(0, len(levels), step)]
    last = float(levels[-1])
    if last not in picked:
        picked.append(last)
    return picked


def _build_stage_snr_levels(full_levels: dict[str, list[float]], mode: str) -> dict[str, list[float]]:
    selected = mode.strip().lower()
    if selected == "full":
        return {k: [float(x) for x in v] for k, v in full_levels.items()}
    if selected == "medium":
        return {
            "EOG": _subsample_snr_levels(full_levels["EOG"], stride=2),
            "EMG": _subsample_snr_levels(full_levels["EMG"], stride=2),
        }
    # coarse
    return {
        "EOG": _subsample_snr_levels(full_levels["EOG"], stride=2),
        "EMG": _subsample_snr_levels(full_levels["EMG"], stride=3),
    }


def _eval_profile(
    profile: dict[str, object],
    *,
    eeg: np.ndarray,
    eog: np.ndarray,
    emg: np.ndarray,
    sfreq_eog: float,
    sfreq_emg: float,
    snr_levels_by_noise: dict[str, list[float]],
    n_trials: int,
    seeds: list[int],
    cache: dict[str, dict[str, Any]],
    logger: logging.Logger | None = None,
    prune_floor: float | None = None,
    prune_slack: float = 0.04,
    min_seeds_before_prune: int = 1,
    include_per_seed: bool = False,
) -> dict[str, object]:
    name = str(profile["name"])
    filter_cfg = profile["filter"]
    asr_cfg = profile["asr"]
    fuzzy_cfg = profile["fuzzy"]
    params = dict(profile.get("params", {}))

    cache_key = json.dumps(
        {
            "params": _canonicalize_params(params),
            "n_trials": int(n_trials),
            "seeds": [int(s) for s in seeds],
        },
        sort_keys=True,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        out = dict(cached)
        out["name"] = name
        if logger is not None:
            logger.info("Cache hit for %s", name)
        return out

    per_seed: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, float]] = []
    pruned_early = False
    for seed_idx, seed in enumerate(seeds, start=1):
        bench.BENCHMARK_FILTER_CONFIG = filter_cfg
        bench.BENCHMARK_ASR_CONFIG = asr_cfg

        profile_rng = np.random.default_rng(int(seed))
        reports = [
            bench.evaluate_noise_type(
                eeg,
                eog,
                "EOG",
                snr_levels_by_noise["EOG"],
                n_trials,
                sfreq_eog,
                fuzzy_cfg,
                bench.denoise_full_pipeline,
                True,
                profile_rng,
                target_len=512,
                filter_config=filter_cfg,
            ),
            bench.evaluate_noise_type(
                eeg,
                emg,
                "EMG",
                snr_levels_by_noise["EMG"],
                n_trials,
                sfreq_emg,
                fuzzy_cfg,
                bench.denoise_full_pipeline,
                True,
                profile_rng,
                target_len=1024,
                filter_config=filter_cfg,
            ),
        ]
        seed_summary = _summarize_reports(reports)
        seed_summaries.append(seed_summary)
        if include_per_seed:
            per_seed.append({"seed": int(seed), "summary": seed_summary})

        # Early-prune weak candidates when a floor score is available.
        if prune_floor is not None and seed_idx >= int(max(1, min_seeds_before_prune)) and seed_idx < len(seeds):
            partial_summaries = seed_summaries
            p_mean, p_std = _aggregate_seed_summaries(partial_summaries)
            p_score, _ = _robust_accuracy_score(p_mean, p_std)
            remaining = len(seeds) - seed_idx
            optimistic_bound = float(p_score + float(prune_slack) * float(remaining))
            if optimistic_bound < float(prune_floor):
                pruned_early = True
                if logger is not None:
                    logger.info(
                        "Early-pruned %s at seed %d/%d | partial_score=%.6f | optimistic=%.6f | floor=%.6f",
                        name,
                        seed_idx,
                        len(seeds),
                        float(p_score),
                        optimistic_bound,
                        float(prune_floor),
                    )
                break

    summary_mean, summary_std = _aggregate_seed_summaries(seed_summaries)
    score, score_parts = _robust_accuracy_score(summary_mean, summary_std)

    row = {
        "name": name,
        "params": _canonicalize_params(params),
        "score": float(score),
        "score_parts": score_parts,
        "summary_mean": summary_mean,
        "summary_std": summary_std,
        "filter_config": asdict(filter_cfg),
        "asr_config": asdict(asr_cfg),
        "fuzzy_config": asdict(fuzzy_cfg),
        "n_trials": int(n_trials),
        "seeds": [int(s) for s in seeds],
        "seed_count": int(len(seeds)),
        "evaluated_seed_count": int(len(seed_summaries)),
        "pruned_early": bool(pruned_early),
        "per_seed": per_seed if include_per_seed else [],
    }

    # Cache only fully evaluated candidates so cache entries are threshold-independent.
    if int(len(seed_summaries)) == int(len(seeds)) and not bool(pruned_early):
        cache[cache_key] = dict(row)
    return row


def _run_stage(
    stage_name: str,
    profiles: list[dict[str, object]],
    *,
    eeg: np.ndarray,
    eog: np.ndarray,
    emg: np.ndarray,
    sfreq_eog: float,
    sfreq_emg: float,
    snr_levels_by_noise: dict[str, list[float]],
    n_trials: int,
    seeds: list[int],
    cache: dict[str, dict[str, Any]],
    logger: logging.Logger,
    prune_enabled: bool,
    prune_margin: float,
    prune_slack: float,
    prune_warmup: int,
    include_per_seed: bool,
    parallel_enabled: bool,
    max_workers: int,
    min_parallel_candidates: int,
    data_root: Path,
    executor: ProcessPoolExecutor | None = None,
) -> list[dict[str, object]]:
    total = len(profiles)
    stage_start = time.perf_counter()
    use_parallel = bool(parallel_enabled) and total >= int(max(2, min_parallel_candidates))
    logger.info(
        "%s START | candidates=%d | trials=%d | seeds=%s | mode=%s",
        stage_name,
        total,
        n_trials,
        ",".join(str(s) for s in seeds),
        "parallel" if use_parallel else "serial",
    )

    results: list[dict[str, object]] = []
    best_score = -1e12
    best_name = ""

    if use_parallel:
        cpu_count = max(1, os.cpu_count() or 1)
        workers = int(max_workers) if int(max_workers) > 0 else cpu_count
        workers = max(1, min(workers, cpu_count, total))
        logger.info(
            "%s parallel settings | workers=%d | cpu_count=%d | shared_pool=%s",
            stage_name,
            workers,
            cpu_count,
            str(executor is not None),
        )

        task_payloads = [
            {
                "name": str(profile["name"]),
                "params": dict(profile.get("params", {})),
                "n_trials": int(n_trials),
                "seeds": [int(s) for s in seeds],
                "snr_levels_by_noise": {
                    "EOG": [float(v) for v in snr_levels_by_noise["EOG"]],
                    "EMG": [float(v) for v in snr_levels_by_noise["EMG"]],
                },
                "include_per_seed": bool(include_per_seed),
            }
            for profile in profiles
        ]

        owns_executor = executor is None
        active_executor = executor
        if active_executor is None:
            active_executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_profile_worker,
                initargs=(str(data_root),),
            )

        try:
            futures = {active_executor.submit(_evaluate_profile_worker, payload): payload["name"] for payload in task_payloads}
            done = 0
            for fut in as_completed(futures):
                row = fut.result()
                results.append(row)
                done += 1

                score = float(row["score"])
                if score > best_score:
                    best_score = score
                    best_name = str(row["name"])
                    logger.info(
                        "%s new best | %s | score=%.6f | %s",
                        stage_name,
                        best_name,
                        best_score,
                        _summary_brief(dict(row.get("summary_mean", {}))),
                    )

                elapsed = time.perf_counter() - stage_start
                logger.info(
                    "%s progress %d/%d | current=%s score=%.6f | seeds_eval=%d/%d | pruned=%s | elapsed=%.1fs",
                    stage_name,
                    done,
                    total,
                    str(row["name"]),
                    score,
                    int(row.get("evaluated_seed_count", len(seeds))),
                    len(seeds),
                    str(bool(row.get("pruned_early", False))),
                    elapsed,
                )
        finally:
            if owns_executor and active_executor is not None:
                active_executor.shutdown(wait=True)
    else:
        for idx, profile in enumerate(profiles, start=1):
            prune_floor = None
            if bool(prune_enabled) and idx > int(max(0, prune_warmup)) and best_score > -1e11:
                prune_floor = float(best_score - float(prune_margin))

            row = _eval_profile(
                profile,
                eeg=eeg,
                eog=eog,
                emg=emg,
                sfreq_eog=sfreq_eog,
                sfreq_emg=sfreq_emg,
                snr_levels_by_noise=snr_levels_by_noise,
                n_trials=n_trials,
                seeds=seeds,
                cache=cache,
                logger=logger,
                prune_floor=prune_floor,
                prune_slack=prune_slack,
                min_seeds_before_prune=1,
                include_per_seed=include_per_seed,
            )
            results.append(row)

            score = float(row["score"])
            if score > best_score:
                best_score = score
                best_name = str(row["name"])
                logger.info(
                    "%s new best | %s | score=%.6f | %s",
                    stage_name,
                    best_name,
                    best_score,
                    _summary_brief(dict(row.get("summary_mean", {}))),
                )

            elapsed = time.perf_counter() - stage_start
            logger.info(
                "%s progress %d/%d | current=%s score=%.6f | seeds_eval=%d/%d | pruned=%s | elapsed=%.1fs",
                stage_name,
                idx,
                total,
                str(row["name"]),
                score,
                int(row.get("evaluated_seed_count", len(seeds))),
                len(seeds),
                str(bool(row.get("pruned_early", False))),
                elapsed,
            )

    sorted_results = sorted(results, key=lambda x: float(x["score"]), reverse=True)
    elapsed_total = time.perf_counter() - stage_start
    if sorted_results:
        top = sorted_results[0]
        logger.info(
            "%s END | elapsed=%.1fs | best=%s score=%.6f | %s",
            stage_name,
            elapsed_total,
            str(top["name"]),
            float(top["score"]),
            _summary_brief(dict(top.get("summary_mean", {}))),
        )
    else:
        logger.info("%s END | elapsed=%.1fs | no results", stage_name, elapsed_total)
    return sorted_results


def main() -> None:
    logger = _configure_logger()
    run_start = time.perf_counter()

    data_root = Path(r"C:/Users/Admin/Documents/GitHub/EEGdenoiseNet/data")
    eeg = np.load(data_root / "EEG_all_epochs.npy")
    eog = np.load(data_root / "EOG_all_epochs.npy")
    emg = np.load(data_root / "EMG_all_epochs.npy")

    sfreq_eog = 256.0
    sfreq_emg = 512.0
    snr_levels_full = {
        "EOG": [float(v) for v in range(-7, 3)],
        "EMG": [float(v) for v in range(-7, 5)],
    }

    seed = _env_int("NEURO_SWEEP_SEED", 4242)
    py_rng = random.Random(seed)
    dump_all_seed_rows = _env_bool("NEURO_SWEEP_SAVE_ALL_SEED_ROWS", True)

    # Deeper multi-stage settings (all configurable).
    stage1_trials = _env_int("NEURO_SWEEP_STAGE1_TRIALS", 5)
    stage1_random_candidates = _env_int("NEURO_SWEEP_STAGE1_RANDOM_CANDIDATES", 36)
    stage1_seed_count = _env_int("NEURO_SWEEP_STAGE1_SEED_COUNT", 2)

    stage2_trials = _env_int("NEURO_SWEEP_STAGE2_TRIALS", 8)
    stage2_topk = _env_int("NEURO_SWEEP_STAGE2_TOPK", 10)
    stage2_local_per_base = _env_int("NEURO_SWEEP_STAGE2_LOCAL_PER_BASE", 6)
    stage2_mutation_scale = _env_float("NEURO_SWEEP_STAGE2_MUTATION_SCALE", 0.08)
    stage2_seed_count = _env_int("NEURO_SWEEP_STAGE2_SEED_COUNT", 3)

    stage3_trials = _env_int("NEURO_SWEEP_STAGE3_TRIALS", 14)
    stage3_topk = _env_int("NEURO_SWEEP_STAGE3_TOPK", 8)
    stage3_local_per_base = _env_int("NEURO_SWEEP_STAGE3_LOCAL_PER_BASE", 4)
    stage3_mutation_scale = _env_float("NEURO_SWEEP_STAGE3_MUTATION_SCALE", 0.05)
    stage3_seed_count = _env_int("NEURO_SWEEP_STAGE3_SEED_COUNT", 4)

    stage4_trials = _env_int("NEURO_SWEEP_STAGE4_TRIALS", 24)
    stage4_topk = _env_int("NEURO_SWEEP_STAGE4_TOPK", 3)
    stage4_seed_count = _env_int("NEURO_SWEEP_STAGE4_SEED_COUNT", 6)

    seed_stride = _env_int("NEURO_SWEEP_SEED_STRIDE", 977)

    fast_snr_subset = _env_bool("NEURO_SWEEP_FAST_SNR_SUBSET", True)
    stage1_snr_mode = os.getenv("NEURO_SWEEP_STAGE1_SNR_MODE", "coarse")
    stage2_snr_mode = os.getenv("NEURO_SWEEP_STAGE2_SNR_MODE", "medium")
    stage3_snr_mode = os.getenv("NEURO_SWEEP_STAGE3_SNR_MODE", "full")
    stage4_snr_mode = os.getenv("NEURO_SWEEP_STAGE4_SNR_MODE", "full")
    if not fast_snr_subset:
        stage1_snr_mode = stage2_snr_mode = stage3_snr_mode = stage4_snr_mode = "full"

    stage1_snr_levels = _build_stage_snr_levels(snr_levels_full, stage1_snr_mode)
    stage2_snr_levels = _build_stage_snr_levels(snr_levels_full, stage2_snr_mode)
    stage3_snr_levels = _build_stage_snr_levels(snr_levels_full, stage3_snr_mode)
    stage4_snr_levels = _build_stage_snr_levels(snr_levels_full, stage4_snr_mode)

    prune_enabled = _env_bool("NEURO_SWEEP_PRUNE_ENABLED", True)
    prune_margin = _env_float("NEURO_SWEEP_PRUNE_MARGIN", 0.012)
    prune_slack = _env_float("NEURO_SWEEP_PRUNE_SLACK", 0.04)
    prune_warmup = _env_int("NEURO_SWEEP_PRUNE_WARMUP", 3)

    parallel_enabled = _env_bool("NEURO_SWEEP_PARALLEL", True)
    max_workers = _env_int("NEURO_SWEEP_MAX_WORKERS", 0)
    min_parallel_candidates = _env_int("NEURO_SWEEP_MIN_PARALLEL_CANDIDATES", 4)

    logger.info(
        "Sweep start | seed=%d | stride=%d | stage1_rand=%d | stage2_topk=%d | stage3_topk=%d | stage4_topk=%d",
        seed,
        seed_stride,
        stage1_random_candidates,
        stage2_topk,
        stage3_topk,
        stage4_topk,
    )
    logger.info(
        "Speed settings | fast_snr_subset=%s | stage_snr_modes=(%s,%s,%s,%s) | prune_enabled=%s | prune_margin=%.4f | prune_slack=%.4f | prune_warmup=%d | parallel=%s | max_workers=%d | min_parallel_candidates=%d",
        str(fast_snr_subset),
        stage1_snr_mode,
        stage2_snr_mode,
        stage3_snr_mode,
        stage4_snr_mode,
        str(prune_enabled),
        prune_margin,
        prune_slack,
        prune_warmup,
        str(parallel_enabled),
        max_workers,
        min_parallel_candidates,
    )

    baseline_params = {
        "notch_harmonics": 3,
        "severe_drift_ratio_threshold": 4.0,
        "severe_emg_ratio_threshold": 0.45,
        "severe_ocular_hpf_hz": 5.0,
        "severe_myogenic_hpf_hz": 10.0,
        "severe_myogenic_lpf_hz": 45.0,
        "asr_window_seconds": 1.0,
        "asr_step_seconds": 0.5,
        "asr_calibration_percentile": 30.0,
        "asr_variance_cutoff": 5.0,
        "asr_max_attenuation": 0.75,
        "artifact_score_threshold": 0.62,
        "min_denoise_strength": 0.12,
        "max_denoise_strength": 0.72,
        "severe_epoch_ptp_ratio": 6.0,
        "severe_hf_ratio": 0.55,
        "severe_lf_ratio": 0.9,
        "severe_strength_boost": 0.08,
    }

    eval_cache: dict[str, dict[str, Any]] = {}

    parallel_executor: ProcessPoolExecutor | None = None
    if parallel_enabled:
        cpu_count = max(1, os.cpu_count() or 1)
        configured_workers = int(max_workers) if int(max_workers) > 0 else cpu_count
        worker_count = max(1, min(configured_workers, cpu_count))
        logger.info(
            "Initializing shared process pool | workers=%d | cpu_count=%d",
            worker_count,
            cpu_count,
        )
        parallel_executor = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_profile_worker,
            initargs=(str(data_root),),
        )

    # Stage 1: global exploration.
    profiles_stage1: list[dict[str, object]] = [_build_profile("baseline_current", baseline_params)]
    seen_stage1 = {_params_key(baseline_params)}
    for i in range(stage1_random_candidates):
        for _ in range(50):
            cand = _sample_candidate_params(py_rng)
            k = _params_key(cand)
            if k not in seen_stage1:
                seen_stage1.add(k)
                profiles_stage1.append(_build_profile(f"s1_rand_{i + 1:03d}", cand))
                break
    logger.info("Stage1 candidate pool prepared: %d", len(profiles_stage1))

    stage1_seeds = _seed_list(seed, stage1_seed_count, seed_stride)
    stage1_sorted = _run_stage(
        "STAGE1_GLOBAL",
        profiles_stage1,
        eeg=eeg,
        eog=eog,
        emg=emg,
        sfreq_eog=sfreq_eog,
        sfreq_emg=sfreq_emg,
        snr_levels_by_noise=stage1_snr_levels,
        n_trials=stage1_trials,
        seeds=stage1_seeds,
        cache=eval_cache,
        logger=logger,
        prune_enabled=prune_enabled,
        prune_margin=prune_margin,
        prune_slack=prune_slack,
        prune_warmup=prune_warmup,
        include_per_seed=dump_all_seed_rows,
        parallel_enabled=parallel_enabled,
        max_workers=max_workers,
        min_parallel_candidates=min_parallel_candidates,
        data_root=data_root,
        executor=parallel_executor,
    )

    # Stage 2: local refinement around stage-1 leaders.
    stage2_profiles: list[dict[str, object]] = []
    seen_stage2: set[str] = set()
    for r in stage1_sorted[:stage2_topk]:
        base_params = dict(r["params"])
        base_key = _params_key(base_params)
        if base_key not in seen_stage2:
            seen_stage2.add(base_key)
            stage2_profiles.append(_build_profile(f"s2_base_{str(r['name'])}", base_params))

        for j in range(stage2_local_per_base):
            for _ in range(50):
                cand = _mutate_candidate_params(base_params, py_rng, stage2_mutation_scale)
                cand_key = _params_key(cand)
                if cand_key not in seen_stage2:
                    seen_stage2.add(cand_key)
                    stage2_profiles.append(_build_profile(f"s2_mut_{str(r['name'])}_{j + 1:02d}", cand))
                    break
    logger.info("Stage2 candidate pool prepared: %d", len(stage2_profiles))

    stage2_seeds = _seed_list(seed + 111, stage2_seed_count, seed_stride)
    stage2_sorted = _run_stage(
        "STAGE2_LOCAL",
        stage2_profiles,
        eeg=eeg,
        eog=eog,
        emg=emg,
        sfreq_eog=sfreq_eog,
        sfreq_emg=sfreq_emg,
        snr_levels_by_noise=stage2_snr_levels,
        n_trials=stage2_trials,
        seeds=stage2_seeds,
        cache=eval_cache,
        logger=logger,
        prune_enabled=prune_enabled,
        prune_margin=prune_margin,
        prune_slack=prune_slack,
        prune_warmup=prune_warmup,
        include_per_seed=dump_all_seed_rows,
        parallel_enabled=parallel_enabled,
        max_workers=max_workers,
        min_parallel_candidates=min_parallel_candidates,
        data_root=data_root,
        executor=parallel_executor,
    )

    # Stage 3: tighter local search around stage-2 leaders.
    stage3_profiles: list[dict[str, object]] = []
    seen_stage3: set[str] = set()
    for r in stage2_sorted[:stage3_topk]:
        base_params = dict(r["params"])
        base_key = _params_key(base_params)
        if base_key not in seen_stage3:
            seen_stage3.add(base_key)
            stage3_profiles.append(_build_profile(f"s3_base_{str(r['name'])}", base_params))

        for j in range(stage3_local_per_base):
            for _ in range(50):
                cand = _mutate_candidate_params(base_params, py_rng, stage3_mutation_scale)
                cand_key = _params_key(cand)
                if cand_key not in seen_stage3:
                    seen_stage3.add(cand_key)
                    stage3_profiles.append(_build_profile(f"s3_mut_{str(r['name'])}_{j + 1:02d}", cand))
                    break
    logger.info("Stage3 candidate pool prepared: %d", len(stage3_profiles))

    stage3_seeds = _seed_list(seed + 222, stage3_seed_count, seed_stride)
    stage3_sorted = _run_stage(
        "STAGE3_FINE",
        stage3_profiles,
        eeg=eeg,
        eog=eog,
        emg=emg,
        sfreq_eog=sfreq_eog,
        sfreq_emg=sfreq_emg,
        snr_levels_by_noise=stage3_snr_levels,
        n_trials=stage3_trials,
        seeds=stage3_seeds,
        cache=eval_cache,
        logger=logger,
        prune_enabled=prune_enabled,
        prune_margin=prune_margin,
        prune_slack=prune_slack,
        prune_warmup=prune_warmup,
        include_per_seed=dump_all_seed_rows,
        parallel_enabled=parallel_enabled,
        max_workers=max_workers,
        min_parallel_candidates=min_parallel_candidates,
        data_root=data_root,
        executor=parallel_executor,
    )

    # Stage 4: final confirmation on larger trial/seed budget.
    stage4_profiles: list[dict[str, object]] = []
    seen_stage4: set[str] = set()
    for r in stage3_sorted[:stage4_topk]:
        params = dict(r["params"])
        key = _params_key(params)
        if key not in seen_stage4:
            seen_stage4.add(key)
            stage4_profiles.append(_build_profile(f"s4_final_{str(r['name'])}", params))
    logger.info("Stage4 candidate pool prepared: %d", len(stage4_profiles))

    stage4_seeds = _seed_list(seed + 333, stage4_seed_count, seed_stride)
    stage4_sorted = _run_stage(
        "STAGE4_CONFIRM",
        stage4_profiles,
        eeg=eeg,
        eog=eog,
        emg=emg,
        sfreq_eog=sfreq_eog,
        sfreq_emg=sfreq_emg,
        snr_levels_by_noise=stage4_snr_levels,
        n_trials=stage4_trials,
        seeds=stage4_seeds,
        cache=eval_cache,
        logger=logger,
        prune_enabled=False,
        prune_margin=0.0,
        prune_slack=0.0,
        prune_warmup=0,
        include_per_seed=dump_all_seed_rows,
        parallel_enabled=parallel_enabled,
        max_workers=max_workers,
        min_parallel_candidates=min_parallel_candidates,
        data_root=data_root,
        executor=parallel_executor,
    )
    best = stage4_sorted[0]

    if parallel_executor is not None:
        parallel_executor.shutdown(wait=True)

    out_path = Path(__file__).resolve().parent / "comparisons" / "ground_truth_sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not dump_all_seed_rows:
        for row in stage1_sorted:
            row.pop("per_seed", None)
        for row in stage2_sorted:
            row.pop("per_seed", None)
        for row in stage3_sorted:
            row.pop("per_seed", None)
        for row in stage4_sorted:
            row.pop("per_seed", None)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": "maximize denoising accuracy and closeness to ground truth (robust across seeds and low-SNR)",
                "score_formula": "base + low_snr - stability_penalty; see score_parts per candidate",
                "settings": {
                    "base_seed": seed,
                    "seed_stride": seed_stride,
                    "fast_snr_subset": fast_snr_subset,
                    "snr_modes": {
                        "stage1": stage1_snr_mode,
                        "stage2": stage2_snr_mode,
                        "stage3": stage3_snr_mode,
                        "stage4": stage4_snr_mode,
                    },
                    "prune": {
                        "enabled": prune_enabled,
                        "margin": prune_margin,
                        "slack": prune_slack,
                        "warmup": prune_warmup,
                    },
                    "parallel": {
                        "enabled": parallel_enabled,
                        "max_workers": max_workers,
                        "min_parallel_candidates": min_parallel_candidates,
                    },
                    "stage1": {
                        "trials": stage1_trials,
                        "random_candidates": stage1_random_candidates,
                        "seed_count": stage1_seed_count,
                    },
                    "stage2": {
                        "trials": stage2_trials,
                        "topk_from_stage1": stage2_topk,
                        "local_per_base": stage2_local_per_base,
                        "mutation_scale": stage2_mutation_scale,
                        "seed_count": stage2_seed_count,
                    },
                    "stage3": {
                        "trials": stage3_trials,
                        "topk_from_stage2": stage3_topk,
                        "local_per_base": stage3_local_per_base,
                        "mutation_scale": stage3_mutation_scale,
                        "seed_count": stage3_seed_count,
                    },
                    "stage4": {
                        "trials": stage4_trials,
                        "topk_from_stage3": stage4_topk,
                        "seed_count": stage4_seed_count,
                    },
                    "save_all_seed_rows": dump_all_seed_rows,
                },
                "stages": {
                    "stage1": {
                        "n_trials": stage1_trials,
                        "candidates": len(stage1_sorted),
                        "seed_count": stage1_seed_count,
                        "seeds": stage1_seeds,
                        "snr_levels": stage1_snr_levels,
                        "results": stage1_sorted,
                    },
                    "stage2": {
                        "n_trials": stage2_trials,
                        "candidates": len(stage2_sorted),
                        "seed_count": stage2_seed_count,
                        "seeds": stage2_seeds,
                        "snr_levels": stage2_snr_levels,
                        "results": stage2_sorted,
                    },
                    "stage3": {
                        "n_trials": stage3_trials,
                        "candidates": len(stage3_sorted),
                        "seed_count": stage3_seed_count,
                        "seeds": stage3_seeds,
                        "snr_levels": stage3_snr_levels,
                        "results": stage3_sorted,
                    },
                    "stage4": {
                        "n_trials": stage4_trials,
                        "candidates": len(stage4_sorted),
                        "seed_count": stage4_seed_count,
                        "seeds": stage4_seeds,
                        "snr_levels": stage4_snr_levels,
                        "results": stage4_sorted,
                    },
                },
                "best_profile": best,
            },
            f,
            indent=2,
        )

    total_elapsed = time.perf_counter() - run_start
    logger.info("BEST_PROFILE=%s | score=%.6f", str(best["name"]), float(best["score"]))
    logger.info("Best summary | %s", _summary_brief(dict(best.get("summary_mean", {}))))
    logger.info("Saved sweep results: %s", str(out_path))
    logger.info("Sweep complete | elapsed=%.1fs", total_elapsed)


if __name__ == "__main__":
    main()
