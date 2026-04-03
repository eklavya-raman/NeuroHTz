from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from signal_preprocessing.preprocess import clean_raw_signal
from signal_preprocessing.asr_subspace import ASRConfig, apply_asr_subspace_cleaning
from signal_preprocessing.filtering import FilterConfig, apply_filters, inspect_filter_plan
from signal_preprocessing.fuzzy_artifact import FuzzyArtifactConfig, remove_fuzzy_artifacts_raw


BAND_ORDER = ("delta", "theta", "alpha", "beta", "gamma")
BAND_LIMITS_HZ: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}

PUBLISHED_BAND_RATIOS: dict[str, dict[str, dict[str, float]]] = {
    "EOG": {
        "EMD": {"delta": 0.025, "theta": 0.042, "alpha": 0.096, "beta": 0.585, "gamma": 0.252},
        "Filter": {"delta": 0.000, "theta": 0.000, "alpha": 0.000, "beta": 0.405, "gamma": 0.595},
        "FCNN": {"delta": 0.129, "theta": 0.127, "alpha": 0.085, "beta": 0.500, "gamma": 0.159},
        "Simple CNN": {"delta": 0.131, "theta": 0.127, "alpha": 0.085, "beta": 0.492, "gamma": 0.165},
        "Complex CNN": {"delta": 0.128, "theta": 0.127, "alpha": 0.085, "beta": 0.493, "gamma": 0.166},
        "RNN": {"delta": 0.124, "theta": 0.122, "alpha": 0.088, "beta": 0.506, "gamma": 0.159},
        "ground truth": {"delta": 0.143, "theta": 0.141, "alpha": 0.093, "beta": 0.467, "gamma": 0.157},
        "contaminated": {"delta": 0.514, "theta": 0.216, "alpha": 0.070, "beta": 0.151, "gamma": 0.049},
    },
    "EMG": {
        "EMD": {"delta": 0.227, "theta": 0.162, "alpha": 0.093, "beta": 0.330, "gamma": 0.188},
        "Filter": {"delta": 0.000, "theta": 0.000, "alpha": 0.000, "beta": 0.312, "gamma": 0.687},
        "FCNN": {"delta": 0.147, "theta": 0.144, "alpha": 0.092, "beta": 0.481, "gamma": 0.135},
        "Simple CNN": {"delta": 0.119, "theta": 0.138, "alpha": 0.096, "beta": 0.506, "gamma": 0.142},
        "Complex CNN": {"delta": 0.123, "theta": 0.139, "alpha": 0.097, "beta": 0.492, "gamma": 0.149},
        "RNN": {"delta": 0.139, "theta": 0.138, "alpha": 0.093, "beta": 0.482, "gamma": 0.147},
        "ground truth": {"delta": 0.142, "theta": 0.140, "alpha": 0.093, "beta": 0.464, "gamma": 0.160},
        "contaminated": {"delta": 0.200, "theta": 0.141, "alpha": 0.077, "beta": 0.300, "gamma": 0.281},
    },
}


BENCHMARK_ASR_CONFIG = ASRConfig(
    enabled=True,
    window_seconds=1.0,
    step_seconds=0.6,
    calibration_percentile=35.0,
    variance_cutoff=5.0,
    max_attenuation=0.75,
)

BENCHMARK_FILTER_CONFIG = FilterConfig(
    notch_harmonics=2,
    severe_drift_ratio_threshold=3.6,
    severe_emg_ratio_threshold=0.4,
    severe_ocular_hpf_hz=6.0,
    severe_myogenic_hpf_hz=11.0,
    severe_myogenic_lpf_hz=48.0,
)


def _vector_to_band_dict(v: np.ndarray) -> dict[str, float]:
    return {k: float(v[i]) for i, k in enumerate(BAND_ORDER)}


def _band_dict_to_vector(d: dict[str, float]) -> np.ndarray:
    return np.asarray([float(d[k]) for k in BAND_ORDER], dtype=float)


def band_power_ratios(signal: np.ndarray, sfreq: float) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    if x.size < 32 or sfreq <= 0.0:
        uniform = np.full(len(BAND_ORDER), 1.0 / len(BAND_ORDER), dtype=float)
        return _vector_to_band_dict(uniform)

    x = x - float(np.mean(x))
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sfreq)
    band_vals: list[float] = []
    for b in BAND_ORDER:
        f_lo, f_hi = BAND_LIMITS_HZ[b]
        m = (freqs >= f_lo) & (freqs < f_hi)
        band_vals.append(float(np.mean(spec[m])) if np.any(m) else 0.0)
    vals = np.asarray(band_vals, dtype=float)
    vals = vals / (float(np.sum(vals)) + 1e-12)
    return _vector_to_band_dict(vals)


def band_ratio_l1(a: dict[str, float], b: dict[str, float]) -> float:
    va = _band_dict_to_vector(a)
    vb = _band_dict_to_vector(b)
    return float(np.mean(np.abs(va - vb)))


def rms(g: np.ndarray) -> float:
    g = np.asarray(g, dtype=float)
    return float(np.sqrt(np.mean(g ** 2)) + 1e-12)


def lambda_for_target_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> float:
    # Eq. (2): SNR = 10 * log10(RMS(x) / RMS(lambda * n)).
    # Therefore, lambda = RMS(x) / (RMS(n) * 10^(SNR/10)).
    return float(rms(clean) / (rms(noise) * (10 ** (snr_db / 10.0))))


def mix_with_target_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, float]:
    clean = np.asarray(clean, dtype=float)
    noise = np.asarray(noise, dtype=float)
    lam = lambda_for_target_snr(clean, noise, snr_db)
    return clean + lam * noise, lam


def snr_db(ref: np.ndarray, err: np.ndarray) -> float:
    # Eq. (2) as provided: SNR based on RMS ratio (not power ratio).
    return float(10.0 * np.log10((rms(ref) + 1e-12) / (rms(err) + 1e-12)))


def _resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if target_len <= 1 or x.size <= 1 or x.size == target_len:
        return x.copy()
    src = np.linspace(0.0, 1.0, num=x.size, endpoint=True)
    dst = np.linspace(0.0, 1.0, num=int(target_len), endpoint=True)
    return np.interp(dst, src, x).astype(float)


def rrmse_temporal(clean: np.ndarray, denoised: np.ndarray) -> float:
    return float(rms(denoised - clean) / (rms(clean) + 1e-12))


def rrmse_spectral(clean: np.ndarray, denoised: np.ndarray) -> float:
    spec_clean = np.abs(np.fft.rfft(np.asarray(clean, dtype=float)))
    spec_denoised = np.abs(np.fft.rfft(np.asarray(denoised, dtype=float)))
    return float(rms(spec_denoised - spec_clean) / (rms(spec_clean) + 1e-12))


def rrmse_spectral_psd(clean: np.ndarray, denoised: np.ndarray, sfreq: float) -> float:
    clean = np.asarray(clean, dtype=float)
    denoised = np.asarray(denoised, dtype=float)
    if clean.size < 4 or denoised.size != clean.size or sfreq <= 0.0:
        return rrmse_spectral(clean, denoised)

    n = clean.size
    win = np.hanning(n)
    c = (clean - float(np.mean(clean))) * win
    d = (denoised - float(np.mean(denoised))) * win
    psd_c = (np.abs(np.fft.rfft(c, n=n)) ** 2) / (np.sum(win**2) + 1e-12)
    psd_d = (np.abs(np.fft.rfft(d, n=n)) ** 2) / (np.sum(win**2) + 1e-12)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    m = (freqs >= 0.0) & (freqs <= 120.0)
    if not np.any(m):
        return rrmse_spectral(clean, denoised)
    return float(rms(psd_d[m] - psd_c[m]) / (rms(psd_c[m]) + 1e-12))


def cc(clean: np.ndarray, denoised: np.ndarray) -> float:
    c_std = float(np.std(clean))
    d_std = float(np.std(denoised))
    if c_std <= 1e-12 or d_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(clean, denoised)[0, 1])


def evaluate_noise_type(
    eeg_pool: np.ndarray,
    noise_pool: np.ndarray,
    noise_label: str,
    snr_levels: list[float],
    n_trials: int,
    sfreq: float,
    cfg: FuzzyArtifactConfig,
    denoise_fn,
    capture_filter_diagnostics: bool,
    rng: np.random.Generator,
    target_len: int | None = None,
    filter_config: FilterConfig | None = None,
) -> dict[str, object]:
    results: list[dict[str, float]] = []

    for target_snr in snr_levels:
        rows = []
        lambdas = []
        filter_diag_rows = []
        den_band_rows = []
        clean_band_rows = []
        noisy_band_rows = []
        idx_clean = rng.integers(0, eeg_pool.shape[0], size=n_trials)
        idx_noise = rng.integers(0, noise_pool.shape[0], size=n_trials)

        for i_clean, i_noise in zip(idx_clean, idx_noise):
            clean = eeg_pool[i_clean].astype(float)
            noise = noise_pool[i_noise].astype(float)
            if target_len is not None and int(target_len) > 1:
                clean = _resample_1d(clean, int(target_len))
                noise = _resample_1d(noise, int(target_len))

            noisy, lam = mix_with_target_snr(clean, noise, target_snr)
            lambdas.append(lam)

            # EEGdenoiseNet Eq. (4): normalize both clean/noisy by std(noisy).
            norm_scale = float(np.std(noisy) + 1e-12)
            noisy_norm = noisy / norm_scale

            info = mne.create_info(["Cz"], sfreq=sfreq, ch_types=["eeg"])
            raw_noisy = mne.io.RawArray(noisy_norm[np.newaxis, :], info, verbose="ERROR")
            if capture_filter_diagnostics:
                filter_plan = inspect_filter_plan(
                    raw_noisy,
                    l_freq=0.5,
                    h_freq=100.0,
                    freqs=[50.0, 60.0],
                    config=filter_config or BENCHMARK_FILTER_CONFIG,
                )
                filter_diag_rows.append(
                    (
                        float(filter_plan["drift_ratio"]),
                        float(filter_plan["emg_ratio"]),
                        float(filter_plan["selected_l_freq"]),
                        float(filter_plan["selected_h_freq"]),
                        float(len(filter_plan["selected_notch_freqs"])),
                        1.0 if bool(filter_plan["adaptive_l_raised"]) else 0.0,
                        1.0 if bool(filter_plan["adaptive_h_lowered"]) else 0.0,
                        1.0 if bool(filter_plan.get("severe_ocular_mode", False)) else 0.0,
                        1.0 if bool(filter_plan.get("severe_myogenic_mode", False)) else 0.0,
                    )
                )
            denoised = denoise_fn(raw_noisy, cfg, noise_label).get_data()[0] * norm_scale

            den_band_rows.append(_band_dict_to_vector(band_power_ratios(denoised, sfreq)))
            clean_band_rows.append(_band_dict_to_vector(band_power_ratios(clean, sfreq)))
            noisy_band_rows.append(_band_dict_to_vector(band_power_ratios(noisy, sfreq)))

            err_in = noisy - clean
            err_out = denoised - clean

            in_snr = snr_db(clean, err_in)
            out_snr = snr_db(clean, err_out)
            rmse = float(np.sqrt(np.mean(err_out ** 2)))
            corr = cc(clean, denoised)
            rrmse_t = rrmse_temporal(clean, denoised)
            rrmse_s = rrmse_spectral_psd(clean, denoised, sfreq)
            # Accuracy-like denoising score in [0,1] where 1 means perfect recovery.
            acc_like = float(max(0.0, 1.0 - (np.linalg.norm(err_out) + 1e-12) / (np.linalg.norm(err_in) + 1e-12)))

            rows.append((in_snr, out_snr, out_snr - in_snr, rmse, corr, acc_like, rrmse_t, rrmse_s))

        arr = np.asarray(rows, dtype=float)
        den_band_arr = np.asarray(den_band_rows, dtype=float)
        clean_band_arr = np.asarray(clean_band_rows, dtype=float)
        noisy_band_arr = np.asarray(noisy_band_rows, dtype=float)
        row_out = {
            "target_snr_db": float(target_snr),
            "lambda_mean": float(np.mean(np.asarray(lambdas, dtype=float))),
            "snr_in_mean_db": float(np.mean(arr[:, 0])),
            "snr_out_mean_db": float(np.mean(arr[:, 1])),
            "snr_improvement_db": float(np.mean(arr[:, 2])),
            "rmse_mean": float(np.mean(arr[:, 3])),
            "pearson_r_mean": float(np.mean(arr[:, 4])),
            "accuracy_like_mean": float(np.mean(arr[:, 5])),
            "rrmse_temporal_mean": float(np.mean(arr[:, 6])),
            "rrmse_spectral_mean": float(np.mean(arr[:, 7])),
            "cc_mean": float(np.mean(arr[:, 4])),
            "band_ratio_denoised_mean": _vector_to_band_dict(np.mean(den_band_arr, axis=0)),
            "band_ratio_clean_mean": _vector_to_band_dict(np.mean(clean_band_arr, axis=0)),
            "band_ratio_noisy_mean": _vector_to_band_dict(np.mean(noisy_band_arr, axis=0)),
        }

        if capture_filter_diagnostics and filter_diag_rows:
            diag = np.asarray(filter_diag_rows, dtype=float)
            row_out["filter_diagnostics"] = {
                "drift_ratio_mean": float(np.mean(diag[:, 0])),
                "emg_ratio_mean": float(np.mean(diag[:, 1])),
                "selected_l_freq_mean": float(np.mean(diag[:, 2])),
                "selected_h_freq_mean": float(np.mean(diag[:, 3])),
                "selected_notch_count_mean": float(np.mean(diag[:, 4])),
                "adaptive_l_raised_rate": float(np.mean(diag[:, 5])),
                "adaptive_h_lowered_rate": float(np.mean(diag[:, 6])),
                "severe_ocular_mode_rate": float(np.mean(diag[:, 7])),
                "severe_myogenic_mode_rate": float(np.mean(diag[:, 8])),
            }

        results.append(row_out)

    return {
        "noise_type": noise_label,
        "n_trials_per_snr": int(n_trials),
        "results": results,
    }


def denoise_full_pipeline(raw_noisy: mne.io.Raw, cfg: FuzzyArtifactConfig, _noise_label: str | None = None) -> mne.io.Raw:
    return clean_raw_signal(
        raw_noisy,
        fuzzy_config=cfg,
        asr_config=BENCHMARK_ASR_CONFIG,
        filter_config=BENCHMARK_FILTER_CONFIG,
    )


def denoise_filters_only(raw_noisy: mne.io.Raw, _cfg: FuzzyArtifactConfig, _noise_label: str | None = None) -> mne.io.Raw:
    raw_filtered = apply_filters(
        raw_noisy.copy(),
        l_freq=0.5,
        h_freq=100.0,
        freqs=[50.0, 60.0],
        config=BENCHMARK_FILTER_CONFIG,
    )
    return apply_asr_subspace_cleaning(raw_filtered, config=BENCHMARK_ASR_CONFIG)


def denoise_fuzzy_only(raw_noisy: mne.io.Raw, cfg: FuzzyArtifactConfig, _noise_label: str | None = None) -> mne.io.Raw:
    raw_copy = raw_noisy.copy()
    cleaned, _ = remove_fuzzy_artifacts_raw(raw_copy, config=cfg)
    return cleaned


def denoise_paper_filter_baseline(
    raw_noisy: mne.io.Raw,
    _cfg: FuzzyArtifactConfig,
    noise_label: str | None = None,
) -> mne.io.Raw:
    """Paper-style traditional filtering baseline.

    - Ocular artifacts: high-pass at 12 Hz.
    - Myogenic artifacts: band-pass at 12-40 Hz.
    """
    label = (noise_label or "").strip().upper()
    if label == "EOG":
        return apply_filters(raw_noisy.copy(), l_freq=12.0, h_freq=100.0, freqs=[], config=BENCHMARK_FILTER_CONFIG)
    return apply_filters(raw_noisy.copy(), l_freq=12.0, h_freq=40.0, freqs=[], config=BENCHMARK_FILTER_CONFIG)


def _summarize_method(method_report: dict[str, object]) -> dict[str, float | str]:
    entries = []
    for eval_item in method_report["evaluations"]:
        for row in eval_item["results"]:
            entries.append(row)
    if not entries:
        return {
            "method_name": str(method_report["method_name"]),
            "snr_improvement_db_mean": 0.0,
            "rrmse_temporal_mean": 0.0,
            "rrmse_spectral_mean": 0.0,
            "cc_mean": 0.0,
        }

    snr_imp = float(np.mean([float(x["snr_improvement_db"]) for x in entries]))
    rrmse_t = float(np.mean([float(x["rrmse_temporal_mean"]) for x in entries]))
    rrmse_s = float(np.mean([float(x["rrmse_spectral_mean"]) for x in entries]))
    cc_mean = float(np.mean([float(x["cc_mean"]) for x in entries]))
    return {
        "method_name": str(method_report["method_name"]),
        "snr_improvement_db_mean": snr_imp,
        "rrmse_temporal_mean": rrmse_t,
        "rrmse_spectral_mean": rrmse_s,
        "cc_mean": cc_mean,
    }


def _interpret_against_published_benchmarks(method_reports: list[dict[str, object]]) -> dict[str, object]:
    # Paper summary provided by user: DL methods outperform traditional ones,
    # especially at low SNR; performance decreases as SNR decreases.
    summaries = [_summarize_method(m) for m in method_reports]
    by_name = {str(s["method_name"]): s for s in summaries}

    full = by_name.get("full_pipeline")
    fuzzy = by_name.get("fuzzy_only")
    filt = by_name.get("filters_only")

    if full is None:
        return {
            "verdict": "insufficient_data",
            "message": "Full pipeline results are missing; cannot compare to published benchmarks.",
            "method_summaries": summaries,
        }

    # Heuristic verdict:
    # - Strong DL-like performance would usually show robust low RRMSE and high CC with
    #   consistently non-negative SNR improvement over tested SNR levels.
    # - If high-SNR degradation is present and average SNR gain is modest, classify as below DL claims.
    full_snr_gain = float(full["snr_improvement_db_mean"])
    full_cc = float(full["cc_mean"])
    full_rrmse_t = float(full["rrmse_temporal_mean"])
    full_rrmse_s = float(full["rrmse_spectral_mean"])

    likely_below_dl = (full_snr_gain < 0.35) or (full_cc < 0.6) or (full_rrmse_t > 0.8)
    if likely_below_dl:
        verdict = "below_reported_dl_benchmarks"
        summary_text = (
            "The NeuroHTz traditional pipeline does not appear to outperform the deep-learning benchmarks "
            "described in the reference text. Results are more consistent with traditional-method behavior."
        )
    else:
        verdict = "competitive_with_reported_dl_benchmarks"
        summary_text = (
            "The NeuroHTz pipeline appears competitive with the reported DL behavior under the tested setup, "
            "but direct claim of superiority requires the same dataset splits and exact model protocols."
        )

    component_note = ""
    if fuzzy is not None and filt is not None:
        fuzzy_gain = float(fuzzy["snr_improvement_db_mean"])
        filt_gain = float(filt["snr_improvement_db_mean"])
        if filt_gain >= fuzzy_gain:
            component_note = "Ablation indicates filtering contributes more than fuzzy removal to average SNR gain."
        else:
            component_note = "Ablation indicates fuzzy removal contributes more than filtering to average SNR gain."

    return {
        "verdict": verdict,
        "summary": summary_text,
        "component_note": component_note,
        "method_summaries": summaries,
        "published_reference_context": {
            "source": "User-provided EEGdenoiseNet benchmark description",
            "claims_used": [
                "DL models outperform traditional methods on RRMSEtemporal, RRMSEspectral, and CC",
                "Performance decreases as SNR decreases",
                "Traditional methods show larger degradation at lower SNR",
            ],
            "limitation": "RRMSE/CC comparison remains heuristic; separate table-based band-ratio comparison is provided in this report.",
        },
        "full_pipeline_summary": {
            "snr_improvement_db_mean": full_snr_gain,
            "rrmse_temporal_mean": full_rrmse_t,
            "rrmse_spectral_mean": full_rrmse_s,
            "cc_mean": full_cc,
        },
    }


def _aggregate_band_ratio_per_method_noise(method_reports: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for method in method_reports:
        mname = str(method["method_name"])
        out[mname] = {}
        for ev in method["evaluations"]:
            ntype = str(ev["noise_type"])
            rows = ev["results"]
            if not rows:
                out[mname][ntype] = _vector_to_band_dict(np.zeros(len(BAND_ORDER), dtype=float))
                continue
            mats = np.asarray([_band_dict_to_vector(r["band_ratio_denoised_mean"]) for r in rows], dtype=float)
            out[mname][ntype] = _vector_to_band_dict(np.mean(mats, axis=0))
    return out


def _compare_band_ratios_to_tables(method_reports: list[dict[str, object]]) -> dict[str, object]:
    ours = _aggregate_band_ratio_per_method_noise(method_reports)
    table = PUBLISHED_BAND_RATIOS
    comparisons: dict[str, object] = {}

    for noise_type in ("EOG", "EMG"):
        gt = table[noise_type]["ground truth"]
        contaminated = table[noise_type]["contaminated"]
        filt_ref = table[noise_type]["Filter"]
        emd_ref = table[noise_type]["EMD"]
        dl_refs = [table[noise_type][k] for k in ("FCNN", "Simple CNN", "Complex CNN", "RNN")]

        method_cmp: dict[str, object] = {}
        for mname, per_noise in ours.items():
            if noise_type not in per_noise:
                continue
            ours_ratio = per_noise[noise_type]
            dist_gt = band_ratio_l1(ours_ratio, gt)
            dist_cont = band_ratio_l1(ours_ratio, contaminated)
            dist_filter = band_ratio_l1(ours_ratio, filt_ref)
            dist_emd = band_ratio_l1(ours_ratio, emd_ref)
            best_dl_dist = float(min(band_ratio_l1(ours_ratio, d) for d in dl_refs))

            method_cmp[mname] = {
                "our_band_ratio": ours_ratio,
                "distance_to_ground_truth_l1": dist_gt,
                "distance_to_contaminated_l1": dist_cont,
                "distance_to_published_filter_l1": dist_filter,
                "distance_to_published_emd_l1": dist_emd,
                "distance_to_best_published_dl_l1": best_dl_dist,
                "closer_to_ground_truth_than_contaminated": bool(dist_gt < dist_cont),
            }

        # A practical headline for this noise type.
        full_info = method_cmp.get("full_pipeline")
        if full_info is None:
            headline = "full_pipeline_missing"
        else:
            full_gt = float(full_info["distance_to_ground_truth_l1"])
            full_filter = float(full_info["distance_to_published_filter_l1"])
            full_best_dl = float(full_info["distance_to_best_published_dl_l1"])
            if full_gt < full_filter and full_gt <= full_best_dl:
                headline = "full_pipeline_band_profile_close_to_ground_truth"
            elif full_gt < full_filter:
                headline = "full_pipeline_band_profile_better_than_filter_but_not_best_dl"
            else:
                headline = "full_pipeline_band_profile_not_better_than_filter_reference"

        comparisons[noise_type] = {
            "headline": headline,
            "published_reference_rows": table[noise_type],
            "method_comparisons": method_cmp,
        }

    return {
        "bands": list(BAND_ORDER),
        "notes": "Distances are mean absolute differences across normalized band-power ratios.",
        "comparisons": comparisons,
    }


def main() -> None:
    mne.set_log_level("ERROR")

    data_root = Path(r"C:/Users/Admin/Documents/GitHub/EEGdenoiseNet/data")
    eeg = np.load(data_root / "EEG_all_epochs.npy")
    eog = np.load(data_root / "EOG_all_epochs.npy")
    emg = np.load(data_root / "EMG_all_epochs.npy")

    sfreq_eog = 256.0
    sfreq_emg = 512.0
    snr_levels_by_noise = {
        "EOG": [float(v) for v in range(-7, 3)],
        "EMG": [float(v) for v in range(-7, 5)],
    }
    n_trials = 20
    seed = 42

    cfg = FuzzyArtifactConfig(
        swt_level=3,
        cmse_scale=2,
        max_entropy_points=256,
        prescreen_enabled=True,
        quick_prescreen_enabled=True,
        fuzzy_per_coeff_denoise=False,
        parallel_channels=False,
        artifact_score_threshold=0.60,
        min_denoise_strength=0.12,
        max_denoise_strength=0.74,
        severe_epoch_ptp_ratio=5.5,
        severe_hf_ratio=0.5,
        severe_lf_ratio=0.9,
        severe_strength_boost=0.06,
        severe_swt_level_boost=0,
        severe_approx_denoise=False,
    )

    methods = [
        ("paper_filter_baseline", denoise_paper_filter_baseline, "paper baseline: EOG HPF12, EMG BPF12-40", True, False),
        ("filters_only", denoise_filters_only, "bandpass+notch+asr", True, True),
        ("fuzzy_only", denoise_fuzzy_only, "fuzzy artifact suppression only", False, False),
        ("full_pipeline", denoise_full_pipeline, "bandpass+notch+asr+fuzzy", True, True),
    ]

    method_reports: list[dict[str, object]] = []
    for method_name, denoise_fn, description, uses_filter, uses_asr in methods:
        method_rng = np.random.default_rng(seed)
        method_reports.append(
            {
                "method_name": method_name,
                "description": description,
                "uses_filter_stage": uses_filter,
                "uses_asr_stage": uses_asr,
                "filter_config": asdict(BENCHMARK_FILTER_CONFIG) if uses_filter else None,
                "fuzzy_config": asdict(cfg),
                "asr_config": asdict(BENCHMARK_ASR_CONFIG) if uses_asr else None,
                "evaluations": [
                    evaluate_noise_type(
                        eeg,
                        eog,
                        "EOG",
                        snr_levels_by_noise["EOG"],
                        n_trials,
                        sfreq_eog,
                        cfg,
                        denoise_fn,
                        uses_filter,
                        method_rng,
                        target_len=512,
                        filter_config=BENCHMARK_FILTER_CONFIG,
                    ),
                    evaluate_noise_type(
                        eeg,
                        emg,
                        "EMG",
                        snr_levels_by_noise["EMG"],
                        n_trials,
                        sfreq_emg,
                        cfg,
                        denoise_fn,
                        uses_filter,
                        method_rng,
                        target_len=1024,
                        filter_config=BENCHMARK_FILTER_CONFIG,
                    ),
                ],
            }
        )

    report = {
        "benchmark": "EEGdenoiseNet: noisy vs ground-truth clean EEG",
        "study_type": "Ablation: paper/filter/fuzzy/full pipeline with ASR-enhanced filtering",
        "metrics": [
            "SNR_in",
            "SNR_out",
            "SNR_improvement",
            "RMSE",
            "Pearson_r",
            "accuracy_like",
            "RRMSE_temporal",
            "RRMSE_spectral",
            "CC",
        ],
        "sfreq_assumed": {"EOG": sfreq_eog, "EMG": sfreq_emg},
        "protocol_alignment": {
            "normalization_eq4": "Applied (xhat=x/std(y), yhat=y/std(y); output rescaled)",
            "snr_levels_db": snr_levels_by_noise,
            "segment_length_seconds": 2,
            "emg_upsampling": "EEG and artifact segments resampled to 1024 samples for EMG branch",
            "rrmse_spectral": "PSD-based in 0-120 Hz (fft-length=segment length)",
            "band_ratio": "Ratios over 1-80 Hz with delta/theta/alpha/beta/gamma from paper",
        },
        "methods": method_reports,
    }
    report["interpretation_against_given_benchmarks"] = _interpret_against_published_benchmarks(method_reports)
    report["band_ratio_table_comparison"] = _compare_band_ratios_to_tables(method_reports)

    out_path = ROOT / "comparisons" / "eegdenoisenet_filter_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved benchmark report: {out_path}")


if __name__ == "__main__":
    main()
