#!/usr/bin/env python3
"""
mmWave Backscatter Tag – Multi-Tag Trajectory Estimation & Evaluation

Processes raw mmWave radar data to estimate the 3-D trajectories of multiple
backscatter tags simultaneously.  Each estimated tag trajectory is
assigned to its matching motion-capture ground-truth stream via modulation
frequency and evaluated against ground truth.

Pipeline
--------
1. Phase-based harmonic pairing identifies which spectral peak pairs belong
   to the same physical tag.
2. Phase-rate modulation-frequency estimation characterises each tag's modulation
   frequency, which is cross-referenced against ``mod_freqs.txt`` to assign
   each estimated tag to its named ground-truth stream.
3. Each tag trajectory is evaluated against its assigned
   ground-truth stream.

Usage
-----
    python estimate_trajectory_multitag.py \\
        --traj_dir /path/to/trajectory [options]

Expected data layout inside *traj_dir*::

    <traj_dir>/
    ├── radar/dump/rf_data.npy      # (frames, Tx, Rx, ADC_samples)
    ├── radar/dump/rf_ts.txt        # one timestamp per frame (seconds)
    ├── rosbag/rosbag.csv           # motion-capture ground truth
    ├── rosbag/moving_times.txt     # (optional) per-tag moving intervals
    ├── mod_freqs.txt               # tag-name → modulation frequency (MHz)
    └── timestamp_offset.npy        # (optional) radar–mocap time offset (scalar index)

mod_freqs.txt format::

    All in MHz
    yellow_block: 0.5
    green_block: 1.4
    red_block: 1.6

moving_times.txt format::

    All units in s
    yellow_block: 3-8
    green_block: 20-26,30-35
    red_block: 10-17

Required config.json keys
--------------------------
    ADC_Samples, fft_zeropad, peak_finding_startbin, left_discard,
    right_discard, num_tags, window_size, peak_finding_reference_error_thresh,
    f_start, f_stop, vertical_antenna_index, horizontal_antenna_index,
    distance_no_rel, distance_resolution, adc_rate,
    gt_tag_names  – list of tag names present in rosbag.csv,
                    e.g. ["yellow_block", "green_block", "red_block"]

Memory efficiency
-----------------
Only two single-channel masked IFFTs are live at any moment, keeping peak RAM
to roughly ``2 × rf_fft`` regardless of tag count.
"""

import argparse
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from core.signal_processing import (
    AOA,
    distance_calculation,
    modulation_freq_from_phase_rate,
    peak_finding,
    phase_based_tag_matching,
    signal_masked_by_window,
    snr_linear,
)
from core.kalman import (
    abrupt_check,
    gaussian_window,
    kalman_polar,
    timestamp_find,
)
from core.metrics import cartesian_to_polar, get_error, polar_to_cartesian

# GT tag name excluded from processing (mechanical arm, not a tracked object)
_EXCLUDED_TAG = "robo_arm"


# ── data loading ───────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return json.load(f)


def load_multitag_data(traj_dir, gt_tag_names):
    """
    Load radar captures and Vicon ground-truth streams.

    Returns
    -------
    rf_data    : ndarray (frames, Tx, Rx, ADC_samples)
    t_rf       : 1-D array of radar timestamps (seconds)
    gt_streams : list of (t_gt, x_gt, y_gt, z_gt) – one entry per tag name
    """
    rf_data = np.load(
        os.path.join(traj_dir, "radar", "dump", "rf_data.npy"),
        allow_pickle=True,
    )
    t_rf = np.loadtxt(os.path.join(traj_dir, "radar", "dump", "rf_ts.txt"))

    gt = pd.read_csv(os.path.join(traj_dir, "rosbag", "rosbag.csv"))
    offset_path = os.path.join(traj_dir, "timestamp_offset.npy")
    ts_offset = (np.load(offset_path).item() * 0.01
                 if os.path.exists(offset_path) else 0.0)

    gt_streams = []
    for name in gt_tag_names:
        df = gt[gt["tag_name"] == name]
        gt_streams.append((
            df["time"].values - ts_offset,
            df["X_in_radar"].values,
            df["Y_in_radar"].values,
            df["Z_in_radar"].values,
        ))

    return rf_data, t_rf, gt_streams


def load_mod_freqs(traj_dir):
    """
    Load ``mod_freqs.txt`` from *traj_dir*.

    Returns
    -------
    dict mapping tag_name → modulation frequency in Hz.
    """
    path = os.path.join(traj_dir, "mod_freqs.txt")
    if not os.path.exists(path):
        sys.exit(f"Error: mod_freqs.txt not found in {traj_dir}")
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("all"):
                continue
            name, _, val = line.partition(":")
            name, val = name.strip(), val.strip()
            if name and val:
                result[name] = float(val) * 1e6   # MHz → Hz
    return result


def load_moving_times(traj_dir):
    """
    Load ``rosbag/moving_times.txt``.

    Returns
    -------
    dict mapping tag_name → list of (t_start, t_end) tuples (seconds,
    relative to trajectory start).  Empty dict if the file does not exist.
    """
    path = os.path.join(traj_dir, "rosbag", "moving_times.txt")
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("all"):
                continue
            name, _, intervals_str = line.partition(":")
            name = name.strip()
            intervals = []
            for part in intervals_str.split(","):
                part = part.strip()
                if "-" in part:
                    lo, _, hi = part.partition("-")
                    try:
                        intervals.append((float(lo.strip()), float(hi.strip())))
                    except ValueError:
                        pass
            if name and intervals:
                result[name] = intervals
    return result


def _abs_intervals(rel_intervals, t_start):
    """Convert relative-time intervals to absolute by adding *t_start*."""
    return [(t_start + r_lo, t_start + r_hi) for r_lo, r_hi in rel_intervals]


def _moving_time_mask(t, intervals):
    """Boolean mask – True where *t* falls within any (t_start, t_end) interval."""
    mask = np.zeros(len(t), dtype=bool)
    for t_s, t_e in intervals:
        mask |= (t >= t_s) & (t <= t_e)
    return mask


# ── signal helpers ─────────────────────────────────────────────────────────

def _snr_for_channel(rf_fft_abs, peaks, ch_idx):
    """Per-frame mean linear SNR for spectral channel *ch_idx*."""
    n_frames, n_tx, n_rx = rf_fft_abs.shape[:3]
    snr_arr = np.zeros((n_frames, n_tx, n_rx))
    for tx in range(n_tx):
        for rx in range(n_rx):
            for i in range(n_frames):
                pk = peaks[i, tx, rx, ch_idx]
                snr_arr[i, tx, rx] = snr_linear(rf_fft_abs[i, tx, rx], pk)
    return np.mean(snr_arr, axis=(1, 2))   # (frames,)


def _masked_ifft_no_bg(rf_fft, peaks, ch_idx, window_bins):
    """
    Background-subtracted masked IFFT for a single harmonic channel.

    Returns ndarray of shape ``(frames, Tx, Rx, 1, N_adc)``.
    """
    masked = signal_masked_by_window(
        rf_fft, peaks[..., ch_idx:ch_idx + 1], window_bins)
    ifft = np.fft.ifft(masked, axis=-1)
    return ifft - np.mean(ifft, axis=0)


# ── main pipeline ──────────────────────────────────────────────────────────

def run(traj_dir, config, output_dir, max_frames=None):
    """End-to-end multi-tag estimation, GT assignment, and plotting."""

    # -- unpack config --
    ADC_sample    = config["ADC_Samples"]
    fft_zeropad   = config["fft_zeropad"]
    pk_startbin   = config["peak_finding_startbin"]
    left_discard  = config["left_discard"]
    right_discard = config["right_discard"]
    num_tags      = config["num_tags"]
    window_size   = config["window_size"]
    pk_ref_thresh = config["peak_finding_reference_error_thresh"]
    f_start       = config["f_start"]
    f_stop        = config["f_stop"]
    vert_idx      = config["vertical_antenna_index"]
    horiz_idx     = config["horizontal_antenna_index"]
    dist_no_rel   = config["distance_no_rel"]
    dist_res      = config["distance_resolution"]
    adc_rate      = config["adc_rate"]
    gt_tag_names  = config["gt_tag_names"]
    motion_pct    = config.get("motion_align_percentile", 70)
    aoa_res       = 0.1

    # -- load data --
    print("Loading data …")
    rf_data, t_rf, gt_streams = load_multitag_data(traj_dir, gt_tag_names)
    assert rf_data.shape[-1] == ADC_sample, (
        f"ADC mismatch: data={rf_data.shape[-1]}, config={ADC_sample}")

    if max_frames is not None:
        rf_data = rf_data[:max_frames]
        t_rf    = t_rf[:max_frames]

    n_frames = rf_data.shape[0]

    # -- auxiliary files --
    mod_freqs    = load_mod_freqs(traj_dir)
    moving_times = load_moving_times(traj_dir)

    # Remove the robotic arm from GT candidates – it is not a tracked tag.
    # Keep it in mod_freqs so its RF tag still matches to it (and gets skipped)
    # rather than being misassigned to the nearest real tag.
    gt_streams   = [s for n, s in zip(gt_tag_names, gt_streams)
                    if n != _EXCLUDED_TAG]
    gt_tag_names = [n for n in gt_tag_names if n != _EXCLUDED_TAG]

    print(f"  GT streams ({len(gt_tag_names)}): {gt_tag_names}")
    print(f"  Modulation frequencies: "
          + ", ".join(f"{n}={f/1e3:.1f} kHz" for n, f in mod_freqs.items()))

    # -- pre-process --
    print("Preprocessing …")
    rf_data  = rf_data[:, :, :, left_discard:ADC_sample - right_discard]
    rf_no_bg = rf_data - np.mean(rf_data, axis=0)
    del rf_data

    N_adc  = rf_no_bg.shape[-1]
    rf_fft = np.fft.fft(rf_no_bg, n=fft_zeropad * N_adc, axis=-1)
    del rf_no_bg

    rf_fft_abs      = np.abs(rf_fft)
    rf_fft_abs_mean = np.mean(rf_fft_abs, axis=(0, 1, 2))
    half = ADC_sample // 2

    # -- reference peak detection --
    print("Finding reference peaks …")
    ref_left = peak_finding(
        rf_fft_abs_mean, num_tags,
        startbin=pk_startbin * fft_zeropad,
        stopbin=(half - pk_startbin) * fft_zeropad,
        distance=pk_ref_thresh * fft_zeropad)
    ref_right = peak_finding(
        rf_fft_abs_mean, num_tags,
        startbin=(half + pk_startbin) * fft_zeropad,
        stopbin=(ADC_sample - pk_startbin) * fft_zeropad,
        distance=pk_ref_thresh * fft_zeropad)
    ref_peaks = np.concatenate([ref_left, ref_right])

    # -- per-frame peak tracking --
    print("Tracking peaks …")
    peaks = peak_finding(rf_fft_abs, 2 * num_tags,
                         reference=ref_peaks,
                         reference_error_thresh=5 * fft_zeropad)

    # -- phase-based harmonic pairing --
    print("Determining tag pairings …")
    window_bins = window_size * fft_zeropad

    subset_n  = min(100, n_frames)
    tx_p      = min(1, rf_fft.shape[1] - 1)
    rx_p      = min(2, rf_fft.shape[2] - 1)
    sub_fft   = rf_fft[:subset_n, tx_p:tx_p + 1, rx_p:rx_p + 1, :]
    sub_peaks = peaks[:subset_n, tx_p:tx_p + 1, rx_p:rx_p + 1, :]
    sub_masked = signal_masked_by_window(sub_fft, sub_peaks, window_bins)
    sub_ifft   = np.fft.ifft(sub_masked, axis=-1)
    sub_ifft  -= np.mean(sub_ifft, axis=0)
    sub_ifft_sq = sub_ifft.squeeze(axis=(1, 2))   # (subset_n, 2*num_tags, N_adc)
    del sub_fft, sub_peaks, sub_masked, sub_ifft

    tag_matches, cost_mat = phase_based_tag_matching(sub_ifft_sq, num_tags)

    # -- modulation frequency via phase rate --
    phase_info = modulation_freq_from_phase_rate(
        sub_ifft_sq, tag_matches, adc_rate)
    del sub_ifft_sq

    # -- adjusted frequency bounds --
    bw             = f_stop - f_start
    f_start_adj    = f_start + bw / ADC_sample * left_discard
    f_stop_adj     = f_stop  - bw / ADC_sample * right_discard
    dist_res_round = (dist_res // fft_zeropad) * fft_zeropad

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    # ── per-tag processing ────────────────────────────────────────────────
    for tag_idx, (left_ch, right_ch) in enumerate(tag_matches):
        print(f"\n[Tag {tag_idx}]  channels {left_ch} ↔ {right_ch}")

        # -- GT assignment via closest modulation frequency --
        f_est   = phase_info[tag_idx]["f_mod_phase_hz"]
        gt_name = min(mod_freqs, key=lambda n: abs(mod_freqs[n] - f_est))
        f_match = mod_freqs[gt_name]
        print(f"  Frequency match: {f_est/1e3:.2f} kHz → "
              f"'{gt_name}' ({f_match/1e3:.2f} kHz, "
              f"Δ={abs(f_est - f_match)/1e3:.2f} kHz)")

        if gt_name == _EXCLUDED_TAG:
            print(f"  Matched excluded tag '{gt_name}' – skipping.")
            continue

        t_gt, x_gt, y_gt, z_gt = gt_streams[gt_tag_names.index(gt_name)]

        # -- masked IFFTs --
        ifft_left  = _masked_ifft_no_bg(rf_fft, peaks, left_ch,  window_bins)
        ifft_right = _masked_ifft_no_bg(rf_fft, peaks, right_ch, window_bins)
        mixed = ifft_left * ifft_right
        guide = ifft_right
        snr_avg = _snr_for_channel(rf_fft_abs, peaks, left_ch)

        # -- distance --
        print("  Estimating distance …")
        _, delta_d = distance_calculation(
            mixed,
            f_start=f_start_adj, f_stop=f_stop_adj,
            no_rel=dist_no_rel, step=20, f_interval=20,
            fft_zeropad=fft_zeropad, distance_resolution=dist_res_round,
            bw_ratio=1, average=1)
        chirp_d = distance_calculation(
            mixed,
            f_start=f_start_adj, f_stop=f_stop_adj,
            no_rel=True, step=120,
            fft_zeropad=1, distance_resolution=dist_res_round,
            bw_ratio=1, average=1)
        distance_mean = np.mean(delta_d, axis=(1, 2)) + np.mean(chirp_d)

        # -- angle of arrival --
        print("  Estimating AoA …")
        azi_guide_pk, azi_guide_spec = AOA(
            guide, antenna_index=horiz_idx[2:],
            resolution=aoa_res, d=0.5, radar="AOP", guide=True)
        elev_guide_pk, elev_guide_spec = AOA(
            guide, antenna_index=vert_idx[2:],
            resolution=aoa_res, d=0.5, radar="AOP", guide=True)
        _, azi_spec = AOA(
            mixed, antenna_index=horiz_idx,
            resolution=aoa_res, d=1, radar="AOP", guide=False)
        _, elev_spec = AOA(
            mixed, antenna_index=vert_idx,
            resolution=aoa_res, d=1, radar="AOP", guide=False)

        for i in range(n_frames):
            a_lo = int(max(azi_guide_pk.flat[i]  - 10, 0)   / aoa_res)
            a_hi = int(min(azi_guide_pk.flat[i]  + 10, 180) / aoa_res)
            e_lo = int(max(elev_guide_pk.flat[i] - 10, 0)   / aoa_res)
            e_hi = int(min(elev_guide_pk.flat[i] + 10, 180) / aoa_res)
            azi_spec[i,  :, :a_lo] = 1e-21
            azi_spec[i,  :, a_hi:] = 1e-21
            elev_spec[i, :, :e_lo] = 1e-21
            elev_spec[i, :, e_hi:] = 1e-21

        azi_peak  = abrupt_check(np.argmax(azi_spec,  axis=-1) * aoa_res)
        elev_peak = abrupt_check(np.argmax(elev_spec, axis=-1) * aoa_res)
        azi_unc   = (np.sum( np.abs(azi_guide_spec)[:,  0, :], axis=1)
                     / np.max(np.abs(azi_guide_spec)[:,  0, :], axis=1))
        elev_unc  = (np.sum( np.abs(elev_guide_spec)[:, 0, :], axis=1)
                     / np.max(np.abs(elev_guide_spec)[:, 0, :], axis=1))

        del ifft_left, ifft_right, mixed, guide, azi_spec, elev_spec

        # -- time alignment with GT --
        t_gt_idx, t_rf_idx = timestamp_find(t_rf, t_gt)
        if len(t_gt_idx) == 0:
            print(f"  [WARNING] No overlapping timestamps with GT. Skipping.")
            continue

        dist_gt   = np.sqrt(x_gt ** 2 + y_gt ** 2 + z_gt ** 2)
        d_offset  = 0.080900
        distances = distance_mean - d_offset

        polar_pts = np.hstack((distances,
                               np.deg2rad(azi_peak),
                               np.deg2rad(elev_peak)))
        smoothed  = kalman_polar(polar_pts, snr_avg, azi_unc, elev_unc)[:, :3]
        smoothed  = gaussian_window(smoothed, 200)

        rf_aligned = smoothed[t_rf_idx, :3].copy()
        rf_aligned[:, 0] = distances[t_rf_idx, 0]
        rf_aligned = np.array(
            [polar_to_cartesian(d, a, e) for d, a, e in rf_aligned])

        gt_aligned = np.column_stack(
            (x_gt[t_gt_idx], y_gt[t_gt_idx], z_gt[t_gt_idx]))
        t_rf_al = t_rf[np.asarray(t_rf_idx)]
        t_gt_al = t_gt[np.asarray(t_gt_idx)]

        valid      = ~np.isnan(gt_aligned).any(axis=1)
        gt_aligned = gt_aligned[valid]
        rf_aligned = rf_aligned[valid]
        t_rf_al    = t_rf_al[valid]
        t_gt_al    = t_gt_al[valid]

        if len(gt_aligned) < 3:
            print(f"  [WARNING] Too few valid GT points. Skipping.")
            continue

        abs_intervals = (
            _abs_intervals(moving_times[gt_name], t_gt[0])
            if gt_name in moving_times else [])

        if abs_intervals:
            move_mask = _moving_time_mask(t_gt_al, abs_intervals)
            if move_mask.sum() >= 3:
                rot, _, _ = Rotation.align_vectors(
                    gt_aligned[move_mask], rf_aligned[move_mask],
                    return_sensitivity=True)
            else:
                rot, _, _ = Rotation.align_vectors(
                    gt_aligned, rf_aligned, return_sensitivity=True)
        else:
            gt_speed = np.linalg.norm(np.gradient(gt_aligned, axis=0), axis=1)
            pct_mask = gt_speed >= np.percentile(gt_speed, motion_pct)
            align_gt = gt_aligned[pct_mask] if pct_mask.sum() >= 3 else gt_aligned
            align_rf = rf_aligned[pct_mask] if pct_mask.sum() >= 3 else rf_aligned
            rot, _, _ = Rotation.align_vectors(
                align_gt, align_rf, return_sensitivity=True)

        rf_rot = rot.apply(rf_aligned)

        # -- error evaluation --
        if abs_intervals:
            eval_mask = _moving_time_mask(t_gt_al, abs_intervals)
            if eval_mask.sum() >= 3:
                gt_eval = gt_aligned[eval_mask]
                rf_eval = rf_rot[eval_mask]
            else:
                gt_eval = gt_aligned; rf_eval = rf_rot
                eval_mask = None
        else:
            gt_eval = gt_aligned; rf_eval = rf_rot
            eval_mask = None

        (mean_err, med_err, _, mean_dim, _,
         norm_err, dim_err) = get_error(gt_eval, rf_eval)

        print(f"  Mean  3-D error : {mean_err * 100:6.2f} cm")
        print(f"  Median 3-D error: {med_err  * 100:6.2f} cm")
        print(f"  Mean  per-axis  : "
              f"X={mean_dim[0]*100:.2f}  "
              f"Y={mean_dim[1]*100:.2f}  "
              f"Z={mean_dim[2]*100:.2f} cm")

        summary_rows.append(dict(
            tag         = tag_idx,
            channels    = f"{left_ch}↔{right_ch}",
            gt_name     = gt_name,
            mean_err_cm = mean_err * 100,
            med_err_cm  = med_err  * 100,
        ))

        # -- save results --
        tag_out = os.path.join(output_dir, gt_name)
        os.makedirs(tag_out, exist_ok=True)

        np.savez(
            os.path.join(tag_out, "trajectory_results.npz"),
            estimated    = rf_rot,
            ground_truth = gt_aligned,
            t_rf         = t_rf_al,
            t_gt         = t_gt_al,
            norm_errors  = norm_err,
            dim_errors   = dim_err,
        )

        result = dict(
            gt_name       = gt_name,
            estimated     = rf_rot,
            ground_truth  = gt_aligned,
            t_rf          = t_rf_al,
            t_gt          = t_gt_al,
            norm_errors   = norm_err,
            dim_errors    = dim_err,
            mean_err      = mean_err,
            med_err       = med_err,
            mean_dim      = mean_dim,
            dist_gt       = dist_gt,
            t_gt_full     = t_gt,
            x_gt          = x_gt,
            y_gt          = y_gt,
            z_gt          = z_gt,
            abs_intervals = abs_intervals,
            eval_mask     = eval_mask,
        )
        _save_figures(tag_idx, result, tag_out)

    # -- summary table --
    print(f"\n{'=' * 60}")
    print(f"  {'Tag':>4}  {'Channels':>10}  {'GT stream':>14}  "
          f"{'Mean (cm)':>9}  {'Median (cm)':>11}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*14}  {'-'*9}  {'-'*11}")
    for row in summary_rows:
        print(f"  {row['tag']:>4}  {row['channels']:>10}  "
              f"{row['gt_name']:>14}  "
              f"{row['mean_err_cm']:>9.2f}  {row['med_err_cm']:>11.2f}")
    print(f"{'=' * 60}\n")
    print(f"Results saved to {output_dir}")


# ── figures ────────────────────────────────────────────────────────────────

def _shade_intervals(ax, intervals, label="Moving", color="green", alpha=0.15):
    """Shade every (t_start, t_end) interval on *ax*."""
    for k, (t_s, t_e) in enumerate(intervals):
        ax.axvspan(t_s, t_e, alpha=alpha, color=color,
                   label=label if k == 0 else "_nolegend_")


def _save_figures(tag_idx, r, out_dir):
    """Save four diagnostic figures for one tag."""
    rf_r      = r["estimated"]
    gt_a      = r["ground_truth"]
    t_rf      = r["t_rf"]
    t_gt      = r["t_gt"]
    ne        = r["norm_errors"]
    x_gt      = r["x_gt"]
    y_gt      = r["y_gt"]
    z_gt      = r["z_gt"]
    dist_gt   = r["dist_gt"]
    t_gt_f    = r["t_gt_full"]
    intervals = r["abs_intervals"]
    eval_mask = r["eval_mask"]

    reproj  = np.array([cartesian_to_polar(x, y, z) for x, y, z in rf_r])
    azi_gt  = np.arctan2(x_gt, y_gt) / np.pi * 180
    elev_gt = np.arccos(-z_gt / (dist_gt + 1e-12)) / np.pi * 180

    title_prefix = f"Tag {tag_idx}  ({r['gt_name']})"

    # ── build filtered views for moving-only plots ────────────────────────
    if intervals:
        rf_mask   = _moving_time_mask(t_rf,   intervals)
        gt_mask   = _moving_time_mask(t_gt,   intervals)
        gtf_mask  = _moving_time_mask(t_gt_f, intervals)
        t_rf_p    = t_rf[rf_mask];   rf_r_p  = rf_r[rf_mask]
        t_gt_p    = t_gt[gt_mask];   gt_a_p  = gt_a[gt_mask]
        t_gtf_p   = t_gt_f[gtf_mask]
        repr_p    = reproj[rf_mask]
        dgt_p     = dist_gt[gtf_mask]
        azi_gt_p  = azi_gt[gtf_mask]
        elev_gt_p = elev_gt[gtf_mask]
    else:
        t_rf_p    = t_rf;   rf_r_p  = rf_r
        t_gt_p    = t_gt;   gt_a_p  = gt_a
        t_gtf_p   = t_gt_f
        repr_p    = reproj
        dgt_p     = dist_gt
        azi_gt_p  = azi_gt
        elev_gt_p = elev_gt

    # 1) Cartesian + polar time series (2 × 3)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title_prefix, fontsize=14)

    for c, lab in enumerate(["X", "Y", "Z"]):
        ax = axs[0, c]
        ax.plot(t_rf_p, rf_r_p[:, c], label="Estimated",    alpha=0.8)
        ax.plot(t_gt_p, gt_a_p[:, c], label="Ground Truth", alpha=0.8)
        ax.set(xlabel="Time (s)", ylabel="Position (m)", title=lab)
        ax.legend(); ax.grid(True, alpha=0.3)

    polar_est = [repr_p[:, 0],
                 -np.rad2deg(repr_p[:, 1]) + 90,
                 -np.rad2deg(repr_p[:, 2]) + 180]
    polar_gt  = [dgt_p, azi_gt_p, elev_gt_p]
    polar_lab = ["Range (m)", "Azimuth (deg)", "Elevation (deg)"]
    for c in range(3):
        ax = axs[1, c]
        ax.plot(t_rf_p,  polar_est[c], label="Estimated",    alpha=0.8)
        ax.plot(t_gtf_p, polar_gt[c],  label="Ground Truth", alpha=0.8)
        ax.set(xlabel="Time (s)", ylabel=polar_lab[c],
               title=polar_lab[c].split(" ")[0])
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_comparison.png"), dpi=150)
    plt.close()

    # 2) 3-D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    if intervals:
        # Only draw the moving segments
        for k, (t_s, t_e) in enumerate(intervals):
            mv_gt = (t_gt >= t_s) & (t_gt <= t_e)
            mv_rf = (t_rf >= t_s) & (t_rf <= t_e)
            lbl_gt = "Ground Truth" if k == 0 else "_nolegend_"
            lbl_rf = "Estimated"    if k == 0 else "_nolegend_"
            if mv_gt.any():
                ax.plot(*gt_a[mv_gt].T, "r-", alpha=0.7, label=lbl_gt)
                ax.scatter(*gt_a[mv_gt][0], c="r", marker="o", s=60, zorder=5,
                           label="Start (GT)" if k == 0 else "_nolegend_")
            if mv_rf.any():
                ax.plot(*rf_r[mv_rf].T, "b-", alpha=0.7, label=lbl_rf)
                ax.scatter(*rf_r[mv_rf][0], c="b", marker="o", s=60, zorder=5,
                           label="Start (est)" if k == 0 else "_nolegend_")
    else:
        ax.plot(*rf_r.T, "b-", alpha=0.7, label="Estimated")
        ax.plot(*gt_a.T, "r-", alpha=0.7, label="Ground Truth")
        ax.scatter(*rf_r[0], c="b", marker="o", s=60, zorder=5, label="Start (est)")
        ax.scatter(*gt_a[0], c="r", marker="o", s=60, zorder=5, label="Start (GT)")

    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax.set_title(title_prefix)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lb in zip(handles, labels):
        seen.setdefault(lb, h)
    ax.legend(seen.values(), seen.keys())
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_3d.png"), dpi=150)
    plt.close()

    # 3) Error histogram + time series
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(ne * 100, bins=30, edgecolor="k", alpha=0.7)
    ax1.axvline(r["mean_err"] * 100, color="r", ls="--",
                label=f"Mean {r['mean_err']*100:.1f} cm")
    ax1.axvline(r["med_err"]  * 100, color="g", ls="--",
                label=f"Median {r['med_err']*100:.1f} cm")
    ax1.set(xlabel="3-D Error (cm)", ylabel="Count",
            title=f"Error Distribution – {title_prefix}")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    t_err = t_rf[eval_mask] if eval_mask is not None else t_rf
    ax2.plot(t_err, ne * 100, alpha=0.7)
    if intervals:
        _shade_intervals(ax2, intervals, label="Moving")
        ax2.legend()
    ax2.set(xlabel="Time (s)", ylabel="3-D Error (cm)",
            title=f"Error Over Time – {title_prefix}")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_analysis.png"), dpi=150)
    plt.close()

    # 4) CDF
    fig, ax = plt.subplots(figsize=(8, 5))
    se  = np.sort(ne * 100)
    cdf = np.arange(1, len(se) + 1) / len(se)
    ax.plot(se, cdf, linewidth=2)
    p50 = np.percentile(ne * 100, 50)
    p90 = np.percentile(ne * 100, 90)
    ax.axhline(0.5, color="grey", ls=":", alpha=0.5)
    ax.axhline(0.9, color="grey", ls=":", alpha=0.5)
    ax.annotate(f"50th: {p50:.1f} cm", xy=(p50, 0.50), fontsize=10,
                color="tab:blue",
                xytext=(p50 + 1, 0.45), arrowprops=dict(arrowstyle="->"))
    ax.annotate(f"90th: {p90:.1f} cm", xy=(p90, 0.90), fontsize=10,
                color="tab:blue",
                xytext=(p90 + 1, 0.85), arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="3-D Error (cm)", ylabel="CDF",
           title=f"CDF of Tracking Error – {title_prefix}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_cdf.png"), dpi=150)
    plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="mmWave Backscatter – Multi-Tag Trajectory Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Required config.json keys
            -------------------------
            adc_rate          – ADC sampling rate in Hz
            num_tags          – number of RF tags
            gt_tag_names      – list of tag names in rosbag.csv,
                                e.g. ["yellow_block","green_block","red_block"]

            Results layout
            --------------
            <output_dir>/tag_0/trajectory_results.npz
            <output_dir>/tag_0/trajectory_comparison.png
            <output_dir>/tag_0/trajectory_3d.png
            <output_dir>/tag_0/error_analysis.png
            <output_dir>/tag_0/error_cdf.png
            (one sub-folder per tag)
        """),
    )
    parser.add_argument(
        "-t", "--traj_dir", required=True,
        help="Path to a trajectory data folder",
    )
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config_multitag.json"),
        help="Path to config JSON (default: config.json next to this script)",
    )
    parser.add_argument(
        "-o", "--output_dir", default=None,
        help="Output directory (default: <traj_dir>/results_multitag)",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Process at most this many radar frames (default: all)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.traj_dir):
        sys.exit(f"Error: directory not found – {args.traj_dir}")

    config     = load_config(args.config)
    output_dir = args.output_dir or os.path.join(
        args.traj_dir, "results_multitag")
    run(args.traj_dir, config, output_dir, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
