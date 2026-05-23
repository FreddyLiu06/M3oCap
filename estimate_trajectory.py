#!/usr/bin/env python3
"""
mmWave Backscatter Tag – Single-Tag Trajectory Estimation & Evaluation

Processes raw mmWave radar data to estimate the 3D trajectory of a
backscatter tag, and saves comparison figures and error metrics.

Usage
-----
    python estimate_trajectory.py --traj_dir /path/to/trajectory [options]

Expected data layout inside *traj_dir*::

    <traj_dir>/
    ├── radar/dump/rf_data.npy      # (frames, Tx, Rx, ADC_samples)
    ├── radar/dump/rf_ts.txt        # one timestamp per frame (seconds)
    └── rosbag/rosbag.csv           # motion-capture ground truth
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
    mix_signals_by_tag_matching,
    peak_finding,
    signal_masked_by_window,
    snr_db,
    snr_linear,
)
from core.kalman import (
    abrupt_check,
    gaussian_window,
    kalman_polar,
    timestamp_find,
)
from core.metrics import cartesian_to_polar, get_error, polar_to_cartesian


# ── helpers ────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return json.load(f)


def load_trajectory_data(traj_dir):
    """
    Load radar captures and Vicon ground truth from a trajectory folder.

    Returns
    -------
    rf_data : ndarray (frames, Tx, Rx, ADC_samples)
    t_rf    : 1-D array of radar timestamps
    t_gt, x_gt, y_gt, z_gt : 1-D arrays of ground-truth time / position
    """
    rf_data = np.load(
        os.path.join(traj_dir, "radar", "dump", "rf_data.npy"),
        allow_pickle=True,
    )
    t_rf = np.loadtxt(os.path.join(traj_dir, "radar", "dump", "rf_ts.txt"))

    gt = pd.read_csv(os.path.join(traj_dir, "rosbag", "rosbag.csv"))
    gt_rf = gt[gt["tag_name"] == "yellow_block"]

    return (
        rf_data,
        t_rf,
        gt_rf["time"].values,
        gt_rf["X_in_radar"].values,
        gt_rf["Y_in_radar"].values,
        gt_rf["Z_in_radar"].values,
    )


# ── main pipeline ─────────────────────────────────────────────────────────

def run(traj_dir, config, output_dir, max_frames=None):
    """End-to-end trajectory estimation, evaluation, and plotting."""

    # -- unpack config --
    ADC_sample     = config["ADC_Samples"]
    fft_zeropad    = config["fft_zeropad"]
    pk_startbin    = config["peak_finding_startbin"]
    left_discard   = config["left_discard"]
    right_discard  = config["right_discard"]
    num_tags       = config["num_tags"]
    window_size    = config["window_size"]
    pk_ref_thresh  = config["peak_finding_reference_error_thresh"]
    f_start        = config["f_start"]
    f_stop         = config["f_stop"]
    vert_idx       = config["vertical_antenna_index"]
    horiz_idx      = config["horizontal_antenna_index"]
    dist_no_rel    = config["distance_no_rel"]
    dist_res       = config["distance_resolution"]
    aoa_res        = 0.1          # fine angular resolution for MUSIC

    # -- load data --
    print("Loading data …")
    rf_data, t_rf, t_gt, x_gt, y_gt, z_gt = load_trajectory_data(traj_dir)
    assert rf_data.shape[-1] == ADC_sample, (
        f"ADC mismatch: data={rf_data.shape[-1]}, config={ADC_sample}")

    if max_frames is not None:
        rf_data = rf_data[:max_frames]
        t_rf = t_rf[:max_frames]

    # -- pre-process --
    print("Preprocessing …")
    rf_data = rf_data[:, :, :, left_discard:ADC_sample - right_discard]
    rf_no_bg = rf_data - np.mean(rf_data, axis=0)

    rf_fft = np.fft.fft(rf_no_bg,
                         n=fft_zeropad * rf_no_bg.shape[-1], axis=-1)
    rf_fft_abs_mean = np.mean(np.abs(rf_fft), axis=(0, 1, 2))

    # -- reference peak detection --
    print("Finding reference peaks …")
    half = ADC_sample // 2
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
    peaks = peak_finding(np.abs(rf_fft), 2 * num_tags,
                         reference=ref_peaks,
                         reference_error_thresh=5 * fft_zeropad)

    # -- per-frame SNR (linear & dB) --
    print("Computing SNR …")
    n_frames = rf_fft.shape[0]
    snr_lin = np.zeros((n_frames, rf_fft.shape[1], rf_fft.shape[2]))
    snr_log = np.zeros_like(snr_lin)
    for tx in range(rf_fft.shape[1]):
        for rx in range(rf_fft.shape[2]):
            for i in range(n_frames):
                spec = np.abs(rf_fft[i, tx, rx])
                pk = peaks[i, tx, rx, 1]
                snr_lin[i, tx, rx] = snr_linear(spec, pk)
                snr_log[i, tx, rx] = snr_db(spec, pk)

    snr_max_db = np.max(snr_log, axis=(1, 2))
    det_rate = (snr_max_db > 10).sum() / len(snr_max_db) * 100
    print(f"  Detection rate (SNR > 10 dB): {det_rate:.1f}%")

    # -- signal masking & mixing --
    print("Masking and mixing …")
    masked = signal_masked_by_window(rf_fft, peaks,
                                     window_size * fft_zeropad,
                                     window_type="hamming")
    masked_ifft = np.fft.ifft(masked, axis=-1)
    masked_ifft_no_bg = masked_ifft - np.mean(masked_ifft, axis=0)

    tag_matches = [(k, 2 * num_tags - k - 1) for k in range(num_tags)]
    mixed = mix_signals_by_tag_matching(masked_ifft_no_bg, tag_matches)

    # -- distance --
    print("Estimating distance …")
    f_start_adj = f_start + (f_stop - f_start) / ADC_sample * left_discard
    f_stop_adj = f_stop - (f_stop - f_start) / ADC_sample * right_discard
    dist_res_rounded = (dist_res // fft_zeropad) * fft_zeropad

    _, delta_d = distance_calculation(
        mixed, f_start=f_start_adj, f_stop=f_stop_adj,
        no_rel=dist_no_rel, step=20, f_interval=20,
        fft_zeropad=fft_zeropad, distance_resolution=dist_res_rounded,
        bw_ratio=1, average=1)
    chirp_d = distance_calculation(
        mixed, f_start=f_start_adj, f_stop=f_stop_adj,
        no_rel=True, step=120,
        fft_zeropad=1, distance_resolution=dist_res_rounded,
        bw_ratio=1, average=1)
    distance_mean = np.mean(delta_d, axis=(1, 2)) + np.mean(chirp_d)

    # -- angle of arrival --
    print("Estimating AoA …")
    guide_sig = np.take(masked_ifft_no_bg, axis=-2, indices=[1])

    azi_guide_pk, azi_guide_spec = AOA(
        guide_sig, antenna_index=horiz_idx[2:],
        resolution=aoa_res, d=0.5, radar="AOP", guide=True)
    elev_guide_pk, elev_guide_spec = AOA(
        guide_sig, antenna_index=vert_idx[2:],
        resolution=aoa_res, d=0.5, radar="AOP", guide=True)

    _, azi_spec = AOA(mixed, antenna_index=horiz_idx,
                      resolution=aoa_res, d=1, radar="AOP", guide=False)
    _, elev_spec = AOA(mixed, antenna_index=vert_idx,
                       resolution=aoa_res, d=1, radar="AOP", guide=False)

    # guided windowing
    for i in range(n_frames):
        a_lo = int(max(azi_guide_pk.flat[i] - 10, 0) / aoa_res)
        a_hi = int(min(azi_guide_pk.flat[i] + 10, 180) / aoa_res)
        e_lo = int(max(elev_guide_pk.flat[i] - 10, 0) / aoa_res)
        e_hi = int(min(elev_guide_pk.flat[i] + 10, 180) / aoa_res)
        azi_spec[i, :, :a_lo] = 1e-21
        azi_spec[i, :, a_hi:] = 1e-21
        elev_spec[i, :, :e_lo] = 1e-21
        elev_spec[i, :, e_hi:] = 1e-21

    azi_peak = np.argmax(azi_spec, axis=-1) * aoa_res
    elev_peak = np.argmax(elev_spec, axis=-1) * aoa_res

    azi_peak = abrupt_check(azi_peak)
    elev_peak = abrupt_check(elev_peak)

    # -- ground-truth polar --
    dist_gt = np.sqrt(x_gt ** 2 + y_gt ** 2 + z_gt ** 2)
    azi_gt = np.arctan2(x_gt, y_gt) / np.pi * 180
    elev_gt = np.arccos(-z_gt / dist_gt) / np.pi * 180

    # -- timestamp alignment & distance offset --
    t_gt_idx, t_rf_idx = timestamp_find(t_rf, t_gt)
    d_offset = 0.0809

    # -- Kalman filtering --
    print("Kalman filtering …")
    snr_avg = np.mean(snr_lin, axis=(1, 2))
    azi_unc = (np.sum(np.abs(azi_guide_spec)[:, 0, :], axis=1)
               / np.max(np.abs(azi_guide_spec)[:, 0, :], axis=1))
    elev_unc = (np.sum(np.abs(elev_guide_spec)[:, 0, :], axis=1)
                / np.max(np.abs(elev_guide_spec)[:, 0, :], axis=1))

    polar_pts = np.hstack((distance_mean - d_offset,
                           np.deg2rad(azi_peak),
                           np.deg2rad(elev_peak)))

    smoothed = kalman_polar(polar_pts, snr_avg, azi_unc, elev_unc)[:, :3]
    smoothed = gaussian_window(smoothed, 200)

    print("Aligning trajectories …")
    rf_aligned = smoothed[t_rf_idx, :3].copy()
    rf_aligned[:, 0] = (distance_mean - d_offset)[t_rf_idx, 0]
    rf_aligned = np.array(
        [polar_to_cartesian(d, a, e) for d, a, e in rf_aligned])

    gt_aligned = np.column_stack((
        x_gt[t_gt_idx], y_gt[t_gt_idx], z_gt[t_gt_idx]))
    t_rf_al = np.asarray(t_rf[t_rf_idx])
    t_gt_al = np.asarray(t_gt[t_gt_idx])

    # drop NaN frames
    valid = ~np.isnan(gt_aligned).any(axis=1)
    gt_aligned = gt_aligned[valid]
    rf_aligned = rf_aligned[valid]
    t_rf_al = t_rf_al[valid]
    t_gt_al = t_gt_al[valid]

    rot, _, _ = Rotation.align_vectors(gt_aligned, rf_aligned,
                                       return_sensitivity=True)
    rf_rot = rot.apply(rf_aligned)

    # -- errors --
    (mean_err, med_err, tot_err,
     mean_dim, med_dim, norm_err, dim_err) = get_error(gt_aligned, rf_rot)

    print(f"\n{'=' * 55}")
    print(f"  Trajectory : {os.path.basename(traj_dir)}")
    print(f"  Frames     : {len(norm_err)}")
    print(f"  Detection  : {det_rate:.1f}%")
    print(f"  Mean  3-D error : {mean_err * 100:6.2f} cm")
    print(f"  Median 3-D error: {med_err * 100:6.2f} cm")
    print(f"  Mean  per-axis  : "
          f"X={mean_dim[0]*100:.2f}  Y={mean_dim[1]*100:.2f}  "
          f"Z={mean_dim[2]*100:.2f} cm")
    print(f"{'=' * 55}\n")

    # ── save numeric results ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "trajectory_results.npz"),
        estimated=rf_rot, ground_truth=gt_aligned,
        t_rf=t_rf_al, t_gt=t_gt_al,
        norm_errors=norm_err, dim_errors=dim_err,
    )

    # ── figures ───────────────────────────────────────────────────────────
    print("Saving figures …")
    reproj = np.array([cartesian_to_polar(x, y, z) for x, y, z in rf_rot])

    # 1) Cartesian + polar comparison (2×3)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Trajectory: {os.path.basename(traj_dir)}", fontsize=14)

    cart_labels = ["X", "Y", "Z"]
    for c in range(3):
        axs[0, c].plot(t_rf_al, rf_rot[:, c],
                        label="Estimated", alpha=0.8)
        axs[0, c].plot(t_gt_al, gt_aligned[:, c],
                        label="Ground Truth", alpha=0.8)
        axs[0, c].set(xlabel="Time (s)", ylabel="Position (m)",
                       title=cart_labels[c])
        axs[0, c].legend()
        axs[0, c].grid(True, alpha=0.3)

    polar_est = [reproj[:, 0],
                 -np.rad2deg(reproj[:, 1]) + 90,
                 -np.rad2deg(reproj[:, 2]) + 180]
    polar_gt_arr = [dist_gt, azi_gt, elev_gt]
    polar_lab = ["Range (m)", "Azimuth (deg)", "Elevation (deg)"]

    for c in range(3):
        axs[1, c].plot(t_rf_al, polar_est[c],
                        label="Estimated", alpha=0.8)
        axs[1, c].plot(t_gt, polar_gt_arr[c],
                        label="Ground Truth", alpha=0.8)
        axs[1, c].set(xlabel="Time (s)", ylabel=polar_lab[c],
                       title=polar_lab[c].split(" ")[0])
        axs[1, c].legend()
        axs[1, c].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_comparison.png"), dpi=150)
    plt.close()

    # 2) 3-D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*rf_rot.T, "b-", alpha=0.7, label="Estimated")
    ax.plot(*gt_aligned.T, "r-", alpha=0.7, label="Ground Truth")
    ax.scatter(*rf_rot[0], c="b", marker="o", s=60, zorder=5,
               label="Start (est)")
    ax.scatter(*gt_aligned[0], c="r", marker="o", s=60, zorder=5,
               label="Start (GT)")
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax.set_title(f"3-D Trajectory: {os.path.basename(traj_dir)}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_3d.png"), dpi=150)
    plt.close()

    # 3) Error histogram + time-series
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(norm_err * 100, bins=30, edgecolor="k", alpha=0.7)
    ax1.axvline(mean_err * 100, color="r", ls="--",
                label=f"Mean {mean_err*100:.1f} cm")
    ax1.axvline(med_err * 100, color="g", ls="--",
                label=f"Median {med_err*100:.1f} cm")
    ax1.set(xlabel="3-D Error (cm)", ylabel="Count", title="Error Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_rf_al, norm_err * 100, alpha=0.7)
    ax2.set(xlabel="Time (s)", ylabel="3-D Error (cm)", title="Error Over Time")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"), dpi=150)
    plt.close()

    # 4) CDF
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_err = np.sort(norm_err * 100)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf, linewidth=2)
    p50 = np.percentile(norm_err * 100, 50)
    p90 = np.percentile(norm_err * 100, 90)
    ax.axhline(0.5, color="grey", ls=":", alpha=0.5)
    ax.axhline(0.9, color="grey", ls=":", alpha=0.5)
    ax.annotate(f"50th pctl: {p50:.1f} cm", xy=(p50, 0.50),
                fontsize=10, color="tab:blue",
                xytext=(p50 + 1, 0.45), arrowprops=dict(arrowstyle="->"))
    ax.annotate(f"90th pctl: {p90:.1f} cm", xy=(p90, 0.90),
                fontsize=10, color="tab:blue",
                xytext=(p90 + 1, 0.85), arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="3-D Error (cm)", ylabel="CDF",
           title="Cumulative Distribution of Tracking Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_cdf.png"), dpi=150)
    plt.close()

    print(f"All results saved to {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="mmWave Backscatter Tag – Single-Tag Trajectory Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Expected data layout inside <traj_dir>:
              radar/dump/rf_data.npy   – raw radar ADC data
              radar/dump/rf_ts.txt     – radar timestamps
              rosbag/rosbag.csv        – motion-capture ground truth
        """),
    )
    parser.add_argument(
        "-t", "--traj_dir", required=True,
        help="Path to a single trajectory data folder",
    )
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config.json"),
        help="Path to radar/pipeline configuration JSON "
             "(default: config.json next to this script)",
    )
    parser.add_argument(
        "-o", "--output_dir", default=None,
        help="Where to write results (default: <traj_dir>/results)",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Process at most this many radar frames (default: all)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.traj_dir):
        sys.exit(f"Error: directory not found – {args.traj_dir}")

    config = load_config(args.config)
    output_dir = args.output_dir or os.path.join(args.traj_dir, "results")
    run(args.traj_dir, config, output_dir, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
