"""
Signal processing primitives for mmWave backscatter tag localization.

Includes peak finding, SNR estimation, frequency-domain signal masking,
tag signal mixing, distance estimation, and MUSIC-based angle of arrival.
"""

import numpy as np
from scipy.signal import find_peaks
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from pyargus.directionEstimation import gen_scanning_vectors, corr_matrix_estimate


# ---------------------------------------------------------------------------
# SNR helpers
# ---------------------------------------------------------------------------

def snr_db(signal, peak):
    """Signal-to-noise ratio in dB at a given peak index."""
    window = max(20 * signal.shape[-1] // 1000, 20)
    s = np.max(signal[peak - window // 10 : peak + window // 10])
    n = np.median(signal[peak - window : peak + window])
    return 20 * np.log10(s / n)


def snr_linear(signal, peak):
    """Signal-to-noise ratio in linear scale at a given peak index."""
    window = max(20 * signal.shape[-1] // 1000, 20)
    s = np.max(signal[peak - window // 10 : peak + window // 10])
    n = np.median(signal[peak - window : peak + window])
    return s / n


# ---------------------------------------------------------------------------
# Peak finding with Kalman tracking
# ---------------------------------------------------------------------------

def _kalman_smooth_1d(signal, estimation_error=0, measurement_error=100):
    """Lightweight 1-D Kalman smoother used internally for peak tracking."""
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([np.mean(signal[:20]), 0.0])
    f.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    f.H = np.array([[1.0, 0.0]])
    error = np.std(signal)
    f.P *= error * estimation_error
    f.R = error * measurement_error
    f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

    smoothed = []
    for z in signal:
        f.predict()
        f.update(z)
        smoothed.append(f.x[0])
    return np.array(smoothed)


def _find_highest_peak(s, reference):
    peaks, _ = find_peaks(s)
    if len(peaks) >= 1:
        return peaks[s[peaks].argmax()]
    return reference


def peak_finding(signal, num_peaks, startbin=200, stopbin=None,
                 reference=None, reference_error_thresh=5,
                 continus=False, **find_peaks_args):
    """
    Detect *num_peaks* spectral peaks per row of *signal*.

    When *reference* is given the search is constrained to a neighbourhood
    around each reference location and the result is Kalman-smoothed across
    the leading axis (typically time).

    Parameters
    ----------
    signal : ndarray, shape (..., num_bins)
    num_peaks : int
    startbin, stopbin : int
        Spectral search range.
    reference : 1-D array or None
        Reference peak locations for guided search.
    reference_error_thresh : int
        Max deviation (in bins) from each reference peak.
    continus : bool
        Track peaks across rows with a running reference window.
    """
    absolute_signal = np.abs(signal)
    if stopbin is None:
        stopbin = absolute_signal.shape[-1]
    if continus:
        assert len(signal.shape) == 2

    def _find(s):
        if len(_find.ref_list) >= 10:
            _find.ref = np.mean(_find.ref_list, axis=0, dtype=int)
            del _find.ref_list[0]

        if _find.ref is None:
            peaks, _ = find_peaks(s[startbin:stopbin], **find_peaks_args)
            if len(peaks) >= num_peaks:
                top = peaks[s[peaks + startbin].argsort()][::-1][:num_peaks]
                peaks = np.sort(top).astype(int)
            else:
                peaks = np.pad(peaks, (num_peaks - len(peaks), 0),
                               constant_values=-startbin)
            if continus:
                _find.ref_list.append(peaks + startbin)
            return peaks + startbin
        else:
            result = []
            for r in _find.ref:
                lo, hi = r - reference_error_thresh, r + reference_error_thresh
                result.append(
                    _find_highest_peak(s[lo:hi], reference_error_thresh)
                    + lo
                )
            if continus:
                _find.ref_list.append(np.array(result))
            return np.array(result)

    _find.ref_list = []
    _find.ref = reference

    peaks_found = np.apply_along_axis(_find, arr=absolute_signal, axis=-1)

    if reference is not None:
        smoother = lambda s: _kalman_smooth_1d(s, 10, 1)
        peaks_found = (np.apply_along_axis(smoother, arr=peaks_found, axis=0)
                       .round().astype(int))
    return peaks_found


# ---------------------------------------------------------------------------
# Signal masking / mixing
# ---------------------------------------------------------------------------

def signal_masked_by_window(signal, middle_points, window_size,
                            window_type="hamming"):
    """
    Apply a frequency-domain window centred on each detected peak.

    Parameters
    ----------
    signal : ndarray, shape (frames, Tx, Rx, [tags], bins)
    middle_points : ndarray, shape (frames, Tx, Rx, num_peaks)
    window_size : int
    window_type : ``"hamming"`` | ``"uniform"``
    """
    num_bins = signal.shape[-1]
    win = np.zeros(num_bins)
    if window_type == "hamming":
        win[:window_size] = (
            0.54 - 0.46 * np.cos(
                2 * np.pi * np.arange(window_size) / (window_size - 1))
        )
    else:
        win[:window_size] = 1.0

    masks = np.apply_along_axis(
        lambda mp: np.roll(win, mp[0] - window_size // 2),
        arr=np.expand_dims(middle_points, axis=-1), axis=-1,
    )
    if len(signal.shape) == len(masks.shape):
        return signal * masks
    return np.expand_dims(signal, axis=-2) * masks


def mix_signals_by_tag_matching(signal, tag_matches):
    """
    Element-wise product of matched left/right tag harmonics.

    Parameters
    ----------
    signal : ndarray, shape (frames, ..., 2*num_tags, bins)
    tag_matches : list of (left_idx, right_idx) tuples
    """
    left = signal.take(indices=[m[0] for m in tag_matches], axis=-2)
    right = signal.take(indices=[m[1] for m in tag_matches], axis=-2)
    return left * right


# ---------------------------------------------------------------------------
# Distance estimation
# ---------------------------------------------------------------------------

def _enhance_by_combination(signal, num_frame_combination=5,
                            num_time_padded=0, mix=False):
    """Improve SNR by coherently combining consecutive frames."""
    n = signal.shape[0]
    padded = np.pad(
        signal,
        pad_width=(*[(0, 0)] * (len(signal.shape) - 1), (0, num_time_padded)),
        mode="constant",
    )
    aligned = padded[:n // num_frame_combination * num_frame_combination]
    reshaped = aligned.reshape(-1, num_frame_combination, *aligned.shape[1:])
    if mix:
        t = reshaped.transpose(0, 2, 3, 4, 1, 5)
        return t.reshape(t.shape[0], t.shape[1], t.shape[2], t.shape[3], -1)
    t = reshaped.transpose(0, 2, 3, 1, 4)
    return t.reshape(t.shape[0], t.shape[1], t.shape[2], -1)


def distance_calculation(signal, f_start=61e9, f_stop=62e9, c=3e8,
                         no_rel=False, calibration=0,
                         distance_resolution=4, fft_zeropad=1,
                         step=20, bw_ratio=1, f_interval=50, average=1):
    """
    Estimate tag distance from the mixed radar signal.

    Two modes controlled by *no_rel*:

    * ``no_rel=True``  – absolute distance from IF-frequency peak only.
    * ``no_rel=False`` – combines IF-frequency peak with phase-based
      relative distance for higher precision.  Returns ``(distance, delta_d)``.

    Parameters
    ----------
    signal : ndarray
        Mixed (product of matched harmonics) time-domain signal.
    f_start, f_stop : float
        Chirp start / stop frequency (Hz).
    c : float
        Speed of light (m/s).
    no_rel : bool
        If True return only absolute (chirp-based) distance.
    calibration : float
        Additive distance calibration (m).
    distance_resolution : int
        FFT zero-padding factor for distance FFT.
    fft_zeropad : int
        Original zero-padding factor used earlier in the pipeline.
    step : int
        Frame sub-sampling stride for the absolute-distance path.
    bw_ratio : float
        Bandwidth ratio adjustment.
    f_interval : int
        Frequency sub-sampling stride for phase-based path.
    average : int
        1 = simple mean, 2 = frame-combination enhancement.
    """
    assert distance_resolution // fft_zeropad >= 1
    mean_frame_num = 16
    range_resolution = c / ((f_stop - f_start) * bw_ratio) / 2

    if no_rel:
        interval = step
        if average == 2:
            comb = _enhance_by_combination(
                signal, num_frame_combination=mean_frame_num,
                mix=True, num_time_padded=11000)
            comb_fft = np.fft.fft(
                comb,
                n=(distance_resolution // mean_frame_num) * comb.shape[-1],
                axis=-1)
            freq = np.argmax(np.abs(comb_fft), axis=-1) / distance_resolution / 2
        else:
            win = np.hamming(signal.shape[-1])
            fft_out = np.fft.fft(
                win * signal[::interval],
                n=(distance_resolution // fft_zeropad) * signal.shape[-1])
            freq = np.argmax(np.abs(fft_out), axis=-1) / distance_resolution
        return 0.5 * freq * range_resolution

    # Combined chirp + phase distance
    sig_fft = np.fft.fft(
        signal[:mean_frame_num],
        n=(distance_resolution // fft_zeropad) * signal.shape[-1])
    sig_fft_abs = np.abs(sig_fft)

    if average == 1:
        freq = (np.argmax(np.mean(sig_fft_abs[:mean_frame_num], axis=0), axis=-1)
                / distance_resolution)
    elif average == 2:
        comb = _enhance_by_combination(
            signal, num_frame_combination=mean_frame_num,
            mix=True, num_time_padded=9000)
        comb_fft = np.fft.fft(
            comb[0],
            n=(distance_resolution // mean_frame_num) * comb.shape[-1])
        freq = np.argmax(np.abs(comb_fft), axis=-1) / distance_resolution / 4
    else:
        freq = np.argmax(sig_fft_abs, axis=-1) / distance_resolution

    chirp_d = 0.5 * freq * range_resolution
    sample_freq = f_start + (f_stop - f_start) / signal.shape[-1] * np.arange(signal.shape[-1])
    sample_wl = c / sample_freq[::f_interval]
    phase = 0.5 * np.unwrap(np.angle(signal[..., ::f_interval]), axis=0)
    rel_d = sample_wl * phase / (4 * np.pi)
    rel_d_mean = np.mean(rel_d, axis=-1)
    rel_d_mean = rel_d_mean - rel_d_mean[0]

    if average:
        res_d = chirp_d + rel_d_mean
    else:
        res_d = np.mean(chirp_d - rel_d_mean, axis=0) + rel_d_mean

    return res_d + calibration, rel_d_mean


# ---------------------------------------------------------------------------
# Angle of Arrival (MUSIC)
# ---------------------------------------------------------------------------

def _music(R, scanning_vectors, signal_dimension=1):
    """
    MUSIC pseudo-spectrum from a spatial correlation matrix.

    Parameters
    ----------
    R : (M, M) complex ndarray – spatial correlation matrix
    scanning_vectors : (M, K) complex ndarray – steering vectors for K angles
    signal_dimension : int – number of signal sources
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(R)
    pairs = sorted(
        [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(M)],
        key=lambda x: x[0],
    )
    noise_dim = M - signal_dimension
    E = np.column_stack([pairs[i][1] for i in range(noise_dim)])
    E_EH = E @ E.conj().T
    return 1.0 / np.sum(
        np.conjugate(scanning_vectors) * (E_EH @ scanning_vectors), axis=0
    )


def AOA(signals, antenna_index, resolution=1,
        start_angle=0, stop_angle=180, d=1,
        radar="AOP", guide=False):
    """
    Estimate angle of arrival via MUSIC.

    Parameters
    ----------
    signals : (frames, Tx, Rx, num_tags, bins)
    antenna_index : list[int]
        Indices into the flattened Tx*Rx virtual array.
    resolution : float
        Angular resolution (degrees).
    start_angle, stop_angle : float
        Peak-search bounds (degrees).
    d : float
        Antenna spacing in half-wavelengths.
    radar : str
        ``"AOP"`` applies phase corrections for TI AWR antenna-on-package.
    guide : bool
        Apply phase sign flip on specific virtual antenna channels.

    Returns
    -------
    angles_peak : (frames, num_tags)
        Peak angle per frame and tag.
    angles : (frames, num_tags, num_angle_bins)
        Full MUSIC pseudo-spectrum.
    """
    sig = signals.copy()
    expanded = sig.reshape(sig.shape[0], sig.shape[1] * sig.shape[2],
                           *sig.shape[3:])

    if radar == "AOP" and guide:
        for idx in [1, 3, 11, 9, 7, 5]:
            if idx < expanded.shape[1]:
                expanded[:, idx] *= -1

    selected = expanded[:, antenna_index]
    num_ant = len(antenna_index)

    incident = np.arange(0, 180, resolution)
    sv = gen_scanning_vectors(num_ant, np.arange(num_ant) * d,
                              np.zeros(num_ant), incident)

    n_frames, n_tags = signals.shape[0], signals.shape[-2]
    angles = np.zeros((n_frames, n_tags, len(incident)), dtype=np.complex64)

    for i in range(n_frames):
        for t in range(n_tags):
            cov = corr_matrix_estimate(selected[i, :, t].T, imp="fast")
            angles[i, t] = _music(cov, sv)

    angles_peak = np.zeros((n_frames, n_tags))
    for t in range(n_tags):
        angles_peak[:, t] = peak_finding(
            np.abs(angles[:, t]), num_peaks=1,
            startbin=int(start_angle / resolution),
            stopbin=int(stop_angle / resolution),
            continus=False, reference_error_thresh=30,
        ).squeeze(axis=-1)

    return angles_peak * resolution, angles


# ---------------------------------------------------------------------------
# Phase-based tag harmonic pairing
# ---------------------------------------------------------------------------

def phase_based_tag_matching(masked_ifft_no_bg, num_tags):
    """
    Pair positive/negative IF harmonics using phase anti-correlation.

    Complementary backscatter harmonics have phases that nearly cancel when
    summed (phase_left + phase_right ≈ 0).  A cost matrix is built as
    ``cost[i, j] = sum |phase_left_i + phase_right_j|`` (after centering each
    on its first sample) and the optimal one-to-one assignment is found with
    the Hungarian algorithm.

    The cost function implicitly exploits the fast-time phase rate of each
    channel: the IFFT of a peak at bin k oscillates at 2π·k/N_fft radians per
    fast-time sample.  For a conjugate pair (k_L + k_R = N_fft) the combined
    phase wraps to full 2π cycles and sums to ≈ 0 after centering; any other
    combination produces a non-zero linear ramp in the summed phase, yielding a
    high cost.  This means the algorithm inherently encodes per-tag modulation
    frequency: the correctly matched pair reveals k_L (and hence f_mod) while
    wrong pairings would point to a spurious combined frequency.

    Parameters
    ----------
    masked_ifft_no_bg : ndarray, shape ``(frames, 2*num_tags, bins)``
        Background-subtracted masked IFFT for a single Tx/Rx channel.
        Left harmonics occupy indices ``0 … num_tags-1``;
        right harmonics occupy indices ``num_tags … 2*num_tags-1``.
    num_tags : int

    Returns
    -------
    tag_matches : list of ``(left_idx, right_idx)`` tuples
        Length *num_tags*.  ``left_idx`` is in ``[0, num_tags)`` and
        ``right_idx`` is in ``[num_tags, 2*num_tags)``.
    cost_matrix : ndarray, shape ``(num_tags, num_tags)``
        Full pairing cost matrix.  The ratio of the assigned cost to the
        next-best cost per row indicates pairing confidence.
    """
    from scipy.optimize import linear_sum_assignment

    def _phase_feature(ch):
        sig = masked_ifft_no_bg[:, ch, :].ravel()
        phase = np.unwrap(np.angle(sig))
        return phase - phase[0]

    left_phases  = np.array([_phase_feature(i)           for i in range(num_tags)])
    right_phases = np.array([_phase_feature(i + num_tags) for i in range(num_tags)])

    combined = left_phases[:, np.newaxis, :] + right_phases[np.newaxis, :, :]
    cost_mat = np.sum(np.abs(combined), axis=2)

    pos_inds, neg_inds = linear_sum_assignment(cost_mat)
    tag_matches = [(int(p), int(n) + num_tags) for p, n in zip(pos_inds, neg_inds)]
    return tag_matches, cost_mat


# ---------------------------------------------------------------------------
# Modulation-frequency extraction from IF peak positions and phase rate
# ---------------------------------------------------------------------------

def modulation_freq_from_peaks(ref_peaks, num_tags, tag_matches,
                                N_fft, adc_rate, f_start, f_stop, c=3e8):
    """
    Estimate each tag's modulation frequency and coarse range from IF peak bins.

    Physics
    -------
    An FMCW backscatter tag at range R modulated at f_mod produces two tones in
    the dechirped IF signal:

        upper sideband  f_beat + f_mod  →  bin k_L  (lower half of FFT)
        lower sideband  f_beat - f_mod  →  wraps to bin k_R = N_fft·(1 − (f_mod−f_beat)/f_ADC)
                                           (upper half of FFT)  when f_mod > f_beat

    where f_beat = 2·R·slope/c  (typically << f_mod for short-range tags).

    Inverting:
        f_mod  = (k_L + N_fft − k_R) · f_ADC / (2·N_fft)
        f_beat = (k_L + k_R − N_fft) · f_ADC / (2·N_fft)
        R_est  = f_beat · c / (2 · chirp_slope)

    where chirp_slope = BW · f_ADC / N_fft  [Hz/s],  BW = f_stop − f_start.

    Parameters
    ----------
    ref_peaks : 1-D array of length 2·num_tags
        Reference peak bin indices.  First *num_tags* are left (lower half);
        last *num_tags* are right (upper half).
    num_tags : int
    tag_matches : list of (left_idx, right_idx) tuples
        Output of :func:`phase_based_tag_matching`.
    N_fft : int
        FFT length actually used (ADC samples after discards × fft_zeropad).
    adc_rate : float
        ADC sampling rate in Hz (hardware-dependent; set in config.json).
    f_start, f_stop : float
        Chirp start / stop frequency in Hz.
    c : float
        Speed of light (m/s).

    Returns
    -------
    list of dict, one per tag:
        ``tag_idx``   – tag index
        ``left_bin``  – k_L (bin index in lower half)
        ``right_bin`` – k_R (bin index in upper half)
        ``f_mod_hz``  – estimated modulation frequency (Hz)
        ``f_beat_hz`` – estimated range-induced beat frequency (Hz)
        ``range_m``   – coarse range estimate from f_beat (m)
    """
    BW = f_stop - f_start
    chirp_slope = BW * adc_rate / N_fft      # Hz / s
    results = []
    for tag_idx, (left_ch, right_ch) in enumerate(tag_matches):
        k_L = int(ref_peaks[left_ch])
        k_R = int(ref_peaks[right_ch])
        f_mod  = (k_L + N_fft - k_R) * adc_rate / (2 * N_fft)
        f_beat = (k_L + k_R - N_fft) * adc_rate / (2 * N_fft)
        range_m = (f_beat * c / (2 * chirp_slope)) if chirp_slope > 0 else 0.0
        results.append(dict(
            tag_idx   = tag_idx,
            left_bin  = k_L,
            right_bin = k_R,
            f_mod_hz  = f_mod,
            f_beat_hz = f_beat,
            range_m   = range_m,
        ))
    return results


def modulation_freq_from_phase_rate(masked_ifft_no_bg, tag_matches, adc_rate):
    """
    Cross-validate per-tag modulation frequency via the fast-time phase rate.

    The masked IFFT of an IF peak at bin k is a complex exponential that
    rotates at 2π·k/N_fft radians per fast-time sample.  After raveling all
    frames into a single 1-D sequence the mean phase step per sample is:

        Δφ/sample  ≈  2π · f_mod / f_ADC

    so  f_mod_phase  =  (Δφ/sample) · f_ADC / (2π)

    This estimate is mathematically equivalent to reading the bin position
    directly, but provides an independent numerical cross-check:
    agreement between ``f_mod_hz`` (bin-based) and ``f_mod_phase_hz``
    (phase-rate-based) confirms that the peak is a clean single-tone signal;
    discrepancy indicates multi-tone contamination or peak-tracking jitter.

    Parameters
    ----------
    masked_ifft_no_bg : ndarray, shape ``(frames, 2·num_tags, N_adc)``
        Background-subtracted masked IFFT (single Tx/Rx channel).
    tag_matches : list of (left_ch, right_ch) tuples
    adc_rate : float
        ADC sampling rate in Hz.

    Returns
    -------
    list of dict, one per tag:
        ``tag_idx``        – tag index
        ``f_mod_phase_hz`` – modulation frequency from fast-time phase rate (Hz)
        ``phase_rate_radps`` – mean phase step per fast-time sample (rad/sample)
    """
    results = []
    for tag_idx, (left_ch, _) in enumerate(tag_matches):
        sig = masked_ifft_no_bg[:, left_ch, :].ravel()
        phase = np.unwrap(np.angle(sig))
        mean_phase_rate = float(np.mean(np.diff(phase)))   # rad / fast-time sample
        f_mod_phase = mean_phase_rate * adc_rate / (2 * np.pi)
        results.append(dict(
            tag_idx           = tag_idx,
            f_mod_phase_hz    = f_mod_phase,
            phase_rate_radps  = mean_phase_rate,
        ))
    return results
