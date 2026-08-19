# Motion Capture with Millimeter-Wave Tags

Code for **M³oCap: Motion Capture with Millimeter-Wave Tags** (SenSys 2026).

Estimates 3-D trajectories of FMCW backscatter tags using a mmWave radar and evaluates them against motion-capture ground truth.

## Repository layout

```
.
├── estimate_trajectory.py          # single-tag pipeline
├── estimate_trajectory_multitag.py # multi-tag pipeline
├── config.json                     # radar & pipeline configuration
├── requirements.txt
└── core/
    ├── signal_processing.py        # peak finding, masking, distance, AoA (MUSIC)
    ├── kalman.py                   # Kalman filter, smoothing, timestamp alignment
    └── metrics.py                  # coordinate conversion & error metrics
```

## Dataset

Download examples of dataset from [here](https://cmu.box.com/s/b0ttkk84kwqx14a2p1bfzghcjs8yn8wc). If you would like more of our dataset please contact Xin: ([xiny@hkust-gz.edu.cn](mailto:xiny@hkust-gz.edu.cn)) and Freddy ([freddyliu@cmu.edu](mailto:freddyliu@cmu.edu)).

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

### Single tag

```bash
python estimate_trajectory.py \
    --traj_dir /path/to/trajectory_folder \
    --config config.json \
    --output_dir /path/to/output
```

### Multiple tags

```bash
python estimate_trajectory_multitag.py \
    --traj_dir /path/to/trajectory_folder \
    --config config_multitag.json \
    --output_dir /path/to/output
```

Results (`.npz` file and four PNG figures) are written to `--output_dir`.
For the multi-tag script, one sub-folder per tag (named by its GT stream, e.g. `yellow_block/`, `green_block/`) is created inside `--output_dir`.

## Expected data layout

Each trajectory folder must contain:

```
<traj_dir>/
├── radar/dump/
│   ├── rf_data.npy          # shape (frames, Tx, Rx, ADC_samples)
│   └── rf_ts.txt            # one timestamp (seconds) per line
└── rosbag/
    ├── rosbag.csv           # columns: tag_name, time, X_in_radar, Y_in_radar, Z_in_radar
    └── moving_times.txt     # (optional, multi-tag only) per-tag moving intervals
```

The multi-tag script additionally requires:

```
<traj_dir>/
├── mod_freqs.txt            # tag-name → modulation frequency (MHz)
└── timestamp_offset.npy     # (optional) radar–mocap clock offset (scalar index)
```

### mod_freqs.txt format

```
All in MHz
yellow_block: 0.5
green_block: 1.4
red_block: 1.6
```

### moving_times.txt format

```
All units in s
yellow_block: 3-8
green_block: 20-26,30-35
red_block: 10-17
```

## Command-line options

Both scripts share the same flags:


| Flag                 | Default              | Description                                |
| -------------------- | -------------------- | ------------------------------------------ |
| `-t`, `--traj_dir`   | *(required)*         | Path to a trajectory data folder           |
| `-c`, `--config`     | `config.json`        | Radar / pipeline configuration file        |
| `-o`, `--output_dir` | `<traj_dir>/results` | Where to write figures and metrics         |
| `--max_frames`       | all                  | Limit the number of radar frames processed |


## Configuration

`config.json` describes the radar hardware and signal-processing parameters.


| Field                                                 | Description                                                              |
| ----------------------------------------------------- | ------------------------------------------------------------------------ |
| `f_start` / `f_stop`                                  | Chirp start and stop frequencies (Hz)                                    |
| `ADC_Samples`                                         | Number of ADC samples per chirp                                          |
| `adc_rate`                                            | ADC sampling rate (Hz)                                                   |
| `num_tags`                                            | Number of backscatter tags                                               |
| `vertical_antenna_index` / `horizontal_antenna_index` | Virtual-array antenna indices for elevation and azimuth MUSIC            |
| `left_discard` / `right_discard`                      | ADC edge samples to discard                                              |
| `fft_zeropad`                                         | Zero-padding factor for the initial FFT                                  |
| `window_size`                                         | Hamming-window width (FFT bins) for signal masking                       |
| `distance_resolution`                                 | Zero-padding factor for the distance-estimation FFT                      |
| `gt_tag_names`                                        | Tag names present in `rosbag.csv`                                        |
| `motion_align_percentile`                             | Velocity percentile threshold for automatic motion detection (multi-tag) |


## Output


| File                        | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| `trajectory_results.npz`    | Estimated and ground-truth trajectories, timestamps, per-frame errors |
| `trajectory_comparison.png` | X/Y/Z and range/azimuth/elevation vs. time                            |
| `trajectory_3d.png`         | 3-D trajectory overlay                                                |
| `error_analysis.png`        | Error histogram and error vs. time                                    |
| `error_cdf.png`             | Cumulative distribution of 3-D tracking error                         |


## Citation

```bibtex
@inproceedings{10.1145/3774906.3800488,
author = {Yang, Xin and Liu, Freddy Yifei and Li, Yunshuang and Gong, Yao and Jayaraman, Dinesh and Abari, Omid and Zhao, Mingmin},
title = {Motion Capture with Millimeter-Wave Tags},
year = {2026},
isbn = {9798400723094},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3774906.3800488},
doi = {10.1145/3774906.3800488},
booktitle = {Proceedings of the 2026 ACM/IEEE International Conference on Embedded Artificial Intelligence and Sensing Systems},
pages = {319–333},
numpages = {15},
series = {SenSys '26}
}`
```

