# DT-Assisted PHY for NextG Manufacturing

Reproducibility package for a paper on digital-twin-assisted OFDM channel estimation, neural receivers, and MCS adaptation, validated on live SDR hardware.

## Repository Structure

```
dt-assisted-phy/
├── config/                      # Shared parameters
│   ├── scene_config.py
│   └── ofdm_params.py
│
├── calibration/                 # DT calibration
│   ├── calibrate_materials.py
│   └── rt_sweep.py
│
├── channel_estimation/          # Classical receivers (no Q-D model)
│   ├── ofdm_postprocess.py
│   ├── eval_ota_compare.py
│   ├── compute_dt_error_stats.py
│   ├── compute_crb.py
│   ├── precompute_rhh.py
│   └── compute_hdt.py
│
├── channel_model/               # Q-D channel model (neural RX & MCS only)
│   └── general_qd_channel.py
│
├── neural_receiver/             # Neural RX (uses Q-D)
│   ├── neural_receiver.py
│   ├── data_gen.py
│   └── train.py
│
├── mcs_adaptation/              # MCS selection (uses Q-D)
│   ├── mcs_integrated.py
│   └── eesm.py
│
├── figures/                     # Figure generation scripts
│   ├── generate_all.py
│   ├── gen_fig_*.py
│   └── static/
│
├── data/                        # Precomputed results
│   ├── calibration/
│   ├── channel/
│   ├── ota_results/
│   └── mcs/
│
├── models/                      # Trained neural RX checkpoints
│
└── output/                      # Generated figures saved here
```

## License

MIT License. See [LICENSE](LICENSE).
