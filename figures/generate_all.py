#!/usr/bin/env python3
"""Generate all paper figures from precomputed data.

Run from repo root:
    python figures/generate_all.py

Figures that depend only on JSON/NPZ data (no Sionna required):
  Fig 3  — R_ee eigenspectrum
  Fig 4  — Calibration loss convergence
  Fig 7  — OTA NMSE grid + gain over LS
  Fig 8  — OFDM resource grid (fallback mode)
  Fig 9  — BCRB + empirical NMSE
  Fig 10 — BER grid (5 receivers)
  Fig 11 — MCS throughput + table

Figures with Sionna RT fallback (uses precomputed data if RT unavailable):
  Fig 2  — Path loss comparison
  Fig 8  — OFDM resource grid (Sionna mode)
"""
import subprocess
import sys
import os

SCRIPTS = [
    ('Fig 2  — Path loss',                'gen_fig_pathloss.py'),
    ('Fig 3  — R_ee eigenspectrum',       'gen_fig_ree_eigenspectrum.py'),
    ('Fig 4  — Calibration convergence',  'gen_fig_calibration.py'),
    ('Fig 7  — NMSE grid + gain',         'gen_fig_nmse_grid.py'),
    ('Fig 8  — OFDM resource grid',       'gen_fig_resource_grid.py'),
    ('Fig 9  — BCRB + empirical NMSE',    'gen_fig_crb.py'),
    ('Fig 10 — BER grid',                 'gen_fig_ber_grid.py'),
    ('Fig 11 — MCS combined',             'gen_fig_mcs_combined.py'),
]

FIG_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    failed = []
    for label, script in SCRIPTS:
        path = os.path.join(FIG_DIR, script)
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  {script}")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, path], cwd=os.path.dirname(FIG_DIR))
        if result.returncode != 0:
            failed.append(label)
            print(f"  *** FAILED ***")

    print(f"\n{'='*60}")
    if failed:
        print(f"  {len(SCRIPTS) - len(failed)}/{len(SCRIPTS)} figures generated successfully")
        print(f"  Failed: {', '.join(failed)}")
    else:
        print(f"  All {len(SCRIPTS)} figures generated successfully")
    print(f"  Output directory: output/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
