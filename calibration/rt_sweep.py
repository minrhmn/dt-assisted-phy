"""
RT Parameter Sweep — find optimal Sionna RT configuration for Weeks Hall DT.
Tests 10 configurations varying max_depth, scattering, diffraction, samples_per_src.
Two validation datasets: 20-position CFR (primary) + 35-position LoS PL sweep (secondary).
"""
import os, sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import json, warnings, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
FC = 3.5e9
FFT_SIZE = 256
CIR_THRESHOLD_DB = -25
N_BINS = 50
BW_HZ = 50e6

SCENE_PATH = '/home/native/project/weeks_hall_refined/weeks_hall_refined.xml'
MEAS_DIR = '/home/native/project/results_v3'
POS_FILE = '/home/native/project/sionna/data_results/rx_positions_refined.json'
OUT_DIR = '/home/native/project/sionna/figures/rt_sweep'
TABLE_DIR = os.path.join(OUT_DIR, 'tables')
os.makedirs(TABLE_DIR, exist_ok=True)

REUSE_CSV = {
    'D5':   '/home/native/project/sionna/figures/precal_rt0.0/tables/table1_metrics_50mhz.csv',
    'D5_S': '/home/native/project/sionna/figures/precal_rt0.1/tables/table1_metrics_50mhz.csv',
}

ALL_KEYS = [f'posp{i}' for i in range(1, 21)]
REP_POS = ['posp1', 'posp8', 'posp5', 'posp19']

with open(POS_FILE) as f:
    pos_cfg = json.load(f)
TX_POS = pos_cfg['tx']
RX_POS = pos_cfg['rx']

SWEEP_CONFIGS = {
    'D3':    dict(max_depth=3, scatter=False, diffraction=False, samples=1_000_000),
    'D5':    dict(max_depth=5, scatter=False, diffraction=False, samples=1_000_000),
    'D7':    dict(max_depth=7, scatter=False, diffraction=False, samples=1_000_000),
    'D5_S':  dict(max_depth=5, scatter=True,  diffraction=False, samples=1_000_000),
    'D7_S':  dict(max_depth=7, scatter=True,  diffraction=False, samples=1_000_000),
    'D5_D':  dict(max_depth=5, scatter=False, diffraction=True,  samples=1_000_000),
    'D5_SD': dict(max_depth=5, scatter=True,  diffraction=True,  samples=1_000_000),
    'D7_SD': dict(max_depth=7, scatter=True,  diffraction=True,  samples=1_000_000),
    'S1e5':  dict(max_depth=5, scatter=True,  diffraction=False, samples=100_000),
    'S1e7':  dict(max_depth=5, scatter=True,  diffraction=False, samples=10_000_000),
}

CONFIG_COLORS = {
    'D3': '#1f77b4', 'D5': '#ff7f0e', 'D7': '#2ca02c',
    'D5_S': '#d62728', 'D7_S': '#9467bd',
    'D5_D': '#8c564b', 'D5_SD': '#e377c2', 'D7_SD': '#7f7f7f',
    'S1e5': '#bcbd22', 'S1e7': '#17becf',
}

# LoS PL sweep setup
LOS_TX = [7.5, 0, 1.0]
LOS_RX_X = np.arange(6.5, -11.0, -0.5)
LOS_RX_Z = 1.2
PWR_MEAS_SA = np.array([
    -32.7, -42.7, -42.0, -46.0, -41.5,
    -49.6, -50.0, -47.1, -45.5, -49.5,
    -58.8, -60.0, -52.5, -45.7, -45.0,
    -45.5, -47.0, -48.1, -64.0, -62.8,
    -54.9, -50.8, -53.6, -50.8, -53.0,
    -55.1, -62.1, -52.7, -62.9, -53.6,
    -53.9, -60.0, -56.8, -60.0, -64.0
])
D_MEAS_SA = np.linspace(1.0, 18.0, 35)

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def path_gain_dB(H):
    return 10 * np.log10(np.mean(np.abs(H)**2) + 1e-30)

def clean_cir(cir, threshold_dB=-25):
    pdp = np.abs(cir)**2
    pk = np.max(pdp)
    if pk == 0: return cir.copy()
    cir_c = cir.copy()
    cir_c[pdp < pk * 10**(threshold_dB/10)] = 0
    return cir_c

def align_cir(cir_meas, cir_dt):
    pk_dt = int(np.argmax(np.abs(cir_dt)**2))
    pk_m  = int(np.argmax(np.abs(cir_meas)**2))
    shift = pk_dt - pk_m
    return np.roll(cir_meas, shift), shift, pk_dt, pk_m

def peak_match_cir(cir_meas, cir_dt):
    pm = np.max(np.abs(cir_meas))
    pd = np.max(np.abs(cir_dt))
    if pm < 1e-30: return cir_meas.copy()
    return cir_meas * (pd / pm)

def rms_delay_spread(cir, bw):
    pdp = np.abs(cir)**2
    s = np.sum(pdp)
    if s == 0: return 0.0
    p = pdp / s
    tau = np.arange(len(p)) / bw
    mu = np.sum(tau * p)
    return np.sqrt(np.sum((tau - mu)**2 * p)) * 1e9

def normalize_phase(H):
    return H * np.exp(-1j * np.angle(np.sum(H)))

def nmse_dB(H_pred, H_meas):
    Hp = normalize_phase(H_pred)
    Hm = normalize_phase(H_meas)
    theta = np.angle(np.sum(Hp.conj() * Hm))
    Hp_a = Hp * np.exp(1j * theta)
    return 10*np.log10(np.mean(np.abs(Hp_a - Hm)**2) / (np.mean(np.abs(Hm)**2) + 1e-30))

def nmpe_dB(H_pred, H_meas):
    pp = np.abs(H_pred)**2; pm = np.abs(H_meas)**2
    ppn = pp / (np.mean(pp) + 1e-30)
    pmn = pm / (np.mean(pm) + 1e-30)
    return 10*np.log10(np.mean((ppn - pmn)**2) + 1e-30)

def corr_mag(H_pred, H_meas):
    a = np.abs(H_pred)**2; b = np.abs(H_meas)**2
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
    r, _ = pearsonr(a, b)
    return r

def energy_scale(H_dt, H_meas):
    return H_dt * np.sqrt(np.sum(np.abs(H_meas)**2) / (np.sum(np.abs(H_dt)**2) + 1e-30))

def count_taps(cir, threshold_dB=-25):
    pdp = np.abs(cir)**2
    pk = np.max(pdp)
    if pk == 0: return 0
    return int(np.sum(pdp >= pk * 10**(threshold_dB/10)))

def fit_pl(d, pl):
    a, b = np.polyfit(np.log10(d), pl, 1)
    n = a / 10.0
    resid = pl - (a*np.log10(d) + b)
    return n, b, np.std(resid)

# ═══════════════════════════════════════════════════════════════════
# LOAD MEASURED DATA (once)
# ═══════════════════════════════════════════════════════════════════
print("Loading measured data (50 MHz)...")
meas = {}
for k in ALL_KEYS:
    d = np.load(os.path.join(MEAS_DIR, k, 'bw50p0.npz'))
    cfr = d['cfr_avg'].astype(np.complex64)
    meas[k] = {'cfr': cfr, 'cir': np.fft.ifft(cfr).astype(np.complex64)}
print(f"  {len(meas)} positions loaded.\n")

# SA measured PL (anchored to make PL at 1m ≈ Sionna-level)
SA_ANCHOR = 42.0  # approximate Sionna PL at 1m
SA_OFFSET = SA_ANCHOR + PWR_MEAS_SA[0]
PL_SA = SA_OFFSET - PWR_MEAS_SA
N_SA, PL0_SA, SIG_SA = fit_pl(D_MEAS_SA, PL_SA)
print(f"SA reference: n={N_SA:.3f}, PL0={PL0_SA:.2f}, sigma={SIG_SA:.2f} dB\n")

# ═══════════════════════════════════════════════════════════════════
# SIONNA IMPORTS (deferred for speed)
# ═══════════════════════════════════════════════════════════════════
import sionna
from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                       PathSolver, LambertianPattern, DirectivePattern,
                       BackscatteringPattern)
import drjit as dr

sys.path.insert(0, '/home/native/project/3d_model_processing')
from material_config import SCATTERING as SCATTER_CFG

_PAT = {"lambertian": LambertianPattern(), "directive": DirectivePattern(),
        "backscattering": BackscatteringPattern()}
METAL_MACHINE_S = 0.40

def apply_scatter(scene):
    applied = []
    for name, cfg in SCATTER_CFG.items():
        mat = scene.get(name)
        if mat is None and name.startswith("mat-"):
            mat = scene.get(name[4:])
        if mat is None: continue
        s = METAL_MACHINE_S if name == "mat-metal-machine" else cfg["s"]
        mat.scattering_coefficient = s
        mat.xpd_coefficient = cfg["xpd"]
        mat.scattering_pattern = _PAT[cfg["pattern"]]
        applied.append(name)
    return applied

def make_scene(scatter=False):
    sc = load_scene(SCENE_PATH, merge_shapes=False)
    sc.frequency = FC
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0,
                              horizontal_spacing=0, pattern='iso', polarization='H')
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0,
                              horizontal_spacing=0, pattern='iso', polarization='H')
    if scatter:
        n = apply_scatter(sc)
        print(f"  Scattering applied to {len(n)} materials")
    return sc

# ═══════════════════════════════════════════════════════════════════
# PHASE A: 20-POSITION EVALUATION
# ═══════════════════════════════════════════════════════════════════
freqs_fft = np.fft.fftfreq(FFT_SIZE, d=1.0/BW_HZ)
freqs_cont = np.linspace(-BW_HZ/2, BW_HZ/2, FFT_SIZE, endpoint=False)

results_20 = {}   # config_id -> DataFrame
raw_data = {}     # config_id -> {pos: {H_cont, cir_d, cir_m_trunc, cir_d_trunc, ...}}

for cid, cfg in SWEEP_CONFIGS.items():
    t0 = time.time()
    print(f"{'='*60}")
    print(f"[20-pos] {cid}: depth={cfg['max_depth']} scatter={cfg['scatter']} "
          f"diffract={cfg['diffraction']} samples={cfg['samples']:.0e}")
    print(f"{'='*60}")

    # Check for reusable CSV
    if cid in REUSE_CSV:
        print(f"  Loading from {REUSE_CSV[cid]}")
        df = pd.read_csv(REUSE_CSV[cid])
        results_20[cid] = df
        print(f"  Loaded {len(df)} positions in {time.time()-t0:.1f}s")
        continue

    scene = make_scene(scatter=cfg['scatter'])
    scene.add(Transmitter('tx', position=TX_POS))
    scene.add(Receiver('rx', position=[0, 0, 1.2]))
    solver = PathSolver()

    pos_data = {}
    rows = []

    for k in ALL_KEYS:
        scene.receivers['rx'].position = RX_POS[k]
        paths = solver(scene, max_depth=cfg['max_depth'],
                       samples_per_src=cfg['samples'],
                       max_num_paths_per_src=cfg['samples'],
                       los=True, specular_reflection=True,
                       diffuse_reflection=cfg['scatter'],
                       refraction=False,
                       diffraction=cfg['diffraction'],
                       edge_diffraction=cfg['diffraction'],
                       synthetic_array=True, seed=42)

        n_paths = int(np.array(paths.tau).shape[-1])
        H_fft = np.squeeze(paths.cfr(frequencies=freqs_fft, normalize_delays=True,
                                      normalize=False, out_type='numpy')).astype(np.complex64)
        H_cont = np.squeeze(paths.cfr(frequencies=freqs_cont, normalize_delays=True,
                                       normalize=False, out_type='numpy')).astype(np.complex64)

        cfr_m = meas[k]['cfr']
        cir_m_raw = meas[k]['cir']
        cir_d_raw = np.fft.ifft(H_fft).astype(np.complex64)

        # Raw CFR metrics (energy-scaled)
        H_dt_es = energy_scale(H_fft, cfr_m)
        pg_m = path_gain_dB(cfr_m)
        pg_d = path_gain_dB(H_fft)
        _nmpe = nmpe_dB(H_dt_es, cfr_m)
        _nmse = nmse_dB(H_dt_es, cfr_m)
        _corr = corr_mag(H_dt_es, cfr_m)

        # CIR cleaning
        cir_m_c = clean_cir(cir_m_raw, CIR_THRESHOLD_DB)
        cir_d_c = clean_cir(cir_d_raw, CIR_THRESHOLD_DB)
        cir_m_a, shift, _, _ = align_cir(cir_m_c, cir_d_c)
        cir_m_pm = peak_match_cir(cir_m_a, cir_d_c)
        cir_m_t = cir_m_pm[:N_BINS].copy()
        cir_d_t = cir_d_c[:N_BINS].copy()

        ds_m = rms_delay_spread(cir_m_t, BW_HZ)
        ds_d = rms_delay_spread(cir_d_t, BW_HZ)

        # Cleaned CFR metrics
        H_clean_m = np.fft.fft(cir_m_t, FFT_SIZE)
        H_clean_d = np.fft.fft(cir_d_t, FFT_SIZE)
        H_clean_d_es = energy_scale(H_clean_d, H_clean_m)
        nmpe_c = nmpe_dB(H_clean_d_es, H_clean_m)
        nmse_c = nmse_dB(H_clean_d_es, H_clean_m)

        # CIR metrics
        cir_nmse = nmse_dB(cir_d_t, cir_m_t)
        pdp_m = np.abs(cir_m_t)**2; pdp_d = np.abs(cir_d_t)**2
        mask = (pdp_m > 0) | (pdp_d > 0)
        if np.any(mask):
            cir_mae = np.mean(np.abs(10*np.log10(pdp_d[mask]+1e-30) -
                                      10*np.log10(pdp_m[mask]+1e-30)))
        else:
            cir_mae = 0.0

        n_taps_m = count_taps(cir_m_t, CIR_THRESHOLD_DB)
        n_taps_d = count_taps(cir_d_t, CIR_THRESHOLD_DB)

        rows.append(dict(
            Position=k, N_paths=n_paths,
            PG_meas_dB=pg_m, PG_dt_raw_dB=pg_d,
            PG_raw_err_dB=pg_d - pg_m,
            NMPE_dB=_nmpe, NMSE_dB=_nmse, R_corr=_corr,
            DS_meas_ns=ds_m, DS_dt_ns=ds_d, DS_err_ns=ds_d - ds_m,
            NMPE_clean_dB=nmpe_c, NMSE_clean_dB=nmse_c,
            CIR_NMSE_dB=cir_nmse, CIR_MAE_dB=cir_mae,
            N_taps_meas=n_taps_m, N_taps_dt=n_taps_d,
        ))

        pos_data[k] = dict(H_cont=H_cont, cir_m_trunc=cir_m_t, cir_d_trunc=cir_d_t)

    df = pd.DataFrame(rows)
    results_20[cid] = df
    raw_data[cid] = pos_data

    cfg_dir = os.path.join(OUT_DIR, cid)
    os.makedirs(cfg_dir, exist_ok=True)
    df.to_csv(os.path.join(cfg_dir, f'metrics_50mhz.csv'), index=False, float_format='%.3f')

    elapsed = time.time() - t0
    print(f"  {cid} done: NMPE={df.NMPE_dB.mean():.2f} CIR_NMSE={df.CIR_NMSE_dB.mean():.2f} "
          f"R={df.R_corr.mean():.3f} DS_RMSE={np.sqrt(np.mean(df.DS_err_ns**2)):.1f}ns "
          f"({elapsed:.0f}s)\n")

    del scene, solver
    dr.flush_malloc_cache()

# ═══════════════════════════════════════════════════════════════════
# PHASE B: LoS PATH LOSS SWEEP (all configs)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LoS PATH LOSS SWEEP (35 positions, TX at [7.5, 0, 1])")
print("="*60 + "\n")

results_los = {}

for cid, cfg in SWEEP_CONFIGS.items():
    t0 = time.time()
    print(f"[LoS] {cid}...", end=" ", flush=True)

    scene = make_scene(scatter=cfg['scatter'])
    scene.add(Transmitter('tx', position=LOS_TX))
    scene.add(Receiver('rx', position=[0, 0, LOS_RX_Z]))
    solver = PathSolver()

    dists, pls = [], []
    for x in LOS_RX_X:
        scene.receivers['rx'].position = [float(x), 0.0, LOS_RX_Z]
        paths = solver(scene, max_depth=cfg['max_depth'],
                       samples_per_src=cfg['samples'],
                       max_num_paths_per_src=cfg['samples'],
                       los=True, specular_reflection=True,
                       diffuse_reflection=cfg['scatter'],
                       refraction=False,
                       diffraction=cfg['diffraction'],
                       edge_diffraction=cfg['diffraction'],
                       synthetic_array=True, seed=42)
        H = np.squeeze(paths.cfr(frequencies=freqs_fft, normalize_delays=True,
                                  normalize=False, out_type='numpy'))
        pg = np.mean(np.abs(H)**2)
        pls.append(-10*np.log10(pg + 1e-30))
        dists.append(np.sqrt((7.5 - float(x))**2 + (1.0 - LOS_RX_Z)**2))

    dists = np.array(dists); pls = np.array(pls)
    n_dt, pl0_dt, sig_dt = fit_pl(dists, pls)
    results_los[cid] = dict(d=dists, pl=pls, n=n_dt, pl0=pl0_dt, sigma=sig_dt)
    print(f"n={n_dt:.3f} PL0={pl0_dt:.2f} sigma={sig_dt:.2f} ({time.time()-t0:.0f}s)")

    del scene, solver
    dr.flush_malloc_cache()

# ═══════════════════════════════════════════════════════════════════
# AGGREGATE SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BUILDING SUMMARY TABLE")
print("="*60 + "\n")

summary_rows = []
for cid, cfg in SWEEP_CONFIGS.items():
    df = results_20[cid]

    # Relative PG (zero-mean)
    pg_m_rel = df.PG_meas_dB - df.PG_meas_dB.mean()
    pg_d_rel = df.PG_dt_raw_dB - df.PG_dt_raw_dB.mean()
    pg_rmse = np.sqrt(np.mean((pg_d_rel - pg_m_rel)**2))
    pg_r, _ = pearsonr(pg_m_rel, pg_d_rel)

    ds_rmse = np.sqrt(np.mean(df.DS_err_ns**2)) if 'DS_err_ns' in df.columns else np.nan

    # N_taps columns may differ between reused/fresh
    n_taps_dt = df.N_taps_dt.mean() if 'N_taps_dt' in df.columns else np.nan

    los = results_los.get(cid, {})

    summary_rows.append(dict(
        config=cid,
        max_depth=cfg['max_depth'],
        scatter=cfg['scatter'],
        diffraction=cfg['diffraction'],
        samples=cfg['samples'],
        PG_RMSE_dB=pg_rmse,
        PG_R=pg_r,
        DS_RMSE_ns=ds_rmse,
        NMPE_mean=df.NMPE_dB.mean(),
        NMPE_std=df.NMPE_dB.std(),
        CIR_NMSE_mean=df.CIR_NMSE_dB.mean() if 'CIR_NMSE_dB' in df.columns else np.nan,
        CIR_NMSE_std=df.CIR_NMSE_dB.std() if 'CIR_NMSE_dB' in df.columns else np.nan,
        R_corr_mean=df.R_corr.mean(),
        R_corr_std=df.R_corr.std(),
        N_taps_dt=n_taps_dt,
        N_paths_mean=df.N_paths.mean(),
        PL_n=los.get('n', np.nan),
        PL_sigma=los.get('sigma', np.nan),
    ))

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(TABLE_DIR, 'sweep_summary.csv'), index=False, float_format='%.3f')

print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print(f"\nSA reference: n={N_SA:.3f}\n")

# ═══════════════════════════════════════════════════════════════════
# FIGURES — PER CONFIG (PDP + CFR overlays)
# ═══════════════════════════════════════════════════════════════════
print("Generating per-config figures...")

for cid in raw_data:
    cfg_dir = os.path.join(OUT_DIR, cid)
    os.makedirs(cfg_dir, exist_ok=True)
    pdata = raw_data[cid]

    # PDP overlay (4 representative positions)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, pos in zip(axes.flat, REP_POS):
        if pos not in pdata: continue
        pd_m = np.abs(pdata[pos]['cir_m_trunc'])**2
        pd_d = np.abs(pdata[pos]['cir_d_trunc'])**2
        tau = np.arange(N_BINS) / BW_HZ * 1e9
        pd_m_dB = 10*np.log10(pd_m + 1e-30)
        pd_d_dB = 10*np.log10(pd_d + 1e-30)
        ax.plot(tau, pd_m_dB, 'b-', lw=1.5, label='Measured')
        ax.plot(tau, pd_d_dB, 'r--', lw=1.5, label='Sionna')
        ax.set_title(pos, fontsize=11)
        ax.set_xlabel('Delay [ns]')
        ax.set_ylabel('PDP [dB]')
        ax.set_ylim(bottom=np.max(pd_d_dB)-40)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'{cid}: PDP Overlay (50 MHz)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_dir, 'pdp_overlay.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # CFR overlay
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, pos in zip(axes.flat, REP_POS):
        if pos not in pdata: continue
        H_d = pdata[pos]['H_cont']
        cfr_m = meas[pos]['cfr']
        H_m_shifted = np.fft.fftshift(cfr_m)
        H_d_es = energy_scale(H_d, H_m_shifted)
        f_mhz = freqs_cont / 1e6
        ax.plot(f_mhz, 20*np.log10(np.abs(H_m_shifted)+1e-30), 'b-', lw=1, label='Measured')
        ax.plot(f_mhz, 20*np.log10(np.abs(H_d_es)+1e-30), 'r--', lw=1, label='Sionna')
        ax.set_title(pos, fontsize=11)
        ax.set_xlabel('Freq offset [MHz]')
        ax.set_ylabel('|H| [dB]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'{cid}: CFR Magnitude Overlay (50 MHz)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_dir, 'cfr_overlay.png'), dpi=200, bbox_inches='tight')
    plt.close()

print(f"  Per-config figures saved for {len(raw_data)} configs.")

# ═══════════════════════════════════════════════════════════════════
# FIGURES — CROSS-CONFIG COMPARISONS
# ═══════════════════════════════════════════════════════════════════
print("Generating cross-config comparison figures...")
cids_ordered = list(SWEEP_CONFIGS.keys())
colors = [CONFIG_COLORS[c] for c in cids_ordered]
x_pos = np.arange(len(cids_ordered))

def bar_fig(metric_col, ylabel, title, fname, higher_better=False, add_ref=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    vals = [summary.loc[summary.config==c, metric_col].values[0] for c in cids_ordered]
    bars = ax.bar(x_pos, vals, color=colors, edgecolor='black', linewidth=0.5)
    if add_ref is not None:
        ax.axhline(add_ref[0], color='green', ls='--', lw=1.5, label=add_ref[1])
        ax.legend()
    best_idx = int(np.argmax(vals) if higher_better else np.argmin(vals))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cids_ordered, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()

# 1. Relative PG RMSE
bar_fig('PG_RMSE_dB', 'Relative PG RMSE [dB]',
        'Relative Path Gain Error (lower = better)', 'fig_pg_error_bar.png')

# 2. DS RMSE
bar_fig('DS_RMSE_ns', 'DS RMSE [ns]',
        'Delay Spread RMSE (lower = better)', 'fig_ds_rmse_bar.png')

# 3. CIR NMSE
bar_fig('CIR_NMSE_mean', 'CIR NMSE [dB]',
        'CIR NMSE (lower = better)', 'fig_cir_nmse_bar.png')

# 4. NMPE
bar_fig('NMPE_mean', 'NMPE [dB]',
        'CFR Normalized Mean Power Error (lower = better)', 'fig_nmpe_bar.png')

# 5. Correlation
bar_fig('R_corr_mean', 'Pearson R',
        'CFR Magnitude Correlation (higher = better)', 'fig_corr_bar.png',
        higher_better=True)

# 6. PL exponent
bar_fig('PL_n', 'Path Loss Exponent n',
        f'PL Exponent from LoS Sweep (SA ref: n={N_SA:.2f})', 'fig_pl_exponent.png',
        add_ref=(N_SA, f'SA measured (n={N_SA:.2f})'))

# 7. Depth trend (scatter on vs off)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
depths_no_s = [3, 5, 7]
cids_no_s = ['D3', 'D5', 'D7']
depths_s = [5, 7]
cids_s = ['D5_S', 'D7_S']

metrics_trend = [
    ('NMPE_mean', 'NMPE [dB]', False),
    ('CIR_NMSE_mean', 'CIR NMSE [dB]', False),
    ('R_corr_mean', 'Correlation R', True),
    ('DS_RMSE_ns', 'DS RMSE [ns]', False),
]
for ax, (col, ylabel, hb) in zip(axes.flat, metrics_trend):
    v_no = [summary.loc[summary.config==c, col].values[0] for c in cids_no_s]
    v_s = [summary.loc[summary.config==c, col].values[0] for c in cids_s]
    ax.plot(depths_no_s, v_no, 'bo-', lw=2, ms=8, label='Specular only')
    ax.plot(depths_s, v_s, 'rs-', lw=2, ms=8, label='With scattering')
    ax.set_xlabel('max_depth')
    ax.set_ylabel(ylabel)
    ax.set_xticks([3, 5, 7])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
plt.suptitle('Effect of max_depth (Scatter ON vs OFF)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_depth_trend.png'), dpi=200, bbox_inches='tight')
plt.close()

# 8. Convergence (samples_per_src)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
conv_cids = ['S1e5', 'D5_S', 'S1e7']
conv_samples = [1e5, 1e6, 1e7]
for ax, (col, ylabel, hb) in zip(axes.flat, metrics_trend):
    vals = [summary.loc[summary.config==c, col].values[0] for c in conv_cids]
    ax.semilogx(conv_samples, vals, 'go-', lw=2, ms=8)
    ax.set_xlabel('samples_per_src')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
plt.suptitle('Convergence vs SBR Ray Count (depth=5, scatter=on)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_convergence.png'), dpi=200, bbox_inches='tight')
plt.close()

# 9. Number of paths/taps bar
fig, ax = plt.subplots(figsize=(12, 5))
n_paths_vals = [summary.loc[summary.config==c, 'N_paths_mean'].values[0] for c in cids_ordered]
ax.bar(x_pos, n_paths_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(cids_ordered, rotation=45, ha='right')
ax.set_ylabel('Mean N paths')
ax.set_title('Average Number of Paths per Position')
ax.set_yscale('log')
ax.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(n_paths_vals):
    ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_n_paths_bar.png'), dpi=200, bbox_inches='tight')
plt.close()

# 10. Summary heatmap
fig, ax = plt.subplots(figsize=(14, 6))
hm_cols = ['PG_RMSE_dB', 'DS_RMSE_ns', 'NMPE_mean', 'CIR_NMSE_mean', 'R_corr_mean', 'PL_n']
hm_labels = ['PG RMSE\n(dB)', 'DS RMSE\n(ns)', 'NMPE\n(dB)', 'CIR NMSE\n(dB)',
             'Corr R', 'PL exp n']
hm_data = np.array([[summary.loc[summary.config==c, col].values[0] for col in hm_cols]
                     for c in cids_ordered])
# Normalize columns to [0,1] for coloring (lower=better except corr and PL_n)
hm_norm = np.zeros_like(hm_data)
for j in range(hm_data.shape[1]):
    col_data = hm_data[:, j]
    mn, mx = np.nanmin(col_data), np.nanmax(col_data)
    if mx - mn > 1e-10:
        hm_norm[:, j] = (col_data - mn) / (mx - mn)
    # Invert for "higher is better" metrics
    if hm_cols[j] in ('R_corr_mean',):
        hm_norm[:, j] = 1 - hm_norm[:, j]
    # PL_n: closer to SA reference is better
    if hm_cols[j] == 'PL_n':
        hm_norm[:, j] = np.abs(col_data - N_SA) / (np.nanmax(np.abs(col_data - N_SA)) + 1e-10)

im = ax.imshow(hm_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
for i in range(len(cids_ordered)):
    for j in range(len(hm_cols)):
        ax.text(j, i, f'{hm_data[i,j]:.2f}', ha='center', va='center', fontsize=9)
ax.set_xticks(range(len(hm_labels)))
ax.set_xticklabels(hm_labels)
ax.set_yticks(range(len(cids_ordered)))
ax.set_yticklabels(cids_ordered)
ax.set_title('RT Sweep Summary (green = better)')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_summary_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()

# 11. LoS PL curves comparison
fig, ax = plt.subplots(figsize=(12, 7))
d_model = np.linspace(1, 18, 100)
lam = 3e8 / FC
fspl = 20*np.log10(4*np.pi*d_model/lam)
inf_sl = 31.84 + 21.50*np.log10(d_model) + 19*np.log10(3.5)
ax.plot(d_model, fspl, 'k--', lw=1, alpha=0.4, label='Free space (n=2)')
ax.plot(d_model, inf_sl, 'k:', lw=1, alpha=0.4, label='3GPP InF-SL (n=2.15)')
ax.scatter(D_MEAS_SA, PL_SA, c='green', marker='o', s=30, zorder=5, label=f'SA meas (n={N_SA:.2f})')
a_sa, b_sa = np.polyfit(np.log10(D_MEAS_SA), PL_SA, 1)
ax.plot(d_model, a_sa*np.log10(d_model)+b_sa, 'g-', lw=2)
for cid in ['D3', 'D5', 'D7', 'D5_S', 'D7_S', 'D7_SD']:
    los = results_los[cid]
    a_c, b_c = np.polyfit(np.log10(los['d']), los['pl'], 1)
    ax.scatter(los['d'], los['pl'], c=CONFIG_COLORS[cid], marker='x', s=15, alpha=0.5)
    ax.plot(d_model, a_c*np.log10(d_model)+b_c, color=CONFIG_COLORS[cid], lw=1.5,
            label=f'{cid} (n={los["n"]:.2f})')
ax.set_xlabel('Distance [m]')
ax.set_ylabel('Path Loss [dB]')
ax.set_title('LoS Path Loss Sweep Comparison @ 3.5 GHz')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_los_pl_curves.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"\nAll figures saved to {OUT_DIR}/")
print(f"Summary table: {TABLE_DIR}/sweep_summary.csv")

# ═══════════════════════════════════════════════════════════════════
# FINAL RANKING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CONFIG RANKING (composite score)")
print("="*60)

# Composite: lower is better for all except R_corr (invert) and PL_n (distance from SA)
def rank_score(row):
    s = 0
    s += row['PG_RMSE_dB'] / 5.0         # normalize ~5 dB scale
    s += row['DS_RMSE_ns'] / 35.0         # normalize ~35 ns scale
    s += row['NMPE_mean'] / 3.0           # normalize ~3 dB scale
    s += row['CIR_NMSE_mean'] / 3.0       # normalize
    s -= row['R_corr_mean']               # higher is better
    s += abs(row['PL_n'] - N_SA) / 0.5    # PL exponent distance from SA
    return s

summary['score'] = summary.apply(rank_score, axis=1)
ranking = summary.sort_values('score')[['config', 'score', 'PG_RMSE_dB', 'DS_RMSE_ns',
                                         'NMPE_mean', 'CIR_NMSE_mean', 'R_corr_mean', 'PL_n']]
print(ranking.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print(f"\nBest config: {ranking.iloc[0]['config']}")
print("\nDone.")
