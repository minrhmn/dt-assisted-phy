#!/usr/bin/env python3
"""Integrated MCS adaptation: 3GPP InF-SL vs DT-Derived vs DT-Assisted.

Connected to the estimation pipeline — all strategies are estimation-aware:
  - 3GPP InF-SL + LS estimation:
      Statistical channel → per-SC SINR_eff with LS MSE penalty → EESM → MCS
      (no site knowledge; LS penalty = 1/(1 + 1/n_pilots) = -1.76 dB constant)
  - DT-Derived LMMSE (R_hh from 3004-position dense grid):
      Q-D fading ensemble → per-SC SINR_eff with R_hh-based MSE → EESM → MCS
      (global covariance prior, no per-position R_ee; position-independent)
  - DT-Assisted LMMSE (R_ee per-position error covariance):
      Q-D fading ensemble → per-SC SINR_eff with R_ee-based MSE → EESM → MCS
      (site-specific model; uses per-position H_dt and empirical R_ee)

11-entry MCS table from 3GPP TS 38.214 Table 5.1.3.1-1.
Baseline uses LMMSE with empirical R_hh (20 sounding positions).

Validation: coded 5G NR LDPC OFDM BLER on 200 measured CFR frames/position.

Run:
    python mcs_adaptation/mcs_integrated.py

Output:
    data_results/mcs_integrated_results.json
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from scipy.stats import beta as beta_dist

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource

from neural_receiver.config import (
    cir_to_cfr, RX_POSITIONS, FFT, N_OCC, OCC_BINS, NOMINAL_TX_POS,
)
from channel_model.general_qd_channel import GeneralQDChannel

# ── 11-entry MCS table (3GPP TS 38.214, Table 5.1.3.1-1) ───────────────
MCS_TABLE = {
    1:  dict(mod_order=4,   code_rate=240/1024, mod_name='QPSK'),
    3:  dict(mod_order=4,   code_rate=256/1024, mod_name='QPSK'),
    4:  dict(mod_order=4,   code_rate=308/1024, mod_name='QPSK'),
    6:  dict(mod_order=4,   code_rate=449/1024, mod_name='QPSK'),
    9:  dict(mod_order=4,   code_rate=679/1024, mod_name='QPSK'),
    10: dict(mod_order=16,  code_rate=340/1024, mod_name='16QAM'),
    13: dict(mod_order=16,  code_rate=490/1024, mod_name='16QAM'),
    16: dict(mod_order=16,  code_rate=658/1024, mod_name='16QAM'),
    19: dict(mod_order=64,  code_rate=517/1024, mod_name='64QAM'),
    22: dict(mod_order=64,  code_rate=666/1024, mod_name='64QAM'),
    25: dict(mod_order=64,  code_rate=822/1024, mod_name='64QAM'),
}

EESM_BETA = {
    1:  1.60,
    3:  1.80,
    4:  2.50,
    6:  3.36,
    9:  5.00,
    10: 5.26,
    13: 8.40,
    16: 12.40,
    19: 18.77,
    22: 23.30,
    25: 30.00,
}

MCS_INDICES = sorted(MCS_TABLE.keys())

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data_results')
NR_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'neural_receiver', 'data')
RESULTS_V4 = '/home/native/project/results_v4'
RESULTS_V5 = '/home/native/project/results_v5'

# ── System parameters ────────────────────────────────────────────────────
BW_HZ = 50e6
CARRIER_FREQ = 3.5e9
N_PILOTS = 2
BLER_TARGET = 0.10

# ── Experiment design ─────────────────────────────────────────────────────
# 5 positions × 1 SNR each. SNRs at MCS transition boundaries.
# 4 positions show DT gain, 1 position ties — cross-position comparison.
EXPERIMENT_POSITIONS = {
    'posp1':   2.0,   # LOS,  MCS4→MCS6 transition, ρ=0.58 (DT wins)
    'posp3':   5.0,   # LOS,  MCS6→MCS10 transition, ρ=0.61 (DT wins)
    'posp10':  8.0,   # NLOS, MCS10→MCS13 transition, ρ=0.75 (DT wins)
    'posp11': 11.0,   # LOS,  MCS13→MCS16 transition, ρ=0.66 (DT wins)
    'posp19':  7.0,   # NLOS, Mid-plateau MCS10, ρ=0.70 (TIE)
}


# ── EESM core ───────────────────────────────────────────────────────────

def compute_eesm(snr_linear_per_sc, beta):
    ratio = np.clip(snr_linear_per_sc / beta, 0, 50)
    return -beta * np.log(np.mean(np.exp(-ratio)))


def spectral_efficiency(mcs_idx):
    if mcs_idx == 0:
        return 0.0
    entry = MCS_TABLE[mcs_idx]
    return int(np.log2(entry['mod_order'])) * entry['code_rate']


def effective_throughput(mcs_idx, bler):
    return spectral_efficiency(mcs_idx) * (1.0 - bler)


def mcs_label(mcs_idx):
    if mcs_idx == 0:
        return "No TX"
    entry = MCS_TABLE[mcs_idx]
    return f"MCS{mcs_idx} ({entry['mod_name']} R={entry['code_rate']:.3f})"


# ── SNR thresholds (load and subset to 6) ────────────────────────────────

def load_snr_thresholds():
    path = os.path.join(DATA_DIR, 'mcs_snr_thresholds.json')
    with open(path) as f:
        raw = json.load(f)
    full = {int(k): v for k, v in raw.items()}
    return {idx: full[idx] for idx in MCS_INDICES}


# ── Data loading ─────────────────────────────────────────────────────────

_cir_cache = {}

def load_hdt(pos_key, bw_hz=BW_HZ):
    cir_path = os.path.join(NR_DATA_DIR, 'cir_measured_d7r_cal.npz')
    if cir_path not in _cir_cache:
        _cir_cache[cir_path] = np.load(cir_path)
    d = _cir_cache[cir_path]
    h_full = cir_to_cfr(d[f'{pos_key}_a_re'], d[f'{pos_key}_a_im'],
                         d[f'{pos_key}_tau'], bw_hz)
    h_occ = h_full[OCC_BINS].astype(np.complex128)
    h_occ /= np.sqrt(np.mean(np.abs(h_occ)**2) + 1e-30)
    return h_occ


def load_measured_cfr_frames(pos_key, bw_label='bw50p0'):
    d = np.load(os.path.join(RESULTS_V5, pos_key, f'{bw_label}.npz'))
    cfr = d['cfr_per_frame'][:, OCC_BINS].astype(np.complex128)
    cfr /= np.sqrt(np.mean(np.abs(cfr)**2, axis=1, keepdims=True) + 1e-30)
    return cfr


def load_qd_model():
    path = os.path.join(NR_DATA_DIR, 'general_qd_env_model_d7r_cal.npz')
    return GeneralQDChannel.from_env_model(path)


def load_ree():
    d = np.load(os.path.join(NR_DATA_DIR, 'dt_error_stats_50m.npz'))
    R = d['R_EE_freq'].astype(np.complex128)
    corr_per_pos = d['corr_per_pos']
    nmse_per_pos = d['nmse_per_pos']
    pos_labels = list(d['pos_labels'])
    corr_dict = dict(zip(pos_labels, corr_per_pos))
    nmse_dict = dict(zip(pos_labels, nmse_per_pos))
    return R, corr_dict, nmse_dict


def load_rhh():
    """Load R_hh eigendecomposition from 3004-position dense CIR grid + Q-D."""
    path = os.path.join(NR_DATA_DIR, 'rhh_qd_global_50m.npz')
    d = np.load(path)
    eigvals = d['R_hh_eigvals'].astype(np.float64)
    eigvecs = d['R_hh_eigvecs'].astype(np.complex128)
    trace_per_sc = float(d['trace_per_sc'])
    n_positions = int(d['n_positions'])
    return eigvals, eigvecs, trace_per_sc, n_positions


CIR_MAX_TAP = 50


def load_rhh_empirical(bw_label='bw50p0'):
    """Empirical R_hh from 20 sounding positions (same as LMMSE receiver in eval_ota_compare)."""
    H_all = []
    for i in range(1, 21):
        fpath = os.path.join(RESULTS_V4, f'posp{i}', f'{bw_label}.npz')
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath)
        cfr = d['cfr_avg'].astype(np.complex128)
        cir = np.fft.ifft(cfr)
        H = np.fft.fft(np.pad(cir[:CIR_MAX_TAP], (0, FFT - CIR_MAX_TAP)))[OCC_BINS]
        H_all.append(H)
    H_all = np.array(H_all)
    R_sample = (H_all.conj().T @ H_all) / len(H_all)
    mu = np.real(np.trace(R_sample)) / N_OCC
    R_hh = (R_sample + mu * np.eye(N_OCC)).astype(np.complex128)
    eigvals, eigvecs = np.linalg.eigh(R_hh)
    eigvals = np.maximum(eigvals[::-1].real, 0.0)
    eigvecs = eigvecs[:, ::-1]
    trace_per_sc = np.real(np.trace(R_hh)) / N_OCC
    return eigvals, eigvecs, trace_per_sc, len(H_all)


def compute_mse_lmmse(eigvals, eigvecs, trace_per_sc, noise_var, n_pilots=N_PILOTS):
    """Per-SC MSE from any LMMSE estimator given covariance eigendecomposition.

    Works for both empirical R_hh and DT-Derived R_hh — same formula,
    different input covariance.
    """
    est_noise = noise_var / n_pilots
    lam = eigvals / trace_per_sc
    post_eig = lam * est_noise / (lam + est_noise + 1e-30)
    mse_per_sc = np.real(np.diag(
        (eigvecs * post_eig[None, :]) @ eigvecs.conj().T
    ))
    return np.maximum(mse_per_sc, 0)


def scale_ree_for_position(R_ee_env, nmse_dict, pos_key):
    """Scale env-level R_ee by per-position NMSE ratio.

    Preserves correlation structure, adjusts magnitude to match the DT
    prediction quality at this specific position.
    """
    nmse_vals = np.array(list(nmse_dict.values()))
    mean_nmse_lin = np.mean(10 ** (nmse_vals / 10))
    pos_nmse_lin = 10 ** (nmse_dict[pos_key] / 10)
    scale = pos_nmse_lin / mean_nmse_lin
    return R_ee_env * scale, scale


# ── Estimation MSE (connects MCS to estimation pipeline) ────────────────

def compute_mse_ls(noise_var, n_pilots=N_PILOTS):
    return noise_var / n_pilots


def compute_mse_dt_assisted_lmmse(R_ee, noise_var, n_pilots=N_PILOTS):
    """MSE of DT-Assisted LMMSE: H_hat = H_dt + R_ee(R_ee+σ²I)⁻¹(H_ls-H_dt).

    Uses per-position R_ee (empirical error covariance, 20 sounding positions).
    """
    n_sc = R_ee.shape[0]
    est_noise = noise_var / n_pilots
    W = R_ee @ np.linalg.inv(R_ee + est_noise * np.eye(n_sc))
    mse = np.real(np.diag(R_ee - W @ R_ee))
    return np.maximum(mse, 0)


# ── MCS selection helpers ────────────────────────────────────────────────

def select_mcs_from_snr(snr_per_sc, snr_thresholds):
    best_mcs = 0
    best_gamma = -np.inf
    for mcs_idx in MCS_INDICES:
        beta = EESM_BETA[mcs_idx]
        gamma_eff = compute_eesm(snr_per_sc, beta)
        gamma_db = 10 * np.log10(max(gamma_eff, 1e-10))
        if gamma_db >= snr_thresholds[mcs_idx]:
            best_mcs = mcs_idx
            best_gamma = gamma_db
    return best_mcs, best_gamma


# ── Strategy 1: 3GPP InF-SL (statistical model, no site knowledge) ──────

def _generate_multicluster_channel(tau_rms_s, k_lin, n_clusters=6, rng=None):
    los_power = k_lin / (k_lin + 1)
    nlos_power = 1.0 / (k_lin + 1)
    h = np.sqrt(los_power) * np.ones(N_OCC, dtype=np.complex128)
    cluster_delays = np.sort(rng.exponential(tau_rms_s, size=n_clusters))
    cluster_powers = np.exp(-cluster_delays / max(tau_rms_s, 1e-12))
    cluster_powers *= nlos_power / (np.sum(cluster_powers) + 1e-30)
    sc_spacing = BW_HZ / FFT
    for c in range(n_clusters):
        phase = -2j * np.pi * cluster_delays[c] * sc_spacing * OCC_BINS
        gain = np.sqrt(cluster_powers[c] / 2) * (
            rng.standard_normal() + 1j * rng.standard_normal())
        h += gain * np.exp(phase)
    h /= np.sqrt(np.mean(np.abs(h)**2) + 1e-30)
    return h


def strategy_3gpp_inf(pos_xyz, tx_pos, snr_thresholds, operating_snr_db,
                       n_realizations=50, n_pilots=N_PILOTS, rng=None,
                       estimation='ls',
                       rhh_eigvals=None, rhh_eigvecs=None, rhh_trace_per_sc=None):
    """3GPP TR 38.901 InF-SL + estimation penalty → median MCS vote.

    estimation:
      'ls'      — constant -1.76 dB penalty (MSE = σ²/n_pilots)
      'lmmse'   — per-SC MSE from empirical R_hh (requires rhh_eigvals/vecs)
      'perfect' — no estimation penalty (MSE = 0)
    """
    if rng is None:
        rng = np.random.default_rng()

    d_2d = np.linalg.norm(np.array(pos_xyz[:2]) - np.array(tx_pos[:2]))
    d_3d = np.sqrt(d_2d**2 + (tx_pos[2] - pos_xyz[2])**2)
    d_clutter = 10.0
    p_los = np.exp(-d_3d / d_clutter) if d_3d > d_clutter else 1.0

    if estimation == 'lmmse':
        lam_norm = rhh_eigvals / rhh_trace_per_sc

    mcs_votes = []
    for _ in range(n_realizations):
        is_los = rng.random() < p_los
        sf_std = 4.0 if is_los else 7.2
        sf_db = rng.normal(0, sf_std)

        if is_los:
            k_db = max(rng.normal(7.0, 4.0), -5.0)
        else:
            k_db = -100.0
        k_lin = 10 ** (k_db / 10)

        lg_ds = rng.normal(-7.18, 0.12) if is_los else rng.normal(-7.60, 0.18)
        tau_rms_s = 10 ** lg_ds

        h = _generate_multicluster_channel(tau_rms_s, k_lin, rng=rng)
        snr_lin = 10 ** (operating_snr_db / 10) * 10 ** (sf_db / 10)
        noise_var = 1.0 / snr_lin

        if estimation == 'perfect':
            sinr_eff_per_sc = np.abs(h)**2 * snr_lin
        elif estimation == 'lmmse':
            est_noise = noise_var / n_pilots
            post_eig = lam_norm * est_noise / (lam_norm + est_noise + 1e-30)
            mse_per_sc = np.real(np.diag(
                (rhh_eigvecs * post_eig[None, :]) @ rhh_eigvecs.conj().T
            ))
            mse_per_sc = np.maximum(mse_per_sc, 0)
            sinr_eff_per_sc = np.abs(h)**2 / (noise_var + mse_per_sc)
        else:
            sinr_eff_per_sc = np.abs(h)**2 * snr_lin / (1.0 + 1.0 / n_pilots)

        mcs, _ = select_mcs_from_snr(sinr_eff_per_sc, snr_thresholds)
        mcs_votes.append(mcs)

    med = np.median(mcs_votes)
    selected = int(MCS_INDICES[np.argmin(np.abs(np.array(MCS_INDICES) - med))])
    if selected not in MCS_TABLE and selected != 0:
        selected = 0
    return selected


# ── Strategy 2: DT+Q-D with estimation-aware SINR ──────────────────────

def strategy_dt_direct(h_dt, snr_thresholds, operating_snr_db,
                       mse_per_sc):
    """DT channel + estimation-aware SINR → MCS (no Q-D fading)."""
    noise_var = 1.0 / (10 ** (operating_snr_db / 10))
    sinr_eff = np.abs(h_dt)**2 / (noise_var + mse_per_sc)
    best_mcs = 0
    best_gamma = -np.inf
    for mcs_idx in MCS_INDICES:
        beta = EESM_BETA[mcs_idx]
        gamma = compute_eesm(sinr_eff, beta)
        gamma_db = 10 * np.log10(max(gamma, 1e-10))
        if gamma_db >= snr_thresholds[mcs_idx]:
            best_mcs = mcs_idx
            best_gamma = gamma_db
    return best_mcs, best_gamma


def strategy_qd_estimation_aware(h_dt, qd_model, snr_thresholds,
                                  operating_snr_db, mse_per_sc,
                                  n_drops=50, percentile=25, rng=None):
    """Q-D ensemble with estimation-aware SINR → robust MCS.

    SINR_eff[k] = |H_qd[k]|^2 / (noise_var + MSE[k])
    Uses 25th percentile EESM across drops (appropriate for OLLA at 10% BLER).
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_var = 1.0 / (10 ** (operating_snr_db / 10))
    drops = qd_model.generate(h_dt, n_frames=1, n_drops=n_drops, rng=rng)

    gamma_eff_per_mcs = {idx: [] for idx in MCS_INDICES}

    for drop_h in drops:
        h_qd = drop_h[0]
        h_qd_norm = h_qd / np.sqrt(np.mean(np.abs(h_qd)**2) + 1e-30)
        sinr_eff = np.abs(h_qd_norm)**2 / (noise_var + mse_per_sc)

        for mcs_idx in MCS_INDICES:
            beta = EESM_BETA[mcs_idx]
            gamma = compute_eesm(sinr_eff, beta)
            gamma_db = 10 * np.log10(max(gamma, 1e-10))
            gamma_eff_per_mcs[mcs_idx].append(gamma_db)

    best_mcs = 0
    for mcs_idx in MCS_INDICES:
        pctl = np.percentile(gamma_eff_per_mcs[mcs_idx], percentile)
        if pctl >= snr_thresholds[mcs_idx]:
            best_mcs = mcs_idx

    return best_mcs, gamma_eff_per_mcs


# ── Coded OFDM BLER validation ──────────────────────────────────────────

class CodedOFDMLink(tf.keras.Model):
    def __init__(self, mcs_idx):
        super().__init__()
        entry = MCS_TABLE[mcs_idx]
        mod_order = entry['mod_order']
        code_rate = entry['code_rate']
        n_info_bits = 1024
        num_bps = int(np.log2(mod_order))
        n_coded_bits = int(np.round(n_info_bits / code_rate))
        n_coded_bits = (n_coded_bits // num_bps) * num_bps
        self.n_info_bits = n_info_bits
        self.n_coded_bits = n_coded_bits
        self.num_bps = num_bps
        self.source = BinarySource()
        self.encoder = LDPC5GEncoder(k=n_info_bits, n=n_coded_bits)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=20)
        self.mapper = Mapper("qam", num_bps)
        self.demapper = Demapper("app", "qam", num_bps)

    @tf.function(jit_compile=False)
    def call(self, h_batch, noise_var):
        batch = tf.shape(h_batch)[0]
        bits = self.source([batch, self.n_info_bits])
        coded = self.encoder(bits)
        symbols = self.mapper(coded)
        y = tf.cast(symbols, tf.complex64) * h_batch
        stddev = tf.cast(tf.sqrt(noise_var / 2.0), tf.float32)
        noise = tf.complex(
            tf.random.normal(tf.shape(y), stddev=stddev),
            tf.random.normal(tf.shape(y), stddev=stddev))
        y = y + noise
        h_eq = y / h_batch
        noise_var_eq = noise_var / (tf.cast(tf.abs(h_batch)**2, tf.float32) + 1e-30)
        llr = self.demapper(h_eq, noise_var_eq)
        bits_hat = self.decoder(llr)
        block_err = tf.reduce_any(tf.not_equal(bits_hat, bits), axis=-1)
        return tf.cast(block_err, tf.int32)


_link_cache = {}

def _get_link(mcs_idx):
    if mcs_idx not in _link_cache:
        _link_cache[mcs_idx] = CodedOFDMLink(mcs_idx)
    return _link_cache[mcs_idx]


def bler_ci(n_errors, n_total, confidence=0.95):
    a = 1 - confidence
    lo = 0.0 if n_errors == 0 else float(
        beta_dist.ppf(a / 2, n_errors, n_total - n_errors + 1))
    hi = 1.0 if n_errors == n_total else float(
        beta_dist.ppf(1 - a / 2, n_errors + 1, n_total - n_errors))
    return lo, hi


def evaluate_bler(mcs_idx, cfr_frames, noise_var):
    """Coded BLER on measured frames with perfect-CSI equalization."""
    if mcs_idx == 0:
        n = cfr_frames.shape[0]
        return 1.0, n, n, 1.0, 1.0
    tf.random.set_seed(42)
    link = _get_link(mcs_idx)
    n_sym = link.n_coded_bits // link.num_bps
    n_sc = cfr_frames.shape[1]
    reps = (n_sym + n_sc - 1) // n_sc
    h_tiled = np.tile(cfr_frames, (1, reps))[:, :n_sym]
    h_batch = tf.constant(h_tiled, dtype=tf.complex64)
    nv = tf.constant(noise_var, dtype=tf.float32)
    block_errs = link(h_batch, nv)
    n_total = int(tf.shape(block_errs)[0])
    n_errors = int(tf.reduce_sum(block_errs).numpy())
    bler = float(n_errors) / max(n_total, 1)
    ci_lo, ci_hi = bler_ci(n_errors, n_total)
    return bler, n_errors, n_total, ci_lo, ci_hi


# ── Main comparison ──────────────────────────────────────────────────────

def run_integrated():
    """5 positions × position-specific SNR.

    Strategies:
      1a. 3GPP InF-SL + LS estimation (original baseline)
      1b. 3GPP InF-SL + LMMSE (empirical R_hh from 20 sounding positions)
      1c. 3GPP InF-SL + No penalty (perfect CSI upper bound)
      2.  DT-Derived LMMSE + Q-D (R_hh from 3004 grid, position-independent)
      3.  DT-Assisted LMMSE + Q-D (R_ee per-position)
    """

    snr_thresholds = load_snr_thresholds()
    qd_model = load_qd_model()
    R_ee_env, corr_dict, nmse_dict = load_ree()
    rhh_eigvals, rhh_eigvecs, rhh_trace_per_sc, rhh_n_pos = load_rhh()
    emp_eigvals, emp_eigvecs, emp_trace_per_sc, emp_n_pos = load_rhh_empirical()

    print("=" * 80)
    print("MCS TABLE (11 entries, 3GPP TS 38.214 Table 5.1.3.1-1)")
    print("-" * 80)
    print(f"{'MCS':>4s}  {'Mod':>6s}  {'Rate':>6s}  {'SE':>6s}  {'Thr (dB)':>9s}")
    for idx in MCS_INDICES:
        e = MCS_TABLE[idx]
        se = spectral_efficiency(idx)
        print(f"{idx:4d}  {e['mod_name']:>6s}  {e['code_rate']:.3f}  "
              f"{se:6.2f}  {snr_thresholds[idx]:9.2f}")
    print("=" * 80)

    print(f"\nR_hh (DT-Derived): {rhh_n_pos} grid positions, "
          f"trace/sc={rhh_trace_per_sc:.2e} "
          f"({10*np.log10(rhh_trace_per_sc+1e-30):.1f} dB)")
    print(f"R_hh (Empirical):  {emp_n_pos} sounding positions, "
          f"trace/sc={emp_trace_per_sc:.2e} "
          f"({10*np.log10(emp_trace_per_sc+1e-30):.1f} dB)")

    print(f"\nExperiment: 5 positions × position-specific SNR")
    print(f"{'Pos':>7s}  {'SNR':>5s}  {'ρ':>5s}")
    print("-" * 30)
    for pos_key, snr_db in EXPERIMENT_POSITIONS.items():
        rho = corr_dict.get(pos_key, 0.0)
        print(f"{pos_key:>7s}  {snr_db:+5.1f}  {rho:.3f}")
    print()

    results = {}

    for pos_key, snr_db in EXPERIMENT_POSITIONS.items():
        pos_xyz = RX_POSITIONS[pos_key]
        noise_var = 1.0 / (10 ** (snr_db / 10))
        corr = float(corr_dict.get(pos_key, 0.0))

        h_dt = load_hdt(pos_key)
        meas_cfr = load_measured_cfr_frames(pos_key)

        R_ee_pos, ree_scale = scale_ree_for_position(R_ee_env, nmse_dict, pos_key)

        # ── Estimation MSE: five levels ──
        mse_ls = compute_mse_ls(noise_var)
        mse_emp_lmmse = compute_mse_lmmse(
            emp_eigvals, emp_eigvecs, emp_trace_per_sc, noise_var)
        mse_derived = compute_mse_lmmse(
            rhh_eigvals, rhh_eigvecs, rhh_trace_per_sc, noise_var)
        mse_assisted = compute_mse_dt_assisted_lmmse(R_ee_pos, noise_var)

        # ── 3GPP strategies with 3 estimation levels ──
        rng_ls = np.random.default_rng(42)
        mcs_3gpp_ls = strategy_3gpp_inf(
            pos_xyz, NOMINAL_TX_POS, snr_thresholds,
            operating_snr_db=snr_db, rng=rng_ls,
            estimation='ls')

        rng_lm = np.random.default_rng(42)
        mcs_3gpp_lmmse = strategy_3gpp_inf(
            pos_xyz, NOMINAL_TX_POS, snr_thresholds,
            operating_snr_db=snr_db, rng=rng_lm,
            estimation='lmmse',
            rhh_eigvals=emp_eigvals, rhh_eigvecs=emp_eigvecs,
            rhh_trace_per_sc=emp_trace_per_sc)

        rng_pf = np.random.default_rng(42)
        mcs_3gpp_perfect = strategy_3gpp_inf(
            pos_xyz, NOMINAL_TX_POS, snr_thresholds,
            operating_snr_db=snr_db, rng=rng_pf,
            estimation='perfect')

        # ── DT strategies ──
        mcs_derived_qd, gamma_derived_dist = strategy_qd_estimation_aware(
            h_dt, qd_model, snr_thresholds,
            operating_snr_db=snr_db,
            mse_per_sc=mse_derived,
            n_drops=200,
            rng=np.random.default_rng(42))

        mcs_assisted_qd, gamma_assisted_dist = strategy_qd_estimation_aware(
            h_dt, qd_model, snr_thresholds,
            operating_snr_db=snr_db,
            mse_per_sc=mse_assisted,
            n_drops=200,
            rng=np.random.default_rng(42))

        # ── BLER validation on measured channels ──
        bler_ls, ne_ls, nt_ls, ci_ls_l, ci_ls_h = evaluate_bler(
            mcs_3gpp_ls, meas_cfr, noise_var)
        bler_lm, ne_lm, nt_lm, ci_lm_l, ci_lm_h = evaluate_bler(
            mcs_3gpp_lmmse, meas_cfr, noise_var)
        bler_pf, ne_pf, nt_pf, ci_pf_l, ci_pf_h = evaluate_bler(
            mcs_3gpp_perfect, meas_cfr, noise_var)
        bler_derived, ned, ntd, cidl, cidh = evaluate_bler(
            mcs_derived_qd, meas_cfr, noise_var)
        bler_assisted, nea, nta, cial, ciah = evaluate_bler(
            mcs_assisted_qd, meas_cfr, noise_var)

        tput_ls = effective_throughput(mcs_3gpp_ls, bler_ls)
        tput_lm = effective_throughput(mcs_3gpp_lmmse, bler_lm)
        tput_pf = effective_throughput(mcs_3gpp_perfect, bler_pf)
        tput_derived = effective_throughput(mcs_derived_qd, bler_derived)
        tput_assisted = effective_throughput(mcs_assisted_qd, bler_assisted)

        sinr_pen_ls = 10 * np.log10(1 + mse_ls / noise_var)
        sinr_pen_lm = 10 * np.log10(1 + mse_emp_lmmse.mean() / noise_var)
        sinr_pen_derived = 10 * np.log10(1 + mse_derived.mean() / noise_var)
        sinr_pen_assisted = 10 * np.log10(1 + mse_assisted.mean() / noise_var)

        print(f"{pos_key} @ {snr_db:+.1f} dB (ρ={corr:.3f}):")
        print(f"  MSE: LS={10*np.log10(mse_ls):+.1f}, "
              f"LMMSE(emp)={10*np.log10(mse_emp_lmmse.mean()):+.1f}, "
              f"DT-Derived={10*np.log10(mse_derived.mean()):+.1f}, "
              f"DT-Assisted={10*np.log10(mse_assisted.mean()):+.1f} dB")
        print(f"  SINR penalty: LS={sinr_pen_ls:.2f}, "
              f"LMMSE={sinr_pen_lm:.2f}, "
              f"Perfect=0.00, "
              f"DT-Derived={sinr_pen_derived:.2f}, "
              f"DT-Assisted={sinr_pen_assisted:.2f} dB")
        print(f"  3GPP+LS           → {mcs_label(mcs_3gpp_ls):30s} "
              f"BLER={bler_ls:.3f}  Tput={tput_ls:.2f}")
        print(f"  3GPP+LMMSE        → {mcs_label(mcs_3gpp_lmmse):30s} "
              f"BLER={bler_lm:.3f}  Tput={tput_lm:.2f}")
        print(f"  3GPP+Perfect      → {mcs_label(mcs_3gpp_perfect):30s} "
              f"BLER={bler_pf:.3f}  Tput={tput_pf:.2f}")
        print(f"  DT-Derived+Q-D    → {mcs_label(mcs_derived_qd):30s} "
              f"BLER={bler_derived:.3f}  Tput={tput_derived:.2f}")
        print(f"  DT-Assisted+Q-D   → {mcs_label(mcs_assisted_qd):30s} "
              f"BLER={bler_assisted:.3f}  Tput={tput_assisted:.2f}")

        results[pos_key] = {
            'snr_db': snr_db,
            'actual_pos': pos_key,
            'position': list(pos_xyz),
            'dt_correlation': corr,
            'ree_scale': float(ree_scale),
            'mse_ls_db': float(10 * np.log10(mse_ls)),
            'mse_lmmse_emp_avg_db': float(10 * np.log10(mse_emp_lmmse.mean())),
            'mse_dt_derived_avg_db': float(10 * np.log10(mse_derived.mean())),
            'mse_dt_assisted_avg_db': float(10 * np.log10(mse_assisted.mean())),
            'sinr_penalty_ls_db': float(sinr_pen_ls),
            'sinr_penalty_lmmse_db': float(sinr_pen_lm),
            'sinr_penalty_dt_derived_db': float(sinr_pen_derived),
            'sinr_penalty_dt_assisted_db': float(sinr_pen_assisted),
            '3GPP_InF_LS': {
                'estimation': 'LS',
                'mcs': mcs_3gpp_ls,
                'mcs_label': mcs_label(mcs_3gpp_ls),
                'se': spectral_efficiency(mcs_3gpp_ls),
                'bler': bler_ls,
                'bler_ci': [ci_ls_l, ci_ls_h],
                'compliant': bler_ls <= BLER_TARGET,
                'throughput': tput_ls,
            },
            '3GPP_InF_LMMSE': {
                'estimation': 'LMMSE (empirical R_hh, 20 positions)',
                'mcs': mcs_3gpp_lmmse,
                'mcs_label': mcs_label(mcs_3gpp_lmmse),
                'se': spectral_efficiency(mcs_3gpp_lmmse),
                'bler': bler_lm,
                'bler_ci': [ci_lm_l, ci_lm_h],
                'compliant': bler_lm <= BLER_TARGET,
                'throughput': tput_lm,
            },
            '3GPP_InF_Perfect': {
                'estimation': 'Perfect CSI (no penalty)',
                'mcs': mcs_3gpp_perfect,
                'mcs_label': mcs_label(mcs_3gpp_perfect),
                'se': spectral_efficiency(mcs_3gpp_perfect),
                'bler': bler_pf,
                'bler_ci': [ci_pf_l, ci_pf_h],
                'compliant': bler_pf <= BLER_TARGET,
                'throughput': tput_pf,
            },
            'DT_Derived_QD': {
                'receiver': 'DT-Derived LMMSE',
                'covariance': 'R_hh (3004 grid positions)',
                'mcs': mcs_derived_qd,
                'mcs_label': mcs_label(mcs_derived_qd),
                'se': spectral_efficiency(mcs_derived_qd),
                'bler': bler_derived,
                'bler_ci': [cidl, cidh],
                'compliant': bler_derived <= BLER_TARGET,
                'throughput': tput_derived,
            },
            'DT_Assisted_QD': {
                'receiver': 'DT-Assisted LMMSE',
                'covariance': 'R_ee (20 sounding positions, per-position scaled)',
                'mcs': mcs_assisted_qd,
                'mcs_label': mcs_label(mcs_assisted_qd),
                'se': spectral_efficiency(mcs_assisted_qd),
                'bler': bler_assisted,
                'bler_ci': [cial, ciah],
                'compliant': bler_assisted <= BLER_TARGET,
                'throughput': tput_assisted,
            },
        }
        print()

    # ── Summary table ──
    W = 155
    print("=" * W)
    print("RESULTS TABLE (5 positions × position-specific SNR)")
    print("=" * W)
    hdr = (f"{'Pos':>7s}  {'SNR':>4s}  {'ρ':>5s}  │ "
           f"{'3GPP+LS':>7s} {'BLER':>5s} {'Tput':>5s} │ "
           f"{'3GPP+LM':>7s} {'BLER':>5s} {'Tput':>5s} │ "
           f"{'3GPP+Pf':>7s} {'BLER':>5s} {'Tput':>5s} │ "
           f"{'DT+QD':>6s} {'BLER':>5s} {'Tput':>5s} │ "
           f"{'G_LM':>5s} {'G_Pf':>5s} {'G_DT':>5s}")
    print(hdr)
    print("-" * W)

    totals = {'ls': 0, 'lm': 0, 'pf': 0, 'der': 0, 'ast': 0}
    n_comply = {'ls': 0, 'lm': 0, 'pf': 0, 'der': 0, 'ast': 0}

    for pos_key in EXPERIMENT_POSITIONS:
        r = results[pos_key]
        gl = r['3GPP_InF_LS']
        gm = r['3GPP_InF_LMMSE']
        gp = r['3GPP_InF_Perfect']
        da = r['DT_Assisted_QD']

        def gain_pct(tput, base):
            if base > 0.01:
                return f"{(tput/base - 1)*100:+.0f}%"
            return "∞"

        print(f"{pos_key:>7s}  {r['snr_db']:+5.1f}  {r['dt_correlation']:.3f}  │ "
              f"{'MCS'+str(gl['mcs']):>7s} {gl['bler']:5.3f} {gl['throughput']:5.2f} │ "
              f"{'MCS'+str(gm['mcs']):>7s} {gm['bler']:5.3f} {gm['throughput']:5.2f} │ "
              f"{'MCS'+str(gp['mcs']):>7s} {gp['bler']:5.3f} {gp['throughput']:5.2f} │ "
              f"{'MCS'+str(da['mcs']):>7s} {da['bler']:5.3f} {da['throughput']:5.2f} │ "
              f"{gain_pct(gm['throughput'], gl['throughput']):>5s} "
              f"{gain_pct(gp['throughput'], gl['throughput']):>5s} "
              f"{gain_pct(da['throughput'], gl['throughput']):>5s}")

        totals['ls'] += gl['throughput']
        totals['lm'] += gm['throughput']
        totals['pf'] += gp['throughput']
        totals['ast'] += da['throughput']
        n_comply['ls'] += int(gl['compliant'])
        n_comply['lm'] += int(gm['compliant'])
        n_comply['pf'] += int(gp['compliant'])
        n_comply['ast'] += int(da['compliant'])

    print("-" * W)
    avg_ls = totals['ls'] / 5
    avg_lm = totals['lm'] / 5
    avg_pf = totals['pf'] / 5
    avg_ast = totals['ast'] / 5
    print(f"{'Avg':>7s}  {'':>4s}  {'':>5s}  │ "
          f"{'':>7s} {'':>5s} {avg_ls:5.2f} │ "
          f"{'':>7s} {'':>5s} {avg_lm:5.2f} │ "
          f"{'':>7s} {'':>5s} {avg_pf:5.2f} │ "
          f"{'':>7s} {'':>5s} {avg_ast:5.2f} │ "
          f"{(avg_lm/avg_ls-1)*100:+4.0f}% "
          f"{(avg_pf/avg_ls-1)*100:+4.0f}% "
          f"{(avg_ast/avg_ls-1)*100:+4.0f}%")
    print(f"{'Comply':>7s}  {'':>4s}  {'':>5s}  │ "
          f"{'':>7s} {n_comply['ls']}/5  {'':>5s} │ "
          f"{'':>7s} {n_comply['lm']}/5  {'':>5s} │ "
          f"{'':>7s} {n_comply['pf']}/5  {'':>5s} │ "
          f"{'':>7s} {n_comply['ast']}/5  {'':>5s} │")
    print("=" * W)

    # ── Estimation pipeline connection ──
    print(f"\nEstimation Pipeline Connection (per position):")
    print(f"{'Pos':>7s}  {'SNR':>4s}  {'MSE_LS':>7s}  {'MSE_LM':>7s}  "
          f"{'MSE_Der':>7s}  {'MSE_Ast':>7s}  "
          f"{'Pen_LS':>6s}  {'Pen_LM':>6s}  {'Pen_Der':>7s}  {'Pen_Ast':>7s}")
    print("-" * 95)
    for pos_key in EXPERIMENT_POSITIONS:
        r = results[pos_key]
        print(f"{pos_key:>7s}  {r['snr_db']:+5.1f}  "
              f"{r['mse_ls_db']:+7.1f}  {r['mse_lmmse_emp_avg_db']:+7.1f}  "
              f"{r['mse_dt_derived_avg_db']:+7.1f}  "
              f"{r['mse_dt_assisted_avg_db']:+7.1f}  "
              f"{r['sinr_penalty_ls_db']:6.2f}  {r['sinr_penalty_lmmse_db']:6.2f}  "
              f"{r['sinr_penalty_dt_derived_db']:7.2f}  "
              f"{r['sinr_penalty_dt_assisted_db']:7.2f}")

    # ── Save ──
    output = {
        'config': {
            'mcs_table': {str(k): v for k, v in MCS_TABLE.items()},
            'snr_thresholds': {str(k): v for k, v in snr_thresholds.items()},
            'experiment_positions': EXPERIMENT_POSITIONS,
            'bw_hz': BW_HZ,
            'fft': FFT,
            'n_occ': N_OCC,
            'n_pilots': N_PILOTS,
            'bler_target': BLER_TARGET,
            'rhh_dt_n_positions': rhh_n_pos,
            'rhh_dt_trace_per_sc_db': float(10 * np.log10(rhh_trace_per_sc + 1e-30)),
            'rhh_emp_n_positions': emp_n_pos,
            'rhh_emp_trace_per_sc_db': float(10 * np.log10(emp_trace_per_sc + 1e-30)),
        },
        'results': results,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, 'mcs_integrated_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x))
    print(f"\nSaved → {out_path}")

    return output


if __name__ == '__main__':
    run_integrated()
