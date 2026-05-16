"""EESM effective SNR mapping and MCS selection.

Exponential Effective SNR Mapping compresses a vector of per-subcarrier
SNRs into a single scalar effective SNR suitable for AWGN BLER lookup.
The beta parameter is MCS-dependent and calibrated from 3GPP TDL channels.
"""

import numpy as np

# 3GPP TS 38.214 Table 5.1.3.1-1 subset.
# MCS 0/2 from task spec had R<1/5 which Sionna's LDPC doesn't support;
# replaced with MCS 1 (R=0.23) and MCS 3 (R=0.25) to cover low-SNR regime.
# mod_order = constellation size (4=QPSK, 16=16QAM, 64=64QAM),
# so num_bits_per_symbol = log2(mod_order).
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

# EESM beta per MCS (3GPP TR 36.942 calibrated for TDL channels).
# NOTE: these betas are NOT recalibrated for our QD channel model.
# Per-MCS recalibration is deferred (GPU-expensive). Using TDL betas
# may be slightly conservative or aggressive depending on the channel's
# effective diversity order relative to TDL.
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


def compute_eesm(snr_linear_per_sc, beta):
    """Map per-subcarrier SNRs (linear) to a single effective SNR (linear).

    gamma_eff = -beta * ln(1/N * sum(exp(-gamma_k / beta)))
    """
    ratio = np.clip(snr_linear_per_sc / beta, 0, 50)
    gamma_eff = -beta * np.log(np.mean(np.exp(-ratio)))
    return gamma_eff


def select_mcs(snr_linear_per_sc, snr_thresholds):
    """Select highest MCS where EESM effective SNR exceeds the BLER threshold.

    Args:
        snr_linear_per_sc: array [N_sc], per-subcarrier SNR (linear)
        snr_thresholds: dict {mcs_idx: snr_threshold_db}

    Returns:
        (selected_mcs, gamma_eff_db) for the winning MCS
    """
    best_mcs = 0
    best_gamma_eff_db = -np.inf

    for mcs_idx in sorted(snr_thresholds.keys()):
        beta = EESM_BETA.get(mcs_idx, 5.0)
        gamma_eff = compute_eesm(snr_linear_per_sc, beta)
        gamma_eff_db = 10 * np.log10(max(gamma_eff, 1e-10))

        if gamma_eff_db >= snr_thresholds[mcs_idx]:
            best_mcs = mcs_idx
            best_gamma_eff_db = gamma_eff_db

    return best_mcs, best_gamma_eff_db


def spectral_efficiency(mcs_idx):
    """SE = Qm * R  [bits/s/Hz]."""
    if mcs_idx == 0:
        return 0.0
    entry = MCS_TABLE[mcs_idx]
    return int(np.log2(entry['mod_order'])) * entry['code_rate']


def effective_throughput(mcs_idx, bler):
    """T_eff = SE * (1 - BLER) [bits/s/Hz]."""
    return spectral_efficiency(mcs_idx) * (1.0 - bler)


def mcs_label(mcs_idx):
    """Human-readable label for an MCS index."""
    if mcs_idx == 0:
        return "MCS0 (none)"
    entry = MCS_TABLE[mcs_idx]
    qm = int(np.log2(entry['mod_order']))
    return f"MCS{mcs_idx} ({entry['mod_name']} R={entry['code_rate']:.3f})"
