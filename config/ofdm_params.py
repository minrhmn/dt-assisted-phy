"""Shared OFDM system parameters and utility functions."""

import os
import numpy as np

FC = 3.5e9
BW = 50e6
FFT = 256
CP = 64
SYM = FFT + CP
N_OCC = 192
N_DATA_SYM = 12
N_GRID_SYM = 14       # P0 + 12 data + P1
FRAME_SYMS = 16       # SC2 + P0 + D0-D11 + P1 + Guard
FRAME_LEN = FRAME_SYMS * SYM

GUARD_LEFT = 32
GUARD_RIGHT = 31

OCCUPIED_POS = np.array(list(range(-96, 0)) + list(range(1, 97)))
OCC_BINS = OCCUPIED_POS % FFT  # [160..255, 1..96]

QPSK_MAP = np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=np.complex64) / np.sqrt(2)
BITS_PER_SYM = 2

TRAIN_MEASURED = [f'posp{i}' for i in range(1, 17)]
TEST_MEASURED  = [f'posp{i}' for i in range(17, 21)]

BW_OPTIONS = {'20m': 20e6, '25m': 25e6, '50m': 50e6}
BW_NORM = {'20m': 0.0, '25m': 0.25, '50m': 1.0}
Z_VALS = [0.5, 1.0, 1.5, 2.0, 2.5]
NOMINAL_TX_POS = [-6.0, 6.2, 2.5]

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR    = os.path.join(_ROOT, 'data', 'channel')
MODELS_DIR  = os.path.join(_ROOT, 'models')
RESULTS_DIR = os.path.join(_ROOT, 'data', 'ota_results')
OUTPUT_DIR  = os.path.join(_ROOT, 'output')

RX_POSITIONS = {
    'posp1': [-0.9, 2.5, 1.0], 'posp2': [-6.6, -5.2, 1.0],
    'posp3': [-10.5, -4.9, 1.0], 'posp4': [-0.7, -5.4, 1.0],
    'posp5': [4.9, -3.6, 1.0], 'posp6': [7.9, -0.3, 1.0],
    'posp7': [7.8, 5.5, 1.0], 'posp8': [0.5, 0.0, 1.0],
    'posp9': [-9.8, 1.3, 1.0], 'posp10': [-8.7, 6.3, 1.0],
    'posp11': [-2.8, 0.0, 1.0], 'posp12': [-6.1, 0.0, 1.0],
    'posp13': [-10.7, 0.0, 1.0], 'posp14': [-10.1, -2.9, 1.0],
    'posp15': [-7.0, -2.9, 1.0], 'posp16': [-4.1, -2.9, 1.0],
    'posp17': [-1.6, -2.9, 1.0], 'posp18': [8.5, -2.9, 1.0],
    'posp19': [8.5, -5.0, 1.0], 'posp20': [3.7, -6.9, 1.0],
}


def cir_to_cfr(a_re, a_im, tau, bw_hz, n_fft=FFT):
    """Convert raw CIR (path amplitudes + delays) to CFR at given bandwidth."""
    from scipy.fft import fftfreq
    freqs = fftfreq(n_fft, d=1.0 / bw_hz)
    a = (a_re + 1j * a_im).astype(np.complex64)
    phase = np.exp(-1j * 2 * np.pi * np.outer(freqs, tau)).astype(np.complex64)
    return (phase @ a).astype(np.complex64)


def load_cir_as_cfr(npz_path, pos_key, bw_label='50m'):
    """Load a position's raw CIR from .npz and return CFR at given BW."""
    d = np.load(npz_path)
    a_re = d[f'{pos_key}_a_re']
    a_im = d[f'{pos_key}_a_im']
    tau = d[f'{pos_key}_tau']
    bw_hz = BW_OPTIONS[bw_label]
    return cir_to_cfr(a_re, a_im, tau, bw_hz)


def ebnodb_to_noise_var(ebn0_db, bits_per_sym=2):
    """Convert Eb/N0 (dB) to noise variance per complex sample."""
    ebn0_lin = 10 ** (ebn0_db / 10)
    snr = ebn0_lin * bits_per_sym * (FFT / SYM)
    return 1.0 / snr


# External data paths (hardware-specific, not included in repo)
TX_WAVEFORM_ROOT = '/home/native/project/ota_ofdm_processing'
CAPTURES_ROOT = '/home/native/project/captures'
SOUNDING_DIR = '/home/native/project/results_v4'
SCENE_PATH = '/home/native/project/weeks_hall_refined/weeks_hall_refined.xml'


def tx_waveform_path(modulation='qpsk', bw_label='50m'):
    return os.path.join(TX_WAVEFORM_ROOT, f'tx_waveform_{modulation}_{bw_label}.npz')


def load_pilots(modulation='qpsk'):
    """Load P0/P1 pilot values on occupied bins from TX waveform file."""
    path = tx_waveform_path(modulation, '50m')
    d = np.load(path)
    p0 = d['p0_freq'][OCC_BINS].astype(np.complex64)
    p1 = d['p1_freq'][OCC_BINS].astype(np.complex64)
    return p0, p1


def load_const_map(modulation='qpsk'):
    """Load constellation map and bits_per_symbol for the given modulation."""
    if modulation == 'qpsk':
        return QPSK_MAP, BITS_PER_SYM
    d = np.load(tx_waveform_path('16qam', '50m'))
    return d['const_map'].astype(np.complex64), int(d['bits_per_symbol'])


def ota_captures_dir(pos='p1', bw_label='50m', modulation='qpsk'):
    bw_mhz = bw_label.replace('m', 'mhz')
    return os.path.join(CAPTURES_ROOT, pos, bw_mhz, modulation)


def dense_grid_path(bw_label='50m'):
    return os.path.join(DATA_DIR, f'rt_hdt_dense_grid_d7r_{bw_label}.npz')


def measured_path(bw_label='50m'):
    return os.path.join(DATA_DIR, 'cir_measured_d7r_cal.npz')


def ota_cir_path():
    return os.path.join(DATA_DIR, 'cir_ota_d7r_cal.npz')


def error_stats_path(bw_label='50m'):
    return os.path.join(DATA_DIR, f'dt_error_stats_{bw_label}.npz')
