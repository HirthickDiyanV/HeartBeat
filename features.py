"""Preprocessing, segmentation and the 27 handcrafted features (Alhaj & Ozturk, 2026).

Feature groups (Table 2 of the paper):
  Statistical (9): mean, median, std, MAD, Q25, Q75, IQR, skewness, kurtosis
  Entropy (2):     sample entropy, spectral entropy
  Frequency (3):   DFV, DFM, DFR  (PSD up to 256 Hz)
  MFCC (13):       MFCC1..MFCC13 averaged over 25 ms / 10 ms frames
"""
import os
import numpy as np
import pandas as pd
from scipy import signal, stats, fft
from scipy.io import wavfile
from sklearn.neighbors import KDTree

FS = 2000                 # all PhysioNet/CinC 2016 recordings are 2 kHz
SEG_SEC = 5               # non-overlapping 5 s windows
SEG_LEN = FS * SEG_SEC

STAT_NAMES = ["meanValue", "medianValue", "standardDeviation", "meanAbsoluteDeviation",
              "quantile25", "quantile75", "signalIQR", "sampleSkewness", "sampleKurtosis"]
ENTROPY_NAMES = ["signalEntropy", "spectralEntropy"]
FREQ_NAMES = ["dominantFrequencyValue", "dominantFrequencyMagnitude", "dominantFrequencyRatio"]
MFCC_NAMES = [f"MFCC{i}" for i in range(1, 14)]

EXP1_FEATURES = STAT_NAMES + ENTROPY_NAMES + FREQ_NAMES          # 14
EXP2_FEATURES = MFCC_NAMES                                       # 13
EXP3_FEATURES = EXP1_FEATURES + EXP2_FEATURES                    # 27


# ----------------------------------------------------------------- preprocessing
_SOS = signal.butter(4, [25, 400], btype="bandpass", fs=FS, output="sos")


def preprocess(x):
    """4th-order Butterworth band-pass 25-400 Hz, then z-score (Eq. 1)."""
    x = np.asarray(x, dtype=np.float64)
    y = signal.sosfiltfilt(_SOS, x)
    sd = y.std()
    return (y - y.mean()) / (sd if sd > 0 else 1.0)


def segment(x, seg_len=SEG_LEN):
    """Non-overlapping fixed windows; the trailing remainder is dropped."""
    n = len(x) // seg_len
    return [x[i * seg_len:(i + 1) * seg_len] for i in range(n)]


# ----------------------------------------------------------------- feature pieces
def sample_entropy(x, m=2, r_frac=0.2, decimate_to=500):
    """Sample entropy with Chebyshev distance and tolerance r = r_frac * std.

    Segments are decimated to `decimate_to` Hz first to keep runtime reasonable
    (the paper does not state this; set decimate_to=None for the full-rate value).
    """
    if decimate_to and decimate_to < FS:
        x = signal.decimate(x, FS // decimate_to, zero_phase=True)
    x = np.asarray(x, dtype=np.float64)
    r = r_frac * x.std()
    if r == 0 or len(x) <= m + 2:
        return 0.0
    n = len(x) - m

    def count_pairs(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[:n]
        tree = KDTree(emb, metric="chebyshev")
        c = tree.query_radius(emb, r=r, count_only=True)
        return (c - 1).sum()          # drop self matches

    b = count_pairs(m)
    a = count_pairs(m + 1)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(a / b))


def psd_features(x, fmax=256):
    f, p = signal.welch(x, fs=FS, nperseg=1024)
    keep = f <= fmax
    f, p = f[keep], p[keep]
    tot = p.sum()
    i = int(np.argmax(p))
    dfv, dfm = f[i], p[i]
    dfr = dfm / tot if tot > 0 else 0.0
    pn = p / tot if tot > 0 else np.ones_like(p) / len(p)
    pn = pn[pn > 0]
    spec_ent = float(-(pn * np.log2(pn)).sum() / np.log2(len(p)))   # normalised to [0, 1]
    return spec_ent, float(dfv), float(dfm), float(dfr)


def _mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)           # Eq. (2)


def _inv_mel(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_filters=26, n_fft=256, fmin=0.0, fmax=FS / 2):
    pts = _inv_mel(np.linspace(_mel(fmin), _mel(fmax), n_filters + 2))
    bins = np.floor((n_fft + 1) * pts / FS).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB = _mel_filterbank()


def mfcc_mean(x, n_mfcc=13, win_ms=25, hop_ms=10, n_fft=256, skip_c0=True):
    """13 MFCCs per 25 ms Hamming frame (10 ms hop), DCT of log-Mel energies (Eq. 3),
    averaged over all frames of the segment. MFCC1..13 = DCT coefficients 1..13
    (c0 skipped) -- the paper's naming implies this; set skip_c0=False for c0..c12."""
    win, hop = int(FS * win_ms / 1000), int(FS * hop_ms / 1000)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * np.hamming(win)
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    logmel = np.log(spec @ _FB.T + 1e-10)
    c = fft.dct(logmel, type=2, axis=1, norm="ortho")
    c = c[:, 1:n_mfcc + 1] if skip_c0 else c[:, :n_mfcc]
    return c.mean(axis=0)


def extract_features(seg):
    """Return the 27-dim feature vector (ordered as EXP3_FEATURES) for one segment."""
    q25, q75 = np.percentile(seg, [25, 75])
    stat = [seg.mean(), np.median(seg), seg.std(ddof=1),
            np.mean(np.abs(seg - seg.mean())), q25, q75, q75 - q25,
            stats.skew(seg, bias=False), stats.kurtosis(seg, bias=False)]
    spec_ent, dfv, dfm, dfr = psd_features(seg)
    ent = [sample_entropy(seg), spec_ent]
    return np.concatenate([stat, ent, [dfv, dfm, dfr], mfcc_mean(seg)])


# ----------------------------------------------------------------- dataset
def read_references(root):
    """Collect (subset, record, label) from training-a..f/REFERENCE.csv.
    PhysioNet encodes -1 = normal, 1 = abnormal; we return abnormal=1, normal=0."""
    rows = []
    for sub in sorted(d for d in os.listdir(root) if d.startswith("training-")):
        ref = os.path.join(root, sub, "REFERENCE.csv")
        if not os.path.exists(ref):
            continue
        df = pd.read_csv(ref, header=None, names=["record", "label"])
        for rec, lab in zip(df.record, df.label):
            rows.append((sub, str(rec), 1 if int(lab) == 1 else 0))
    return pd.DataFrame(rows, columns=["subset", "record", "label"])


def _process_recording(args):
    root, sub, rec, label = args
    wav_path = os.path.join(root, sub, rec + ".wav")
    dat_path = os.path.join(root, sub, rec + ".dat")
    if os.path.exists(wav_path):
        fs, x = wavfile.read(wav_path)
    elif os.path.exists(dat_path):
        fs = FS
        x = np.fromfile(dat_path, dtype=np.int16)
    else:
        raise FileNotFoundError(f"Neither {rec}.wav nor {rec}.dat found in {os.path.join(root, sub)}")
    if fs != FS:
        x = signal.resample_poly(x.astype(np.float64), FS, fs)
    out = []
    for k, seg in enumerate(segment(preprocess(x))):
        out.append([sub, rec, k, label, *extract_features(seg)])
    return out


def build_feature_table(root, out_csv, n_jobs=-1, limit=None):
    """Run the whole extraction pipeline -> FeatureTable.csv (one row per 5 s segment)."""
    from joblib import Parallel, delayed
    refs = read_references(root)
    if limit:
        refs = refs.sample(limit, random_state=0)
    jobs = [(root, r.subset, r.record, r.label) for r in refs.itertuples()]
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_process_recording)(j) for j in jobs)
    rows = [row for res in results for row in res]
    cols = ["subset", "recording", "segment", "label"] + EXP3_FEATURES
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)
    return df
