"""Generate a tiny synthetic stand-in for the PhysioNet/CinC 2016 training folders.
ONLY for smoke-testing the code -- the numbers it produces mean nothing clinically.
Layout: <out>/training-a/REFERENCE.csv + a0001.wav ...  (label: -1 normal, 1 abnormal)
"""
import os
import sys
import numpy as np
from scipy.io import wavfile

FS = 2000


def beat(t, f0, width):
    return np.exp(-0.5 * ((t) / width) ** 2) * np.sin(2 * np.pi * f0 * t)


def make_record(rng, abnormal, dur):
    n = int(dur * FS)
    x = np.zeros(n)
    hr = rng.uniform(60, 90) / 60
    period = int(FS / hr)
    for start in range(int(0.1 * FS), n - period, period):
        for off, f0 in ((0, 45), (int(0.3 * FS), 70)):                       # S1, S2
            t = (np.arange(-100, 100)) / FS
            i = start + off
            x[i - 100:i + 100] += beat(t, f0, 0.012)[:min(200, n - (i - 100))] * rng.uniform(.8, 1.2)
        if abnormal:                                                          # systolic "murmur"
            m = int(0.2 * FS)
            i = start + int(0.05 * FS)
            x[i:i + m] += rng.normal(0, 0.15, m) * np.hanning(m)
    x += rng.normal(0, 0.02 + 0.03 * abnormal * rng.random(), n)
    x *= rng.uniform(2000, 8000)
    return np.clip(x, -32000, 32000).astype(np.int16)


def main(out, n_rec=120, seed=0):
    rng = np.random.default_rng(seed)
    for sub in ("training-a", "training-b"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    refs = {"training-a": [], "training-b": []}
    for i in range(n_rec):
        sub = "training-a" if i % 2 == 0 else "training-b"
        abn = rng.random() < 0.3
        rec = f"{sub[-1]}{i:04d}"
        wavfile.write(os.path.join(out, sub, rec + ".wav"), FS, make_record(rng, abn, rng.uniform(10, 30)))
        refs[sub].append((rec, 1 if abn else -1))
    for sub, rows in refs.items():
        with open(os.path.join(out, sub, "REFERENCE.csv"), "w") as f:
            f.writelines(f"{r},{l}\n" for r, l in rows)
    print("synthetic data written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
