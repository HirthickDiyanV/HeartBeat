# Heart sound classification: MFCC + statistical features (Alhaj & Ozturk, 2026)

Implementation of the pipeline in *Discover Computing* 29:707 (binary Normal/Abnormal PCG
classification on PhysioNet/CinC 2016).

## Setup
```
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib joblib
```
Download the data: https://physionet.org/content/challenge-2016/1.0.0/ (`training.zip`),
unzip so you have `training-a/ ... training-f/` each with `*.wav` and `REFERENCE.csv`.

## Run
```
python run_experiments.py extract --data /path/to/training --out FeatureTable.csv   # ~13k segments
python run_experiments.py run --features FeatureTable.csv --out results
```
Smoke test without the real data: `python make_synthetic.py synthetic_data`, then extract from it
(synthetic numbers are meaningless; this only checks the code runs).

## What is implemented
| Paper section | File |
|---|---|
| 3.2 Butterworth 25-400 Hz, z-score, 5 s windows | `features.py` |
| 3.3 27 features (9 stat, 2 entropy, 3 frequency, 13 MFCC) | `features.py` |
| 3.5 recording-level stratified 90/10 split + 10-fold grouped CV | `run_experiments.py` |
| 3.6 A-Test (k = 2..10, Eqs. 4-5) | `run_experiments.py` |
| 3.7 14 classifiers (Table 4) in a median-impute + z-score pipeline | `models.py` |
| 3.8 accuracy, sens, spec, prec, F1, ROC/PR-AUC, macro/weighted F1, MCC, kappa, Brier, mean±SD, 95% CI | `run_experiments.py` |
| Figs 2-6 analogues | `results/fig_*.png` |

Outputs per experiment: `*_cv.csv` (mean/SD/CI), `*_test.csv` (held-out), `exp3_all_27_atest.csv`.

## Choices the paper leaves open (my assumptions)
- MFCC: 26 Mel filters, 256-point FFT, DCT coefficients c0-c12 (matching MATLAB's default `mfcc()`).
  Set `skip_c0=True` in `mfcc_mean()` to drop c0 and keep c1-c13 instead.
- Sample entropy: m=2, r=0.2·std, computed at the full 2 kHz sample rate to match MATLAB.
  Set `decimate_to=500` in `sample_entropy()` for ~16× faster extraction at the cost of accuracy.
- PSD: Welch, 1024-point windows, restricted to ≤256 Hz; DFR = peak power / total power in band.
- Fine/Medium RBF use MATLAB's kernel scales (√P/4, √P) converted to gamma; polynomial SVMs use gamma="scale".
- Weighted KNN uses inverse-distance weights (MATLAB's default); AdaBoost uses depth-1 trees; RUSBoost needs `imbalanced-learn`.
- Final held-out numbers come from refitting each model on the full 90% pool.
- Cross-validation defaults to **recording-level grouped folds**; pass `--no-group` to use standard
  segment-level stratified CV (which matches MATLAB Classification Learner's default behaviour and
  produces numbers closest to the paper).

## Expect differences
Exact figures (e.g. KNN Fine 0.944 CV / 0.955 test) depend on the random split and on the choices above,
so small deviations from the paper are normal. Segment count should be close to 13,015 (9,857 / 3,158).
