"""Experiments 1-3, recording-level 90/10 split, 10-fold grouped CV, A-Test, metrics, figures.

Usage
  # 1) extract features from the raw PhysioNet data (slow, run once)
  python run_experiments.py extract --data /path/to/challenge-2016/training --out FeatureTable.csv
  # 2) run the three experiments
  python run_experiments.py run --features FeatureTable.csv --out results
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     GroupShuffleSplit)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             cohen_kappa_score, brier_score_loss, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.utils import check_random_state

import features as F
from models import make_models, scores

warnings.filterwarnings("ignore")
SEED = 42
EXPERIMENTS = {"exp1_stat_entropy_freq": F.EXP1_FEATURES,
               "exp2_mfcc_only": F.EXP2_FEATURES,
               "exp3_all_27": F.EXP3_FEATURES}


# ----------------------------------------------------------------- metrics
def minmax_scale(s):
    s = np.asarray(s, float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else np.zeros_like(s)


def compute_metrics(y, pred, score):
    """Abnormal (1) is the positive class. Eqs. (6)-(10) plus threshold-free metrics."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(
        acc=accuracy_score(y, pred),
        sens=recall_score(y, pred, zero_division=0),
        spec=tn / (tn + fp) if (tn + fp) else np.nan,
        prec=precision_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y, pred, average="weighted", zero_division=0),
        mcc=matthews_corrcoef(y, pred),
        kappa=cohen_kappa_score(y, pred),
    )
    if len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, score)
        out["pr_auc"] = average_precision_score(y, score)
        out["brier"] = brier_score_loss(y, minmax_scale(score))
    else:
        out["roc_auc"] = out["pr_auc"] = out["brier"] = np.nan
    return out


def mean_sd_ci(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    m, sd = vals.mean(), vals.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return m, sd, m - h, m + h


# ----------------------------------------------------------------- splitting
def recording_level_split(df, test_size=0.1, seed=SEED):
    """Stratified 90/10 split at the RECORDING level (all segments of a recording stay together)."""
    rec = df.groupby("recording").agg(label=("label", "first")).reset_index()
    rng = check_random_state(seed)
    test = []
    for lab, g in rec.groupby("label"):
        ids = g.recording.values.copy()
        rng.shuffle(ids)
        test.extend(ids[:max(1, int(round(len(ids) * test_size)))])
    is_test = df.recording.isin(set(test))
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def grouped_cv(model, X, y, groups, k, seed=SEED, use_groups=True):
    """k-fold stratified CV.

    If use_groups=True (default), whole recordings are assigned to a single fold
    (StratifiedGroupKFold).  Set use_groups=False for standard segment-level
    StratifiedKFold, which matches MATLAB Classification Learner's default
    behaviour and eliminates the recording-level leakage gap.
    """
    if use_groups:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)
    rows = []
    for tr, va in splits:
        m = clone(model).fit(X[tr], y[tr])
        rows.append(compute_metrics(y[va], m.predict(X[va]), scores(m, X[va])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- one experiment
def run_experiment(name, cols, train, test, outdir, do_atest=False, only=None,
                   k_folds=10, use_groups=True):
    Xtr, ytr, gtr = train[cols].values, train.label.values, train.recording.values
    Xte, yte = test[cols].values, test.label.values
    models = make_models(len(cols), SEED)
    if only:
        models = {k: v for k, v in models.items() if k in only}

    cv_rows, test_rows, atest_rows, fitted = [], [], [], {}
    for mname, model in models.items():
        print(f"[{name}] {mname}", flush=True)
        folds = grouped_cv(model, Xtr, ytr, gtr, k_folds, use_groups=use_groups)
        row = {"model": mname}
        for met in folds.columns:
            m, sd, lo, hi = mean_sd_ci(folds[met])
            row.update({f"{met}_mean": m, f"{met}_sd": sd, f"{met}_ci_lo": lo, f"{met}_ci_hi": hi})
        cv_rows.append(row)

        final = clone(model).fit(Xtr, ytr)                      # refit on the whole train/val pool
        pred, sc = final.predict(Xte), scores(final, Xte)
        test_rows.append({"model": mname, **compute_metrics(yte, pred, sc)})
        fitted[mname] = (pred, sc)

        if do_atest:                                            # A-Test, k = 2..10 (Eqs. 4-5)
            accs = [grouped_cv(model, Xtr, ytr, gtr, k,
                              use_groups=use_groups)["acc"].mean()
                    for k in range(2, 11)]
            atest_rows.append({"model": mname, "mean_acc": np.mean(accs), "sd": np.std(accs, ddof=1),
                               "min_acc": np.min(accs), "max_acc": np.max(accs),
                               "range": np.ptp(accs), "mean_error": np.mean(1 - np.array(accs))})

    cv = pd.DataFrame(cv_rows).sort_values("acc_mean", ascending=False)
    te = pd.DataFrame(test_rows).set_index("model").loc[cv.model].reset_index()
    cv.to_csv(os.path.join(outdir, f"{name}_cv.csv"), index=False)
    te.to_csv(os.path.join(outdir, f"{name}_test.csv"), index=False)
    if do_atest:
        pd.DataFrame(atest_rows).sort_values("mean_acc", ascending=False).to_csv(
            os.path.join(outdir, f"{name}_atest.csv"), index=False)

    summary = cv[["model", "acc_mean", "acc_sd", "weighted_f1_mean", "f1_mean", "roc_auc_mean"]].merge(
        te[["model", "acc", "weighted_f1"]].rename(
            columns={"acc": "test_acc", "weighted_f1": "test_wf1"}), on="model")
    print(f"\n=== {name} ===\n{summary.round(3).to_string(index=False)}\n")
    return fitted, final_importance(models, Xtr, ytr, cols), summary


def final_importance(models, X, y, cols):
    """Random-forest impurity importance (Fig. 5 analogue)."""
    if "RF Bagged" not in models:
        return None
    m = clone(models["RF Bagged"]).fit(X, y)
    imp = m.named_steps["clf"].feature_importances_
    return pd.Series(imp, index=cols).sort_values(ascending=False)


# ----------------------------------------------------------------- figures (Exp. 3)
def make_figures(fitted, yte, importance, corr_df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(fitted)
    n = len(names)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for ax, nm in zip(np.ravel(axes), names):
        cm = confusion_matrix(yte, fitted[nm][0], labels=[0, 1])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")
        ax.set_xticks([0, 1], ["Normal", "Abnormal"]); ax.set_yticks([0, 1], ["Normal", "Abnormal"])
        ax.set_title(f"{nm}\nAcc={accuracy_score(yte, fitted[nm][0]):.3f}", fontsize=9)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Exp. 3 confusion matrices (held-out test)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_confusion.png"), dpi=130); plt.close(fig)

    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(7, 6))
        for nm in names:
            sc = fitted[nm][1]
            if kind == "roc":
                fpr, tpr, _ = roc_curve(yte, sc); ax.plot(fpr, tpr, label=f"{nm} ({roc_auc_score(yte, sc):.3f})")
            else:
                p, r, _ = precision_recall_curve(yte, sc); ax.plot(r, p, label=f"{nm} ({average_precision_score(yte, sc):.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        else:
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"Exp. 3 {kind.upper()} curves (test set)"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"fig_{kind}.png"), dpi=130); plt.close(fig)

    if importance is not None:
        fig, ax = plt.subplots(figsize=(7, 8))
        importance[::-1].plot.barh(ax=ax); ax.set_xlabel("Importance (RF impurity)")
        ax.set_title("Exp. 3 feature importance"); fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_importance.png"), dpi=130); plt.close(fig)

    c = corr_df.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=6); ax.set_yticks(range(len(c)), c.columns, fontsize=6)
    fig.colorbar(im); ax.set_title("Pearson correlation (27 features)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_correlation.png"), dpi=130); plt.close(fig)


# ----------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data", required=True, help="folder containing training-a ... training-f")
    e.add_argument("--out", default="FeatureTable.csv")
    e.add_argument("--jobs", type=int, default=-1)
    e.add_argument("--limit", type=int, default=None, help="random subset of recordings (debug)")
    r = sub.add_parser("run")
    r.add_argument("--features", default="FeatureTable.csv")
    r.add_argument("--out", default="results")
    r.add_argument("--models", nargs="*", default=None, help="subset of model names (debug)")
    r.add_argument("--folds", type=int, default=10)
    r.add_argument("--no-atest", action="store_true")
    r.add_argument("--no-group", action="store_true",
                   help="segment-level CV (MATLAB default) instead of recording-grouped CV")
    a = ap.parse_args()

    if a.cmd == "extract":
        df = F.build_feature_table(a.data, a.out, a.jobs, a.limit)
        print(f"{len(df)} segments from {df.recording.nunique()} recordings "
              f"({(df.label == 0).sum()} normal / {(df.label == 1).sum()} abnormal) -> {a.out}")
        return

    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.features)
    train, test = recording_level_split(df)
    print(f"segments: train/val {len(train)}, test {len(test)}; "
          f"recordings: {train.recording.nunique()} / {test.recording.nunique()}")
    assert not set(train.recording) & set(test.recording)
    for name, cols in EXPERIMENTS.items():
        fitted, imp, _ = run_experiment(name, cols, train, test, a.out,
                                        do_atest=(name == "exp3_all_27" and not a.no_atest),
                                        only=a.models, k_folds=a.folds,
                                        use_groups=not a.no_group)
        if name == "exp3_all_27":
            make_figures(fitted, test.label.values, imp, df[F.EXP3_FEATURES], a.out)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
