"""
Fair Credit: Algorithmic Bias Auditing and Mitigation in Digital Micro-Lending
Trustworthy & Responsible AI (ADCOR1PC303) - Technical Micro-Project

Pipeline: synthetic applicant generator -> baseline credit model -> fairness audit
          -> mitigation (reweighing, group thresholds) -> profit/fairness trade-off.
Run:      python fair_credit_audit.py        (writes results.json, audit_log.txt, figs/)
"""
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------- configuration
N = 30000                    # synthetic applicants
LOAN = 10_000                # INR principal per loan
GAIN = 0.12 * LOAN           # net interest income when repaid
LOSS = 0.70 * LOAN           # loss given default (30 % recovery)
FEATURES = ["age", "log_income", "formal_job", "upi_txn", "bill_ontime",
            "recharge_reg", "log_device_value", "addr_stability", "contacts"]
PROTECTED = "female"         # used ONLY for auditing / mitigation, never as a model input
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


assert PROTECTED not in FEATURES, "protected attribute must never be a model input"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- data generation
def generate_applicants(n=N, seed=42):
    """Synthetic digital-lending applicants.
    True repayment depends ONLY on latent ability z (same distribution for all groups).
    Observed features carry structural group effects (income gap, informal work,
    cheaper devices, fewer contacts), so they act as proxies for gender / region."""
    r = np.random.default_rng(seed)
    female = r.binomial(1, 0.42, n)
    rural = r.binomial(1, 0.50, n)
    z = r.normal(0, 1, n)
    age = np.clip(r.normal(34, 9, n), 18, 65)
    income = np.exp(9.7 + 0.35 * z - 0.30 * female - 0.22 * rural + r.normal(0, 0.35, n))
    formal = r.binomial(1, sigmoid(-0.4 + 0.5 * z - 1.0 * female - 0.8 * rural))
    upi = r.poisson(np.exp(2.7 + 0.30 * z - 0.30 * female - 0.20 * rural))
    bill = np.clip(0.66 + 0.12 * z + r.normal(0, 0.12, n), 0, 1)
    recharge = np.clip(0.60 + 0.10 * z + r.normal(0, 0.14, n), 0, 1)
    device = np.exp(9.4 + 0.10 * z - 0.35 * female - 0.30 * rural + r.normal(0, 0.40, n))
    addr = np.clip(r.gamma(3.0, 9.0, n) - 7.0 * female + 4.0 * z, 1, None)
    contacts = r.poisson(np.exp(3.5 + 0.10 * z - 0.30 * female))
    repaid = r.binomial(1, sigmoid(1.9 + 1.3 * z))            # 1 = repays, 0 = defaults
    return pd.DataFrame({"age": age, "log_income": np.log(income), "formal_job": formal,
                         "upi_txn": upi, "bill_ontime": bill, "recharge_reg": recharge,
                         "log_device_value": np.log(device), "addr_stability": addr,
                         "contacts": contacts, "female": female, "rural": rural, "repaid": repaid})


# ---------------------------------------------------------------- fairness metrics
def profit(decision, y):
    """Portfolio profit: approved & repaid -> +GAIN ; approved & default -> -LOSS."""
    return float(np.sum(decision * (y * GAIN - (1 - y) * LOSS)))


def audit(decision, y, g, score=None):
    """Return the full fairness + business scorecard for one decision vector."""
    m, f = (g == 0), (g == 1)
    ar_m, ar_f = decision[m].mean(), decision[f].mean()
    tpr_m, tpr_f = decision[m & (y == 1)].mean(), decision[f & (y == 1)].mean()
    fpr_m, fpr_f = decision[m & (y == 0)].mean(), decision[f & (y == 0)].mean()
    out = {"approval_male": ar_m, "approval_female": ar_f,
           "DIR": ar_f / ar_m, "SPD": ar_f - ar_m,
           "TPR_male": tpr_m, "TPR_female": tpr_f, "EOD": tpr_f - tpr_m,
           "FPR_male": fpr_m, "FPR_female": fpr_f, "AOD": 0.5 * ((fpr_f - fpr_m) + (tpr_f - tpr_m)),
           "accuracy": accuracy_score(y, decision), "approved_repay_rate": y[decision == 1].mean(),
           "profit": profit(decision, y), "profit_per_applicant": profit(decision, y) / len(y),
           "approval_overall": decision.mean()}
    if score is not None:
        out["AUC"] = roc_auc_score(y, score)
    return out


def dir_bootstrap(decision, g, n_boot=500, seed=0):
    r = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = r.integers(0, len(g), len(g))
        d, gg = decision[i], g[i]
        vals.append(d[gg == 1].mean() / d[gg == 0].mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- mitigation
def reweighing_weights(g, y):
    """Kamiran & Calders (2012): w(g,y) = P(g) * P(y) / P(g,y)  -> makes g and y independent."""
    w = np.ones(len(y))
    for gv in (0, 1):
        for yv in (0, 1):
            mask = (g == gv) & (y == yv)
            w[mask] = (np.mean(g == gv) * np.mean(y == yv)) / max(mask.mean(), 1e-12)
    return w


def best_global_threshold(score, y, grid=np.linspace(0.05, 0.99, 95)):
    pr = [profit(score >= t, y) for t in grid]
    return float(grid[int(np.argmax(pr))])


def group_thresholds(score, y, g, mode, grid=np.linspace(0.05, 0.99, 95), eo_tol=0.02, dir_min=0.80):
    """Post-processing: pick (t_male, t_female) maximising validation profit subject to a fairness
    constraint ('eo' -> |TPR gap| <= eo_tol ; 'dir' -> DIR >= dir_min). Metrics are separable by group,
    so every pair on the grid is evaluated by broadcasting."""
    m, f = g == 0, g == 1
    pm = np.array([profit(score[m] >= t, y[m]) for t in grid])
    pf = np.array([profit(score[f] >= t, y[f]) for t in grid])
    arm = np.array([(score[m] >= t).mean() for t in grid])
    arf = np.array([(score[f] >= t).mean() for t in grid])
    tm = np.array([(score[m & (y == 1)] >= t).mean() for t in grid])
    tf = np.array([(score[f & (y == 1)] >= t).mean() for t in grid])
    tot = pm[:, None] + pf[None, :]
    if mode == "eo":
        ok = np.abs(tf[None, :] - tm[:, None]) <= eo_tol
    else:
        ok = (arf[None, :] / np.maximum(arm[:, None], 1e-9)) >= dir_min
    tot = np.where(ok, tot, -np.inf)
    if not np.isfinite(tot).any():          # guard: never fall back silently to threshold (0, 0)
        raise ValueError(f"No threshold pair satisfies the '{mode}' fairness constraint on the validation set")
    i, j = np.unravel_index(np.argmax(tot), tot.shape)
    return float(grid[i]), float(grid[j])


def apply_thresholds(score, g, t_m, t_f):
    return np.where(g == 1, score >= t_f, score >= t_m).astype(int)


# ---------------------------------------------------------------- one full experiment
def run_experiment(seed=42, make_extras=False):
    df = generate_applicants(seed=seed)
    strat = df["female"].astype(str) + df["repaid"].astype(str)
    tr, tmp = train_test_split(df, test_size=0.40, random_state=seed, stratify=strat)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed,
                              stratify=tmp["female"].astype(str) + tmp["repaid"].astype(str))
    sc = StandardScaler().fit(tr[FEATURES])
    X = {k: sc.transform(d[FEATURES]) for k, d in (("tr", tr), ("va", va), ("te", te))}
    y_tr, y_va, y_te = tr["repaid"].values, va["repaid"].values, te["repaid"].values
    g_tr, g_va, g_te = tr[PROTECTED].values, va[PROTECTED].values, te[PROTECTED].values

    res, decisions, extras = {}, {}, {}

    # M0: baseline - "fairness through unawareness" (gender excluded) + single global threshold
    base = LogisticRegression(max_iter=1000).fit(X["tr"], y_tr)
    s_va, s_te = base.predict_proba(X["va"])[:, 1], base.predict_proba(X["te"])[:, 1]
    t0 = best_global_threshold(s_va, y_va)
    decisions["Baseline LR"] = (s_te >= t0).astype(int)
    res["Baseline LR"] = audit(decisions["Baseline LR"], y_te, g_te, s_te); res["Baseline LR"]["thresholds"] = [t0, t0]

    # M0b: gradient-boosted baseline (non-linear comparator)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed).fit(X["tr"], y_tr)
    sg_va, sg_te = gb.predict_proba(X["va"])[:, 1], gb.predict_proba(X["te"])[:, 1]
    tg = best_global_threshold(sg_va, y_va)
    decisions["Baseline GBM"] = (sg_te >= tg).astype(int)
    res["Baseline GBM"] = audit(decisions["Baseline GBM"], y_te, g_te, sg_te); res["Baseline GBM"]["thresholds"] = [tg, tg]

    # M1: pre-processing - reweighing
    w = reweighing_weights(g_tr, y_tr)
    rw = LogisticRegression(max_iter=1000).fit(X["tr"], y_tr, sample_weight=w)
    r_va, r_te = rw.predict_proba(X["va"])[:, 1], rw.predict_proba(X["te"])[:, 1]
    t1 = best_global_threshold(r_va, y_va)
    decisions["Reweighing"] = (r_te >= t1).astype(int)
    res["Reweighing"] = audit(decisions["Reweighing"], y_te, g_te, r_te); res["Reweighing"]["thresholds"] = [t1, t1]

    # M2: post-processing - equal-opportunity group thresholds on baseline scores
    a, b = group_thresholds(s_va, y_va, g_va, "eo")
    decisions["EO thresholds"] = apply_thresholds(s_te, g_te, a, b)
    res["EO thresholds"] = audit(decisions["EO thresholds"], y_te, g_te, s_te); res["EO thresholds"]["thresholds"] = [a, b]

    # M2b: post-processing - minimum-intervention DIR >= 0.80 thresholds
    a, b = group_thresholds(s_va, y_va, g_va, "dir")
    decisions["DIR>=0.80 thresholds"] = apply_thresholds(s_te, g_te, a, b)
    res["DIR>=0.80 thresholds"] = audit(decisions["DIR>=0.80 thresholds"], y_te, g_te, s_te)
    res["DIR>=0.80 thresholds"]["thresholds"] = [a, b]

    # M3: combined - reweighing + EO thresholds
    a, b = group_thresholds(r_va, y_va, g_va, "eo")
    decisions["Reweigh + EO"] = apply_thresholds(r_te, g_te, a, b)
    res["Reweigh + EO"] = audit(decisions["Reweigh + EO"], y_te, g_te, r_te); res["Reweigh + EO"]["thresholds"] = [a, b]

    if make_extras:
        # proxy leakage: can gender be recovered from the 'gender-blind' features?
        Xa, Xb, ya, yb = train_test_split(X["tr"], g_tr, test_size=0.3, random_state=seed, stratify=g_tr)
        leak = LogisticRegression(max_iter=1000).fit(Xa, ya)
        extras["proxy_auc"] = float(roc_auc_score(yb, leak.predict_proba(Xb)[:, 1]))
        extras["feature_gender_corr"] = {f: float(np.corrcoef(tr[f], tr["female"])[0, 1]) for f in FEATURES}
        extras["feature_rural_corr"] = {f: float(np.corrcoef(tr[f], tr["rural"])[0, 1]) for f in FEATURES}
        extras["lr_coef"] = dict(zip(FEATURES, map(float, base.coef_[0])))
        pi = permutation_importance(base, X["te"], y_te, n_repeats=10, random_state=seed, scoring="roc_auc")
        extras["perm_importance"] = dict(zip(FEATURES, map(float, pi.importances_mean)))
        extras["calibration"] = {"male": {"mean_pred": float(s_te[g_te == 0].mean()), "actual": float(y_te[g_te == 0].mean())},
                                 "female": {"mean_pred": float(s_te[g_te == 1].mean()), "actual": float(y_te[g_te == 1].mean())}}
        extras["reweigh_weight_range"] = [float(w.min()), float(w.max())]
        extras["n"] = {"train": len(tr), "val": len(va), "test": len(te)}
        extras["base_rate_repay"] = {"male": float(y_te[g_te == 0].mean()), "female": float(y_te[g_te == 1].mean())}
        extras["female_share"] = float(g_te.mean())
        extras["dir_ci_baseline"] = dir_bootstrap(decisions["Baseline LR"], g_te)
        extras["dir_ci_eo"] = dir_bootstrap(decisions["EO thresholds"], g_te)
        extras["dir_ci_reweigh_eo"] = dir_bootstrap(decisions["Reweigh + EO"], g_te)
        extras["dir_ci_dir80"] = dir_bootstrap(decisions["DIR>=0.80 thresholds"], g_te)
        # intersectional approval rates (gender x region), baseline vs EO thresholds
        inter = {}
        for name in ("Baseline LR", "EO thresholds", "Reweigh + EO"):
            d = decisions[name]
            inter[name] = {f"{'F' if gg else 'M'}-{'Rural' if rr else 'Urban'}":
                           float(d[(g_te == gg) & (te['rural'].values == rr)].mean()) for gg in (0, 1) for rr in (0, 1)}
        extras["intersectional"] = inter
        # trade-off frontier: sweep the female threshold, hold male threshold at baseline optimum
        sweep = []
        for tf_ in np.linspace(0.40, 0.99, 60):
            d = apply_thresholds(s_te, g_te, t0, tf_); au = audit(d, y_te, g_te)
            sweep.append({"t_f": float(tf_), "DIR": au["DIR"], "EOD": au["EOD"], "profit": au["profit"]})
        extras["frontier"] = sweep
        # robustness stress test: Gaussian noise injected into standardised test features (thresholds frozen)
        rn = np.random.default_rng(seed + 99); stress = []
        tb, te_ = res["Baseline LR"]["thresholds"][0], res["EO thresholds"]["thresholds"]
        for sig in (0.0, 0.25, 0.5, 1.0):
            sn = base.predict_proba(X["te"] + rn.normal(0, sig, X["te"].shape))[:, 1]
            ab = audit((sn >= tb).astype(int), y_te, g_te, sn)
            ae = audit(apply_thresholds(sn, g_te, te_[0], te_[1]), y_te, g_te, sn)
            stress.append({"sigma": sig, "AUC": ab["AUC"], "DIR_base": ab["DIR"], "DIR_eo": ae["DIR"],
                           "EOD_base": ab["EOD"], "EOD_eo": ae["EOD"], "profit_base": ab["profit_per_applicant"],
                           "profit_eo": ae["profit_per_applicant"]})
        extras["stress"] = stress
        extras["scores_te"] = (s_te, y_te, g_te)
        extras["data_summary"] = df.groupby("female")[FEATURES + ["repaid"]].mean().round(3).to_dict()
    return res, extras


# ---------------------------------------------------------------- figures
def make_figures(res, ex):
    order = ["Baseline LR", "Baseline GBM", "Reweighing", "EO thresholds", "DIR>=0.80 thresholds", "Reweigh + EO"]
    labels = ["Baseline\nLR", "Baseline\nGBM", "Reweighing", "EO\nthresholds", "DIR≥0.80\nthresholds", "Reweigh\n+ EO"]
    col = ["#7f7f7f", "#a6a6a6", "#4c78a8", "#2e8b57", "#8c6bb1", "#e07b39"]
    # Fig 5.1 approval rate by group
    fig, ax = plt.subplots(figsize=(6.4, 3.3)); xx = np.arange(len(order)); wd = 0.38
    ax.bar(xx - wd / 2, [res[k]["approval_male"] * 100 for k in order], wd, label="Male", color="#4c78a8")
    ax.bar(xx + wd / 2, [res[k]["approval_female"] * 100 for k in order], wd, label="Female", color="#e07b39")
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("Approval rate (%)"); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig("figs/fig_approval.png", dpi=200); plt.close()
    # Fig 5.2 DIR and TPR gap
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2))
    axs[0].bar(xx, [res[k]["DIR"] for k in order], color=col); axs[0].axhline(0.8, color="red", ls="--", lw=1)
    axs[0].text(len(order) - 0.5, 0.82, "0.80 threshold", color="red", ha="right", fontsize=8)
    axs[0].set_xticks(xx); axs[0].set_xticklabels(labels, fontsize=6.5); axs[0].set_ylabel("Disparate impact ratio"); axs[0].set_ylim(0, 1.15)
    axs[1].bar(xx, [abs(res[k]["EOD"]) for k in order], color=col)
    axs[1].set_xticks(xx); axs[1].set_xticklabels(labels, fontsize=6.5); axs[1].set_ylabel("|TPR gap| (equal opportunity)")
    plt.tight_layout(); plt.savefig("figs/fig_dir_eod.png", dpi=200); plt.close()
    # Fig 5.3 profit vs DIR
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    fr = ex["frontier"]
    ax.plot([p["DIR"] for p in fr], [p["profit"] / 1e5 for p in fr], color="#999999", lw=1.2, label="Female-threshold sweep")
    for k, c, lb in zip(order, col, ["Baseline LR", "Baseline GBM", "Reweighing", "EO thresholds", "DIR≥0.80", "Reweigh+EO"]):
        ax.scatter(res[k]["DIR"], res[k]["profit"] / 1e5, s=55, color=c, zorder=3, label=lb)
    ax.axvline(0.8, color="red", ls="--", lw=1); ax.set_xlabel("Disparate impact ratio (female / male)")
    ax.set_ylabel("Test-set portfolio profit (Rs. lakh)"); ax.legend(frameon=False, fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig("figs/fig_tradeoff.png", dpi=200); plt.close()
    # Fig 4.x proxy correlations
    fig, ax = plt.subplots(figsize=(5.8, 3.3)); fn = list(ex["feature_gender_corr"].keys())
    cg = [ex["feature_gender_corr"][f] for f in fn]; cr = [ex["feature_rural_corr"][f] for f in fn]; yy = np.arange(len(fn))
    ax.barh(yy - 0.2, cg, 0.4, color="#e07b39", label="corr. with female"); ax.barh(yy + 0.2, cr, 0.4, color="#4c78a8", label="corr. with rural")
    ax.set_yticks(yy); ax.set_yticklabels(fn, fontsize=8); ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("Pearson correlation"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig("figs/fig_proxy.png", dpi=200); plt.close()
    # Fig score distribution
    s, y, g = ex["scores_te"]; fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.hist(s[g == 0], bins=40, alpha=0.6, density=True, color="#4c78a8", label="Male"); ax.hist(s[g == 1], bins=40, alpha=0.6, density=True, color="#e07b39", label="Female")
    ax.axvline(res["Baseline LR"]["thresholds"][0], color="k", ls="--", lw=1); ax.set_xlabel("Predicted repayment probability (baseline LR)"); ax.set_ylabel("Density"); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig("figs/fig_scores.png", dpi=200); plt.close()
    # Fig 3.1 pipeline
    fig, ax = plt.subplots(figsize=(6.6, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    def box(x, y, w, h, t, c):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="#333333", lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.8)
    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1))
    box(0.2, 7.6, 2.6, 1.6, "Data layer\nSynthetic applicant\ngenerator (30,000)", "#dbe9f6"); box(3.7, 7.6, 2.6, 1.6, "Preprocessing\nlog-transform,\nscaling, 60/20/20 split", "#dbe9f6")
    box(7.2, 7.6, 2.6, 1.6, "Baseline models\nLR + GBM\n(gender excluded)", "#e8e8e8"); arr(2.8, 8.4, 3.7, 8.4); arr(6.3, 8.4, 7.2, 8.4)
    box(7.2, 4.6, 2.6, 1.8, "Bias audit module\nDIR, SPD, EOD, AOD,\nFPR gap, bootstrap CI", "#fde5d3"); box(3.8, 4.6, 2.8, 1.8, "Proxy analysis\ncorrelation, gender-\nleakage AUC, perm. imp.", "#fde5d3")
    box(0.2, 4.6, 3.0, 1.8, "Mitigation layer\nReweighing (pre)\nGroup thresholds (post)", "#d6ecd9"); arr(8.5, 7.6, 8.5, 6.4); arr(7.2, 5.5, 6.6, 5.5); arr(3.8, 5.5, 3.2, 5.5)
    box(0.2, 1.6, 3.0, 1.8, "Business evaluation\nprofit / loan, default\nrate in approved book", "#f6e8b1"); box(3.8, 1.6, 2.8, 1.8, "Governance layer\nDPDP / RBI / EU AI Act\ncompliance + model card", "#f6e8b1")
    box(7.2, 1.6, 2.6, 1.8, "Output\naudit_log.txt,\nresults.json, figures", "#dbe9f6"); arr(1.7, 4.6, 1.7, 3.4); arr(5.2, 4.6, 5.2, 3.4); arr(3.2, 2.5, 3.8, 2.5); arr(6.6, 2.5, 7.2, 2.5)
    ax.text(5, 0.5, "Human-in-the-loop review: loan officer overrides and adverse-action reasons are logged", ha="center", fontsize=7.5, style="italic")
    plt.tight_layout(); plt.savefig("figs/fig_pipeline.png", dpi=200); plt.close()


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    log = []
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); log.append(s)
    P("=" * 78); P("FAIR CREDIT AUDIT - digital micro-lending (synthetic data, seed 42)"); P("=" * 78)
    res, ex = run_experiment(42, make_extras=True)
    P(f"Applicants: {N}  | split train/val/test = {ex['n']['train']}/{ex['n']['val']}/{ex['n']['test']}")
    P(f"Female share (test): {ex['female_share']:.3f} | true repay rate  M: {ex['base_rate_repay']['male']:.3f}  F: {ex['base_rate_repay']['female']:.3f}")
    P(f"Proxy leakage: gender recoverable from gender-blind features, AUC = {ex['proxy_auc']:.3f}")
    P("-" * 78)
    P(f"{'Method':<22}{'AR_M':>7}{'AR_F':>7}{'DIR':>7}{'SPD':>8}{'TPR_M':>7}{'TPR_F':>7}{'EOD':>8}{'AUC':>7}{'Profit/app':>12}")
    for k, v in res.items():
        P(f"{k:<22}{v['approval_male']:>7.3f}{v['approval_female']:>7.3f}{v['DIR']:>7.3f}{v['SPD']:>8.3f}"
          f"{v['TPR_male']:>7.3f}{v['TPR_female']:>7.3f}{v['EOD']:>8.3f}{v.get('AUC', float('nan')):>7.3f}{v['profit_per_applicant']:>12.1f}")
    P("-" * 78)
    for k in ("Baseline LR", "DIR>=0.80 thresholds", "EO thresholds", "Reweigh + EO"):
        P(f"Thresholds {k:<22} male={res[k]['thresholds'][0]:.2f} female={res[k]['thresholds'][1]:.2f}")
    P("Bootstrap 95% CI of DIR (500 resamples):")
    for k, n_ in (("dir_ci_baseline", "Baseline LR"), ("dir_ci_dir80", "DIR>=0.80 thresholds"), ("dir_ci_eo", "EO thresholds"), ("dir_ci_reweigh_eo", "Reweigh + EO")):
        P(f"  {n_:<22} [{ex[k][0]:.3f}, {ex[k][1]:.3f}]")
    P(f"Calibration (mean predicted vs actual repay rate): male {ex['calibration']['male']['mean_pred']:.3f}/{ex['calibration']['male']['actual']:.3f} | "
      f"female {ex['calibration']['female']['mean_pred']:.3f}/{ex['calibration']['female']['actual']:.3f}")
    P(f"Reweighing weight range: {ex['reweigh_weight_range'][0]:.3f} - {ex['reweigh_weight_range'][1]:.3f}")
    P("Noise stress test (sigma in standardised units; thresholds frozen):")
    for r_ in ex["stress"]:
        P(f"  sigma={r_['sigma']:.2f} AUC={r_['AUC']:.3f} | DIR base {r_['DIR_base']:.3f} -> EO {r_['DIR_eo']:.3f} | "
          f"EOD base {r_['EOD_base']:+.3f} -> EO {r_['EOD_eo']:+.3f} | profit/app base {r_['profit_base']:.0f} -> EO {r_['profit_eo']:.0f}")
    P("Intersectional approval rates:")
    for k, v in ex["intersectional"].items():
        P(f"  {k:<16}", {a: round(b, 3) for a, b in v.items()})
    P("Permutation importance (baseline LR, drop in AUC):")
    for f, v in sorted(ex["perm_importance"].items(), key=lambda t: -t[1]):
        P(f"  {f:<18}{v:.4f}   coef={ex['lr_coef'][f]:+.3f}   corr(female)={ex['feature_gender_corr'][f]:+.3f}")
    # robustness over 10 data-generation seeds
    P("-" * 78); P("Robustness: 10 independent synthetic datasets (seeds 1-10), mean +/- std")
    keys = ["Baseline LR", "Reweighing", "EO thresholds", "DIR>=0.80 thresholds", "Reweigh + EO"]
    agg = {k: {"DIR": [], "EOD": [], "AUC": [], "profit_per_applicant": []} for k in keys}
    for sd in range(1, 11):
        rr, _ = run_experiment(sd)
        for k in keys:
            for m_ in agg[k]:
                agg[k][m_].append(rr[k][m_])
    robust = {}
    for k in keys:
        robust[k] = {m_: [float(np.mean(v)), float(np.std(v))] for m_, v in agg[k].items()}
        robust[k]["DIR_min"] = float(np.min(agg[k]["DIR"])); robust[k]["seeds_DIR_ge_0.80"] = int(np.sum(np.array(agg[k]["DIR"]) >= 0.80))
        P(f"{k:<22} (DIR>=0.80 in {robust[k]['seeds_DIR_ge_0.80']}/10 seeds, min {robust[k]['DIR_min']:.3f}) DIR {robust[k]['DIR'][0]:.3f}+/-{robust[k]['DIR'][1]:.3f} | EOD {robust[k]['EOD'][0]:+.3f}+/-{robust[k]['EOD'][1]:.3f} | "
          f"AUC {robust[k]['AUC'][0]:.3f}+/-{robust[k]['AUC'][1]:.3f} | profit/app {robust[k]['profit_per_applicant'][0]:.0f}+/-{robust[k]['profit_per_applicant'][1]:.0f}")
    make_figures(res, ex)
    ex.pop("scores_te")
    json.dump({"results": res, "extras": ex, "robust": robust}, open("results.json", "w"), indent=1)
    open("audit_log.txt", "w").write("\n".join(log))
    P("Saved results.json, audit_log.txt, figs/*.png")
