"""Statistics behind the dashboard. Shared by the app, prepare_data.py and verify.py.

Core model (capability decomposition):
    capability = b0 + bC * log10(training compute) + bT * (release year - base year)
bC = gain from 10x more compute, bT = gain from one more year of better methods
at the SAME compute. Their ratio gives an "exchange rate":
    one year of algorithmic progress  ==  10 ** (bT / bC)  times more compute.
For MMLU we fit on the logit of the above-chance score, because MMLU is a bounded
multiple-choice test (25% = guessing, 100% = ceiling).
"""
import numpy as np
import pandas as pd

CHANCE = 0.25
N_BOOT = 1000


def frac_year(dates):
    d = pd.to_datetime(pd.Series(dates))
    return (d.dt.year + (d.dt.dayofyear - 1) / 365.25).to_numpy()


# ---------------------------------------------------------------- compute trend
def fit_trend(years, compute, n_boot=N_BOOT, seed=0):
    """Least-squares line on log10(compute) vs year, with a bootstrap 95% CI."""
    years, logc = np.asarray(years, float), np.log10(np.asarray(compute, float))
    slope, intercept = np.polyfit(years, logc, 1)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(years), len(years))
        boots.append(np.polyfit(years[i], logc[i], 1))
    boots = np.array(boots)
    return dict(slope=slope, intercept=intercept,
                growth=10 ** slope,
                growth_ci=10 ** np.percentile(boots[:, 0], [2.5, 97.5]),
                doubling=12 * np.log10(2) / slope,
                doubling_ci=12 * np.log10(2) / np.percentile(boots[:, 0], [97.5, 2.5]),
                boots=boots)


def find_breakpoint(years, compute, lo=2012, hi=2022, step=0.25):
    """Year that best splits the data into two straight-line eras (least total squared error)."""
    x, y = np.asarray(years, float), np.log10(np.asarray(compute, float))
    best = None
    for b in np.arange(lo, hi + 1e-9, step):
        before, after = x < b, x >= b
        if before.sum() < 10 or after.sum() < 10:
            continue
        fits = [np.polyfit(x[m], y[m], 1) for m in (before, after)]
        sse = sum(((y[m] - np.polyval(f, x[m])) ** 2).sum() for m, f in zip((before, after), fits))
        if best is None or sse < best[0]:
            best = (sse, b, fits)
    if best is None:
        return None
    _, b, (f1, f2) = best
    return dict(year=b, before=f1, after=f2,
                doubling_before=12 * np.log10(2) / f1[0], doubling_after=12 * np.log10(2) / f2[0],
                growth_before=10 ** f1[0], growth_after=10 ** f2[0])


# ---------------------------------------------------------------- capability model
def to_link(score, kind):
    if kind == "mmlu":
        p = np.clip((np.asarray(score, float) - CHANCE) / (1 - CHANCE), 0.01, 0.99)
        return np.log(p / (1 - p))
    return np.asarray(score, float)


def from_link(z, kind):
    if kind == "mmlu":
        return CHANCE + (1 - CHANCE) / (1 + np.exp(-np.asarray(z, float)))
    return np.asarray(z, float)


def fit_capability(score, compute, years, kind, base_year, n_boot=N_BOOT, seed=0):
    y = to_link(score, kind)
    X = np.column_stack([np.ones(len(y)), np.log10(compute), np.asarray(years) - base_year])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    b_c = np.linalg.lstsq(X[:, :2], y, rcond=None)[0]
    r2_compute = 1 - ((y - X[:, :2] @ b_c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        b = np.linalg.lstsq(X[i], y[i], rcond=None)[0]
        rates.append(b[2] / b[1])
    rates = np.array(rates)
    return dict(beta=beta, kind=kind, base_year=base_year, n=len(y), r2=r2, r2_compute=r2_compute,
                rate=10 ** (beta[2] / beta[1]),
                rate_ci=10 ** np.percentile(rates, [2.5, 97.5]))


def predict(fit, compute, years):
    b0, bc, bt = fit["beta"]
    z = b0 + bc * np.log10(compute) + bt * (np.asarray(years) - fit["base_year"])
    return from_link(z, fit["kind"])


def algo_share(rate, compute_growth):
    """Share of yearly progress (in log-compute-equivalent units) that comes from algorithms."""
    return np.log(rate) / (np.log(rate) + np.log(compute_growth))


def decompose(fit, c_a, t_a, c_b, t_b, s_a, s_b):
    """Split the score change A -> B into compute, algorithms and model-specific parts."""
    p_a = predict(fit, c_a, t_a)
    p_mid = predict(fit, c_b, t_a)        # B's compute, A's date
    p_b = predict(fit, c_b, t_b)
    compute_part = float(p_mid - p_a)
    algo_part = float(p_b - p_mid)
    other = float((s_b - s_a) - (compute_part + algo_part))
    return compute_part, algo_part, other


# ---------------------------------------------------------------- falling cost
def cheapest_path(df, threshold, score_col="score", compute_col="compute", date_col="release"):
    """Each time a model reaches the threshold with less compute than any earlier model."""
    rows, best = [], np.inf
    for _, r in df.sort_values(date_col).iterrows():
        if r[score_col] >= threshold and r[compute_col] < best:
            best = r[compute_col]
            rows.append(r)
    return pd.DataFrame(rows)
