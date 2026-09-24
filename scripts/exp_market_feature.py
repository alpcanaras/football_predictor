#!/usr/bin/env python3
"""
Experiment: does feeding the market INTO the model beat blending after it?

The shipped design predicts blind, then log-pools with the de-vigged odds
using one global weight — and that weight is fitted to zero, so the market
wins outright. A single weight cannot express "the market is wrong in THIS
situation", though. If the market probability is a feature the model trains
on, it can learn residual corrections.

Both variants are trained here with the same cutoff so the comparison is
clean (the shipped models saw this window, so they cannot be used).

RESULT (2,607 out-of-sample matches from 2026-06-01, run 2026-09-24):

    A: features only         1.0264
    B: features + market     1.0083
    C: market only           1.0080
    book (de-vigged)         1.0074
    blend A + book           1.0074   (fitted w_model = 0.00)

Three things follow, and they settle the question:

  * Market features close almost the whole gap (1.0264 -> 1.0083), but only
    by teaching the model to copy the market.
  * B is no better than C. The 45 handcrafted features add NOTHING once the
    market is available — the tree simply reads the odds.
  * Neither beats the raw book, and the difference is not significant
    (t = -0.56). A tree approximating a number you already have exactly is,
    at best, that number.

So the ceiling is not a modelling failure; the features are redundant with
the market. The model earns its keep exactly where there are no odds —
alone it scores 1.0264, far better than a coin — and in Toto, where the
opponent is the crowd. Re-run this before anyone proposes beating the book
again.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/alpcanaras/football_predictor')
from sklearn.metrics import log_loss
import xgboost as xgb
from scripts import config

CUT = '2026-06-01'
SEED = 0


def devig(df):
    inv = np.column_stack([1/df['OddsA'].to_numpy(float),
                           1/df['OddsD'].to_numpy(float),
                           1/df['OddsH'].to_numpy(float)])
    return inv / inv.sum(axis=1, keepdims=True)      # class order A, D, H


def main():
    df = pd.read_csv(config.PROCESSED_DATA_FILE, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    feats = [f for f in config.FEATURES_CORE if f in df.columns]
    need = feats + ['result_label', 'OddsH', 'OddsD', 'OddsA']
    df = df.dropna(subset=need)
    df = df.sort_values('Date')

    mk = devig(df)
    df = df.assign(mkt_A=mk[:, 0], mkt_D=mk[:, 1], mkt_H=mk[:, 2])
    # favourite-longshot shape and market confidence: cheap interactions a
    # single blend weight cannot express
    df['mkt_max'] = mk.max(axis=1)
    df['mkt_spread'] = mk.max(axis=1) - mk.min(axis=1)

    tr = df[df['Date'] < CUT]
    te = df[df['Date'] >= CUT]
    y_tr = tr['result_label'].to_numpy(int)
    y_te = te['result_label'].to_numpy(int)
    book_te = te[['mkt_A', 'mkt_D', 'mkt_H']].to_numpy()

    mkt_feats = ['mkt_A', 'mkt_D', 'mkt_H', 'mkt_max', 'mkt_spread']
    params = dict(objective='multi:softprob', num_class=3, max_depth=4,
                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=20, reg_lambda=2.0, n_estimators=400,
                  tree_method='hist', random_state=SEED, verbosity=0)

    print(f"train {len(tr):,} (<{CUT})   test {len(te):,} (>={CUT})\n")
    out = {}
    for name, cols in [('A: features only', feats),
                       ('B: features + market', feats + mkt_feats),
                       ('C: market only', mkt_feats)]:
        m = xgb.XGBClassifier(**params)
        m.fit(tr[cols], y_tr)
        p = m.predict_proba(te[cols])
        out[name] = p
        print(f"  {name:<24} logloss={log_loss(y_te, p, labels=[0,1,2]):.4f}  "
              f"acc={(p.argmax(1)==y_te).mean():.3f}")

    print(f"  {'book (de-vigged)':<24} logloss={log_loss(y_te, book_te, labels=[0,1,2]):.4f}  "
          f"acc={(book_te.argmax(1)==y_te).mean():.3f}")

    # the shipped approach: log-pool features-only with the book
    def pool(a, b, w):
        lg = w*np.log(np.clip(a,1e-9,1)) + (1-w)*np.log(np.clip(b,1e-9,1))
        lg -= lg.max(1, keepdims=True); p = np.exp(lg)
        return p / p.sum(1, keepdims=True)
    best = min(((log_loss(y_te, pool(out['A: features only'], book_te, w),
                          labels=[0,1,2]), w) for w in np.arange(0, 1.01, 0.05)))
    print(f"  {'blend A+book (best w)':<24} logloss={best[0]:.4f}  (w_model={best[1]:.2f})")

    # significance: B vs book, paired
    for label in ('B: features + market', 'C: market only'):
        a = -np.log(np.clip(book_te[np.arange(len(y_te)), y_te], 1e-12, None))
        b = -np.log(np.clip(out[label][np.arange(len(y_te)), y_te], 1e-12, None))
        d = a - b; se = d.std(ddof=1)/np.sqrt(len(d))
        print(f"\n  {label} vs book: {d.mean():+.4f} +- {1.96*se:.4f} "
              f"(t={d.mean()/se:+.2f})" +
              ("  SIGNIFICANT" if abs(d.mean()/se) > 2 else "  (not significant)"))


if __name__ == '__main__':
    main()
