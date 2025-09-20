import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ======================== パラメータ ========================
UNIVERSE = [
    # 自分の50銘柄に置き換えてください（米大型例）
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO","JPM","BAC","WFC",
    "V","MA","HD","PG","KO","PEP","MRK","PFE","JNJ","LLY","XOM","CVX",
    "COP","VZ","T","CSCO","ORCL","ADBE","CRM","INTC","QCOM","TXN","AMD",
    "UNH","ABBV","MCD","NFLX","COST","NKE","LOW","UPS","CAT","HON","PM",
    "MO","BKNG","AXP","BA","GE"
][:50]  # だいたい50銘柄に

START = "2005-01-01"   # データ取得開始（十分長めに）
END   = "2025-12-31"   # データ取得終了
WINDOW_MONTHS = 6      # ← ここを 2 や 3 に変えるだけでOK（直近Wか月で学習、次のWか月運用）
K_STOCKS = 10           # ポートフォリオの銘柄数
ALLOW_SHORT = False    # 最小分散のウェイトで空売りを許すか（ここでは基本 False）
RET_COL = "Adj Close"  # リターン計算に使う列

# 取引コスト・現実的制約は簡易化して無視（必要ならあとで追加）
# =========================================================

# ---------- ユーティリティ ----------
def fetch_prices(tickers, start, end):
    """
    各銘柄を個別にダウンロードして結合する安全版。
    RET_COL（例: 'Adj Close'）だけを横結合します。
    """
    import yfinance as yf
    import pandas as pd

    frames = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        except Exception as e:
            print(f"[WARN] download failed: {t}: {e}")
            continue
        if df is None or df.empty or RET_COL not in df.columns:
            print(f"[SKIP] no data or missing column '{RET_COL}': {t}")
            continue
        s = df[RET_COL].dropna().copy()
        s.name = t   # ← ここがポイント（renameではなく直接 name を設定）
        frames[t] = s

    if not frames:
        raise RuntimeError("価格データを取得できませんでした。ティッカーや期間を確認してください。")

    out = pd.concat(frames.values(), axis=1).sort_index()
    out = out.dropna(how="all")
    return out



def to_month_end(d):
    """その日の属する月の末営業日（実務近似として月末日→次に一番近い取引日を使う）"""
    # ここでは単純に月末日でOK（営業日ずれは小さな差）
    last = (d + relativedelta(day=31)).date()
    return pd.Timestamp(last)

def business_days(df):
    return df.index

def daily_returns(price_df):
    return price_df.pct_change().dropna(how="all")

def minvar_weights_longonly(cov_sub, tol=1e-9):
    """
    長期（非負・合計1）の最小分散ウェイトを近似計算。
    手順：制約なし解 → 負の重みを0に固定 → 残りで再計算 を繰り返し。
    制約なしの最小分散（合計1）解は w = Σ^{-1}1 /(1'Σ^{-1}1)
    """
    n = cov_sub.shape[0]
    idx = np.arange(n)
    active = np.ones(n, dtype=bool)  # True:自由、False:固定0
    one = np.ones((n,1))
    S = cov_sub.values

    # 数値安定化（対角に微小値を足す）
    S = S + np.eye(n)*1e-8

    while True:
        A = S[np.ix_(active, active)]
        o = one[active]
        try:
            invA = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # 予備：pinv
            invA = np.linalg.pinv(A)
        w_active = (invA @ o) / (o.T @ invA @ o)
        w = np.zeros((n,1))
        w[active] = w_active
        # 負の重みがあれば固定0にして繰り返し
        if (w < -tol).any():
            bad = (w[:,0] < -tol) & active
            if bad.sum()==0 or bad.sum()==active.sum():
                # すべて固定になるのはおかしいので安全弁
                w = np.clip(w,0,None)
                w = w / w.sum()
                break
            active[bad] = False
            continue
        # わずかに負ならゼロに丸めて正規化
        w = np.clip(w,0,None)
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w)/n
        else:
            w /= s
        return pd.Series(w[:,0], index=cov_sub.index)

def minvar_weights_unconstrained(cov_sub):
    """空売り許容（合計1のみ）最小分散の閉形式解"""
    S = cov_sub.values + np.eye(cov_sub.shape[0])*1e-8
    invS = np.linalg.inv(S)
    ones = np.ones((cov_sub.shape[0],1))
    w = (invS @ ones) / (ones.T @ invS @ ones)
    return pd.Series(w[:,0], index=cov_sub.index)

def greedy_pick_5(cov):
    """
    '5銘柄に制限'という離散制約は本来MIQPだが、ここでは
    Greedy forward selection：
    - 空集合から開始
    - 毎ステップ、候補を1つ足して最小分散（長期/短期設定に応じた重み）分散が最小になる銘柄を追加
    - 5銘柄に到達したら終了
    """
    chosen = []
    remaining = list(cov.index)
    for _ in range(K_STOCKS):
        best_t = None
        best_var = None
        for t in remaining:
            sub = cov.loc[chosen+[t], chosen+[t]]
            if ALLOW_SHORT:
                w = minvar_weights_unconstrained(sub)
            else:
                w = minvar_weights_longonly(sub)
            var = float(w.values @ (sub.values @ w.values))
            if (best_var is None) or (var < best_var):
                best_var = var
                best_t = t
        chosen.append(best_t)
        remaining.remove(best_t)
    sub = cov.loc[chosen, chosen]
    w = (minvar_weights_unconstrained(sub) if ALLOW_SHORT else minvar_weights_longonly(sub))
    return w.sort_index()

# ---------- データ取得 ----------
prices = fetch_prices(UNIVERSE, START, END)    # 終値系
rets = daily_returns(prices)                   # 日次リターン

# ---------- ローリング期間の境界を作成 ----------
start_dt = pd.to_datetime(START)
end_dt   = pd.to_datetime(END)
periods = []
cur = to_month_end(start_dt)
while True:
    train_end = cur
    train_start = train_end - relativedelta(months=WINDOW_MONTHS) + pd.Timedelta(days=1)
    test_start = train_end + pd.Timedelta(days=1)
    test_end   = test_start + relativedelta(months=WINDOW_MONTHS) - pd.Timedelta(days=1)

    if test_end > end_dt:  # 最後は切り捨て
        break
    periods.append((train_start, train_end, test_start, test_end))
    cur = test_end  # 次は直前テスト末を基準に“非重複”で進める（Wヶ月学習→Wヶ月運用）

# ---------- バックテスト ----------
equity = []
weights_log = []  # 各期間の採用ウェイト
cash = 1_000_000.0  # 初期資産（相対でもOKだがわかりやすく金額）
nav = cash

for i,(tr_s,tr_e,te_s,te_e) in enumerate(periods, start=1):
    # 学習データ
    r_train = rets.loc[tr_s:tr_e].dropna(how="all", axis=1)
    # 欠損の多い銘柄は除外（学習期間に50%以上欠損なら外す）
    ok = r_train.notna().mean() >= 0.5
    r_train = r_train.loc[:, ok.index[ok]]
    # 共分散行列
    cov = r_train.cov(min_periods=20)
    cov = cov.dropna(how="any", axis=0).dropna(how="any", axis=1)
    if cov.shape[0] < K_STOCKS:
        # 銘柄が足りないならスキップ（全額キャッシュ）
        for d in rets.loc[te_s:te_e].index:
            equity.append((d, nav))
        continue

    # Greedyで5銘柄＋最小分散ウェイト
    w = greedy_pick_5(cov)  # Series index=銘柄, values=weight
    weights_log.append({"train_start":tr_s.date(),"train_end":tr_e.date(),
                        "test_start":te_s.date(),"test_end":te_e.date(),
                        **{f"w_{k}":float(v) for k,v in w.items()}})

    # テスト期間のパフォーマンス（固定ウェイト）
    r_test = rets.loc[te_s:te_e, w.index].fillna(0.0)
    port_r = (r_test * w.values).sum(axis=1)  # 日次
    for d, r in port_r.items():
        nav *= (1.0 + r)
        equity.append((d, nav))

# ---------- 結果まとめ ----------
eq = pd.DataFrame(equity, columns=["date","equity"]).set_index("date").sort_index()
roll_max = eq["equity"].cummax()
dd = eq["equity"]/roll_max - 1.0
total_return = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0
years = (eq.index[-1] - eq.index[0]).days / 365.25
cagr = (1+total_return)**(1/years) - 1.0 if years>0 else np.nan

summary = {
    "periods": len(periods),
    "window_months": WINDOW_MONTHS,
    "k_stocks": K_STOCKS,
    "allow_short": ALLOW_SHORT,
    "total_return_%": round(100*total_return,2),
    "CAGR_%": round(100*cagr,2),
    "MaxDD_%": round(100*dd.min(),2)
}
weights_df = pd.DataFrame(weights_log)

print("Summary:", summary)
display(eq.tail(3))
display(weights_df.head())

# ちょい可視化
import matplotlib.pyplot as plt
eq["equity"].plot(figsize=(8,5))
plt.title(f"Rolling Min-Var Portfolio (W={WINDOW_MONTHS} months, K={K_STOCKS}, long-only={not ALLOW_SHORT})")
plt.xlabel("Date"); plt.ylabel("Equity")
plt.show()
