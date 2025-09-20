# ===== Trailing-peak r% stop & 25MA re-entry, cross-sectional portfolio =====
import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# ----------------- パラメータ -----------------
UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO","JPM","BAC","WFC",
    "V","MA","HD","PG","KO","PEP","MRK","PFE","JNJ","LLY","XOM","CVX",
    "COP","VZ","T","CSCO","ORCL","ADBE","CRM","INTC","QCOM","TXN","AMD",
    "UNH","ABBV","MCD","NFLX","COST","NKE","LOW","UPS","CAT","HON","PM",
    "MO","BKNG","AXP","BA","GE"
]  # ← お好みの50銘柄に置換可

START = "2005-01-01"
END   = "2025-12-31"

R_PCT = 0.15         # 直近高値から 15% 下落で売却（例）
MA_WINDOW = 25       # 再エントリー用のSMA日数（例：25日）
EQUAL_WEIGHT = True  # 稼働銘柄を等ウェイトで保有
FEE_BPS = 0.0        # 売買コスト（片道bps, 例: 10）※最初は0でOK
RET_COL = "Adj Close" # リターン算出列（配当再投資の影響込みならAdj Close推奨）

# ----------------- データ取得 -----------------
def fetch_prices(tickers, start, end, col="Adj Close"):
    frames = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or df.empty or col not in df.columns:
            print(f"[SKIP] {t} missing {col}")
            continue
        s = df[col].dropna().copy(); s.name = t
        frames[t] = s
    if not frames:
        raise RuntimeError("価格データが取得できませんでした")
    px = pd.concat(frames.values(), axis=1).sort_index().dropna(how="all")
    return px

prices = fetch_prices(UNIVERSE, START, END, RET_COL)
rets = prices.pct_change().fillna(0.0)  # 日次リターン
dates = prices.index

# ----------------- シグナル生成（銘柄ごと） -----------------
# ルール：
#   in_position: Trueなら保有
#   ・保有中：価格 <= 累積ピーク*(1-R_PCT) で EXIT（翌バー始値…の近似として当日クローズで）
#   ・非保有：価格がSMAを下から上にクロスで ENTRY
#   ・ENTRY時に新たな累積ピークをその日の価格でリセット
#
# ここでは日足終値ベースで近似し、約定スリッページは考慮しません

sma = prices.rolling(MA_WINDOW, min_periods=1).mean()
cross_up = (prices > sma) & (prices.shift(1) <= sma.shift(1))  # 上抜け

inpos = pd.DataFrame(False, index=dates, columns=prices.columns)
trades_log = {t: [] for t in prices.columns}  # [(entry_date, exit_date, ret), ...]

for t in prices.columns:
    p = prices[t].dropna()
    cu = cross_up[t].reindex(p.index).fillna(False)
    ip = False
    peak = None
    entry_px = None
    entry_date = None

    for d in p.index:
        px = p.loc[d]

        if ip:
            # 直近ピーク更新
            peak = max(peak, px)
            # r%超のドローダウンでEXIT
            if px <= peak * (1.0 - R_PCT):
                # トレードを確定
                ret = (px / entry_px) - 1.0
                trades_log[t].append((entry_date, d, ret))
                ip = False
                peak = None
                entry_px = None
                entry_date = None
            else:
                inpos.loc[d, t] = True
        else:
            # 非保有 → 上抜けでENTRY
            if cu.loc[d]:
                ip = True
                peak = px
                entry_px = px
                entry_date = d
                inpos.loc[d, t] = True

# ----------------- ポートフォリオ集計 -----------------
# 等ウェイト方式：
#   その日の inpos=True の銘柄数 n に対して、各銘柄ウェイト = 1/n（n=0なら現金）
holds = inpos.astype(float)
n_active = holds.sum(axis=1)
weights = holds.div(n_active.replace(0, np.nan), axis=0).fillna(0.0)

# 実現売買コスト（前日→当日のウェイト変化分に対して課金する簡易モデル）
fee_rate = FEE_BPS / 10000.0
turnover = (weights.diff().abs().sum(axis=1)).fillna(0.0)  # Σ|Δw_i|
gross_port_ret = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)  # 昨日決めたウェイトで当日リターン
fee = turnover * fee_rate
net_port_ret = gross_port_ret - fee

# NAV
nav = (1.0 + net_port_ret).cumprod()
nav.name = "NAV"

# サマリー
roll_max = nav.cummax()
dd = nav/roll_max - 1.0
total_return = nav.iloc[-1] - 1.0
years = (nav.index[-1] - nav.index[0]).days / 365.25
cagr = (nav.iloc[-1])**(1/years) - 1.0 if years>0 else np.nan

summary = {
    "start": str(nav.index[0].date()),
    "end": str(nav.index[-1].date()),
    "R_pct": R_PCT,
    "MA_window": MA_WINDOW,
    "equal_weight": EQUAL_WEIGHT,
    "fee_bps": FEE_BPS,
    "total_return_%": round(100*total_return,2),
    "CAGR_%": round(100*cagr,2),
    "MaxDD_%": round(100*dd.min(),2),
    "avg_active_names": round(float(n_active.replace(0, np.nan).mean()),2),
    "avg_turnover_%/day": round(100*float(turnover.mean()),2)
}

print("Summary:", summary)

# 直近の稼働銘柄とウェイト（参考）
latest_date = nav.index[-1]
latest_w = weights.loc[latest_date]
print("\nActive names @", latest_date.date(), ":", list(latest_w[latest_w>0].index))
print("Weights (non-zero):")
display(latest_w[latest_w>0].sort_values(ascending=False))

# ちょい可視化
nav.plot(figsize=(9,5))
plt.title(f"Trailing-{int(R_PCT*100)}% stop & SMA{MA_WINDOW} re-entry (EW across active names)")
plt.xlabel("Date"); plt.ylabel("NAV")
plt.show()

# トレード統計（銘柄別の勝率など）—簡易集計
stats = []
for t, logs in trades_log.items():
    if not logs:
        continue
    r = np.array([ret for (_,_,ret) in logs])
    stats.append({
        "ticker": t,
        "trades": len(r),
        "win_rate_%": round(100*np.mean(r>0),2),
        "avg_ret_%": round(100*np.mean(r),2),
        "median_ret_%": round(100*np.median(r),2)
    })
stats_df = pd.DataFrame(stats).sort_values("win_rate_%", ascending=False)
print("\nPer-name trade stats (top 10 by win_rate):")
display(stats_df.head(10))
