# === SPY × GLD ハイブリッド型リバランス（半年ごと/乖離バンドあり）10年バックテスト ===
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import date

# -------- パラメータ（ここだけ調整）--------
TARGET_W_SPY = 0.55      # 変数1：目標比率（例：SPY 60%, GLD 40%）
BAND_PCT     = 10.0       # 変数2：許容乖離（±%）。例：5.0 → 目標から±5%超でリバランス
FEE_BPS      = 5.0       # 片道手数料(bps)。例：5~10を入れると現実寄り
START = "2005-01-01"     # 20年分めやす
END   = pd.Timestamp.today().date().isoformat()  # 今日まで

# -------- データ取得（調整後終値＝配当込近似）--------
spy = yf.download("SPY", start=START, end=END, auto_adjust=False, progress=False)
gld = yf.download("GLD", start=START, end=END, auto_adjust=False, progress=False)
if spy.empty or gld.empty:
    raise RuntimeError("価格データを取得できませんでした。期間/ネットワークをご確認ください。")

# ---- ここから差し替え ----
def load_price(ticker, col="Adj Close"):
    df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    if df is None or df.empty or col not in df.columns:
        raise RuntimeError(f"{ticker}: {col} が取得できませんでした")
    out = df[[col]].copy()
    out.columns = [ticker]  # 列名を直接設定（renameを使わない）
    return out

spy_px = load_price("SPY")
gld_px = load_price("GLD")

# 取引日の共通部分で結合（インナー結合）
px = spy_px.join(gld_px, how="inner")
px = px.dropna()
# ---- ここまで差し替え ----

# -------- 日次リターン --------
rets = px.pct_change().dropna()

# -------- 半年ごとのチェック日（営業日ベースで近い日を採用）--------
# 始点＝最初の取引日月末に近い日でもよいが、ここでは「最初の取引日」から半年刻み
reb_days = []
cur = px.index[0]
end = px.index[-1]
while cur < end:
    reb_days.append(cur)
    cur = (cur + relativedelta(months=6))
# 上の reb_days はカレンダー日なので、各日に一番近い取引日にスナップ
reb_days = [px.index[px.index.get_indexer([d], method="nearest")[0]] for d in reb_days]
reb_set = set(reb_days)

# -------- バックテスト本体 --------
target = np.array([TARGET_W_SPY, 1.0 - TARGET_W_SPY])   # 目標比率ベクトル
fee_rate = FEE_BPS / 10000.0

dates = rets.index
w = target.copy()       # 初期は目標比率でスタート
nav = [1.0]             # 基準価格=1.0
turnovers = []          # |Δw| の合計（片側合計、二資産なので |Δw_spy|+|Δw_gld| = 2*|Δw_spy| 等）

for i, d in enumerate(dates, start=1):
    # 当日のリターンを適用（前日のウェイトで今日の損益）
    r = rets.loc[d, ["SPY","GLD"]].values
    nav.append(nav[-1] * (1.0 + float((w * r).sum())))

    # 今日が“チェック日”なら、終値ベースのウェイト乖離を測ってリバランス判定
    if d in reb_set:
        # 現在の時価ウェイト（今日終値時点）
        # NAVは直前に更新済みなので、w は「今日終値時点」の時価ウェイトと等価
        cur_w = w.copy()
        drift = cur_w - target     # ベクトル乖離
        # SPYの乖離を指標にする（GLDは1-それ）
        drift_spy_pct = 100.0 * float(drift[0])
        # 乖離が閾値を超えたら“翌営業日”から目標に戻す（先読み防止のため翌日適用）
        if abs(drift_spy_pct) >= BAND_PCT:
            new_w = target.copy()
            # 売買コスト：翌日に適用される取引としてここで控除（保守的に当日控除でもOK）
            tw = np.abs(new_w - w).sum()
            turnovers.append(tw)
            # フィー控除をNAVから反映
            nav[-1] = nav[-1] * (1.0 - fee_rate * tw)
            # 翌日の運用ウェイトに反映
            w = new_w
        # 超えなければリバランスなし（wは据え置き）
    # チェック日以外は何もしない

# NAV配列を整形（最初の1.0を外して日次に合わせる）
nav = pd.Series(nav[1:], index=dates, name="Portfolio_NAV")

# ベンチマーク（SPY買いっぱなし、初期=1に正規化）
spy_nav = (1.0 + rets["SPY"]).cumprod()
spy_nav.name = "SPY_BH"

# -------- サマリー --------
roll_max = nav.cummax(); dd = nav/roll_max - 1.0
total_return = nav.iloc[-1] - 1.0
years = (nav.index[-1] - nav.index[0]).days / 365.25
cagr = (nav.iloc[-1])**(1/years) - 1.0 if years>0 else np.nan
turnover_daily = np.mean(turnovers)/1.0 if turnovers else 0.0  # おおざっぱな平均（チェック日のみ）

summary = {
    "period": f"{nav.index[0].date()} → {nav.index[-1].date()}",
    "target_w_spy": TARGET_W_SPY,
    "band_%": BAND_PCT,
    "fee_bps": FEE_BPS,
    "total_return_%": round(100*float(total_return),2),
    "CAGR_%": round(100*float(cagr),2),
    "MaxDD_%": round(100*float(dd.min()),2),
}
print("Summary:", summary)

# -------- グラフ 1: ポートフォリオのエクイティカーブ --------
plt.figure(figsize=(9,5))
nav.plot(label="Portfolio (SPY/GLD hybrid)")
plt.title("Portfolio Equity Curve")
plt.xlabel("Date"); plt.ylabel("NAV")
plt.legend(); plt.tight_layout(); plt.show()

# -------- グラフ 2: SP500単独投資との比較 --------
plt.figure(figsize=(9,5))
nav.plot(label="Portfolio")
spy_nav.plot(label="SPY Buy&Hold", alpha=0.8)
plt.title("Portfolio vs. SPY Buy & Hold")
plt.xlabel("Date"); plt.ylabel("NAV (normalized)")
plt.legend(); plt.tight_layout(); plt.show()
