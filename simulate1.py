START = "2005-01-01"   # 開始日（必要に応じて変えてOK）
END   = "2025-12-31"   # 終了日
TICKERS = ["MO","PM","VZ","PFE","BTI","ENB","EPD"]

import os
os.makedirs("data/prices", exist_ok=True)   # 価格CSV置き場
os.makedirs("bucket_results", exist_ok=True) # 結果出力先
print("✅ folders ready")

import yfinance as yf
import pandas as pd

for t in TICKERS:
    df = yf.download(t, start=START, end=END, auto_adjust=False)  # Close/Adj Close 両方入る
    if isinstance(df.columns, pd.MultiIndex):                     # 念のためMultiIndexを平坦化
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"                                        # ← Date列が超重要
    df.to_csv(f"data/prices/{t}.csv")
    print("Saved", t)
print("✅ price CSVs done")

import yfinance as yf
import pandas as pd

start_ts = pd.to_datetime(START)  # tz無し
end_ts   = pd.to_datetime(END)    # tz無し

all_rows = []
for t in TICKERS:
    s = yf.Ticker(t).dividends    # index=ex-div date（tz付きのことあり）
    if s is None or len(s) == 0:
        print("No dividends for", t);
        continue

    idx = s.index
    if getattr(idx, "tz", None) is not None:  # tz付き→外す
        idx = idx.tz_localize(None)
    s.index = idx

    df = s.reset_index()
    df.columns = ["ex_date", "dividend"]
    df["ticker"] = t
    df = df[(df["ex_date"] >= start_ts) & (df["ex_date"] <= end_ts)]
    df = df.sort_values("ex_date")
    all_rows.append(df[["ticker","ex_date","dividend"]])

cal = pd.concat(all_rows, ignore_index=True).sort_values(["ticker","ex_date"])
cal.to_csv("dividend_rotation_calendar.csv", index=False)
print("✅ saved dividend_rotation_calendar.csv with", len(cal), "rows")
cal.head()

import pandas as pd

cal = pd.read_csv("dividend_rotation_calendar.csv", parse_dates=["ex_date"])
print("銘柄ごとの件数：")
print(cal.groupby("ticker").size().sort_index())

print("\n期間：", cal["ex_date"].min().date(), "→", cal["ex_date"].max().date())

# 年×銘柄ごとの回数（四半期配当なら概ね4回、BTIのように半期もあり）
print("\n年×銘柄の件数（頭だけ）：")
print(cal.assign(year=cal["ex_date"].dt.year).groupby(["ticker","year"]).size().head(20))

# ===== Bucket Rotation Backtester (10-before entry / 10-after exit) =====
import os, math, warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class Config:
    start_date: str = "2015-01-01"
    end_date: str   = "2025-12-31"
    tickers: List[str] = field(default_factory=lambda: ["MO","PM","VZ","PFE","BTI","ENB","EPD"])
    prices_dir: str = "data/prices"
    cal_csv: str    = "dividend_rotation_calendar.csv"
    out_dir: str    = "bucket_results"

    buy_days_before_ex: int = 10     # 10営業日前に買い
    sell_days_after_ex: int = 10     # exの10カレンダー日後に売り（週末は翌営業日）
    use_take_profit: bool = False
    take_profit_pct: float = 0.05    # +5% で利確
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.06      # -6% で損切

    lot_size: int = 1
    capital: float = 1_000_000.0
    fee_bps: float = 10.0            # 片道10bps
    tax_dividend_rate: float = 0.20315

    bucket_ranges: List[Tuple[int,int]] = field(default_factory=lambda: [(1,10),(11,20),(21,31)])

def bday_shift(date: pd.Timestamp, n_business_days: int) -> pd.Timestamp:
    d = date
    count = 0
    while count < n_business_days:
        d -= pd.Timedelta(days=1)
        if d.weekday() < 5:  # 月〜金
            count += 1
    return d

def load_prices(prices_dir: str, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    px = {}
    for t in tickers:
        path = os.path.join(prices_dir, f"{t}.csv")
        if not os.path.exists(path):
            warnings.warn(f"[WARN] price file missing: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        px[t] = df
    return px

def get_close(df: pd.DataFrame, d: pd.Timestamp) -> Optional[float]:
    if d in df.index:
        return float(df.loc[d, "Close"])
    prior = df.index[df.index <= d]
    return None if len(prior)==0 else float(df.loc[prior[-1], "Close"])

def fee_amount(notional: float, bps: float) -> float:
    return abs(notional) * (bps/10000.0)

@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_px: float
    shares: int
    ex_date: pd.Timestamp
    planned_exit: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    exit_px: Optional[float] = None
    pnl: float = 0.0
    div_net: float = 0.0
    hit_take: bool = False
    hit_stop: bool = False

class BucketRotation:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.px = load_prices(cfg.prices_dir, cfg.tickers)
        self.cal = self._load_calendar(cfg.cal_csv)
        self.cash = cfg.capital
        self.equity_curve = []
        self.positions_by_bucket: Dict[int, Optional[Position]] = {0: None, 1: None, 2: None}
        self.trades: List[Position] = []

    def _load_calendar(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["ex_date"])
        df = df[df["ticker"].isin(self.cfg.tickers)].copy()
        df = df[(df["ex_date"] >= pd.Timestamp(self.cfg.start_date)) &
                (df["ex_date"] <= pd.Timestamp(self.cfg.end_date))].copy()
        df["month"] = df["ex_date"].dt.month
        df["day"] = df["ex_date"].dt.day
        df.sort_values(["ex_date","ticker"], inplace=True)
        return df

    def plan_targets(self) -> List[Tuple[int, str, pd.Timestamp, pd.Timestamp]]:
        plans = []
        months = sorted(self.cal["ex_date"].dt.to_period("M").unique())
        for m in months:
            dfm = self.cal[self.cal["ex_date"].dt.to_period("M") == m].copy()
            for bi, (lo, hi) in enumerate(self.cfg.bucket_ranges):
                cand = dfm[(dfm["day"] >= lo) & (dfm["day"] <= hi)].copy()
                if cand.empty: continue
                row = cand.sort_values("ex_date").iloc[0]  # バケツ内で最も早い
                exd = pd.Timestamp(row["ex_date"])
                entry = bday_shift(exd, self.cfg.buy_days_before_ex)
                plans.append((bi, row["ticker"], exd, entry))
        plans.sort(key=lambda x: x[3])  # エントリー日で並べる
        return plans

    def free_bucket_count(self) -> int:
        return sum(1 for v in self.positions_by_bucket.values() if v is None)

    def run(self):
        cfg = self.cfg
        plans = self.plan_targets()
        plan_idx = 0
        bday_calendar = pd.bdate_range(cfg.start_date, cfg.end_date, freq="C")

        for d in bday_calendar:
            # 1) エントリー
            while plan_idx < len(plans) and plans[plan_idx][3] <= d:
                bi, tkr, exd, entry = plans[plan_idx]; plan_idx += 1
                if entry > d: continue
                if tkr not in self.px: continue
                if self.positions_by_bucket[bi] is not None: continue
                price = get_close(self.px[tkr], d)
                if price is None: continue

                target_cash = self.cash / max(1, self.free_bucket_count())
                shares = math.floor((target_cash / price) / cfg.lot_size) * cfg.lot_size
                if shares <= 0: continue
                cost = shares * price
                fee = fee_amount(cost, cfg.fee_bps)
                if self.cash < cost + fee: continue
                self.cash -= (cost + fee)

                planned_exit = pd.Timestamp(exd.date()) + pd.Timedelta(days=cfg.sell_days_after_ex)
                while planned_exit.weekday() >= 5:  # 週末なら翌営業日
                    planned_exit += pd.Timedelta(days=1)

                pos = Position(ticker=tkr, entry_date=d, entry_px=price,
                               shares=shares, ex_date=exd, planned_exit=planned_exit)
                self.positions_by_bucket[bi] = pos

            # 2) 配当入金（ex日にネット配当を現金化）
            for bi, pos in list(self.positions_by_bucket.items()):
                if pos is None: continue
                if d.normalize() == pos.ex_date.normalize():
                    row = self.cal[(self.cal["ticker"]==pos.ticker) & (self.cal["ex_date"]==pos.ex_date)]
                    if not row.empty:
                        gross = float(row.iloc[0]["dividend"]) * pos.shares
                        net = gross * (1.0 - cfg.tax_dividend_rate)
                        pos.div_net += net; self.cash += net

            # 3) 早期/時間エグジット
            for bi, pos in list(self.positions_by_bucket.items()):
                if pos is None: continue
                px = get_close(self.px[pos.ticker], d)
                if px is None: continue

                early = False
                if cfg.use_take_profit and (px - pos.entry_px)/pos.entry_px >= cfg.take_profit_pct:
                    early = True; pos.hit_take = True
                if (not early) and cfg.use_stop_loss and (pos.entry_px - px)/pos.entry_px >= cfg.stop_loss_pct:
                    early = True; pos.hit_stop = True
                if (not early) and d >= pos.planned_exit:
                    early = True

                if early:
                    proceeds = px * pos.shares
                    fee = fee_amount(proceeds, cfg.fee_bps)
                    self.cash += proceeds - fee
                    pos.exit_date = d; pos.exit_px = px
                    entry_fee = fee_amount(pos.entry_px * pos.shares, cfg.fee_bps)
                    pos.pnl = (px - pos.entry_px) * pos.shares - entry_fee - fee + pos.div_net
                    self.trades.append(pos); self.positions_by_bucket[bi] = None

            # 4) MTM（時価評価）
            m2m = 0.0
            for pos in self.positions_by_bucket.values():
                if pos is None: continue
                px = get_close(self.px[pos.ticker], d)
                if px is not None: m2m += px * pos.shares
            equity = self.cash + m2m
            self.equity_curve.append((d, equity))

        # 期末クローズ（残があれば）
        d = pd.Timestamp(cfg.end_date)
        for bi, pos in list(self.positions_by_bucket.items()):
            if pos is None: continue
            px = get_close(self.px[pos.ticker], d)
            if px is None: continue
            proceeds = px * pos.shares
            fee = fee_amount(proceeds, cfg.fee_bps)
            self.cash += proceeds - fee
            pos.exit_date = d; pos.exit_px = px
            entry_fee = fee_amount(pos.entry_px * pos.shares, cfg.fee_bps)
            pos.pnl = (px - pos.entry_px) * pos.shares - entry_fee - fee + pos.div_net
            self.trades.append(pos); self.positions_by_bucket[bi] = None

        return self.results()

    def results(self):
        eq = pd.DataFrame(self.equity_curve, columns=["date","equity"]).set_index("date")
        if len(eq)==0:
            total_return = 0.0; cagr = 0.0; mdd = 0.0
        else:
            total_return = (eq["equity"].iloc[-1] / self.cfg.capital) - 1.0
            days = (eq.index[-1] - eq.index[0]).days
            years = max(days/365.25, 1e-9)
            cagr = (eq["equity"].iloc[-1]/self.cfg.capital)**(1/years) - 1.0
            roll_max = eq["equity"].cummax()
            dd = eq["equity"]/roll_max - 1.0
            mdd = float(dd.min())

        tdf = pd.DataFrame([{
            "ticker": t.ticker,
            "bucket": self._bucket_of_trade(t),
            "entry": t.entry_date.date().isoformat() if t.entry_date is not None else None,
            "exit": t.exit_date.date().isoformat() if t.exit_date is not None else None,
            "ex_date": t.ex_date.date().isoformat(),
            "entry_px": t.entry_px,
            "exit_px": t.exit_px,
            "shares": t.shares,
            "div_net": t.div_net,
            "pnl": t.pnl,
            "hold_days": (t.exit_date - t.entry_date).days if (t.exit_date is not None and t.entry_date is not None) else None,
            "take": t.hit_take, "stop": t.hit_stop
        } for t in self.trades])

        summary = {
            "capital": self.cfg.capital,
            "total_return_pct": round(100*total_return,2),
            "cagr_pct": round(100*cagr,2),
            "max_drawdown_pct": round(100*mdd,2),
            "trades": int(len(tdf)),
            "win_rate_pct": round(100*float((tdf["pnl"]>0).mean()),2) if len(tdf) else None,
            "avg_hold_days": round(float(tdf["hold_days"].mean()),1) if len(tdf) else None,
            "avg_pnl": round(float(tdf["pnl"].mean()),2) if len(tdf) else None,
        }

        # 保存
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        eq.to_csv(os.path.join(self.cfg.out_dir, "equity_curve.csv"))
        tdf.to_csv(os.path.join(self.cfg.out_dir, "trades.csv"), index=False)
        import json
        with open(os.path.join(self.cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return {"summary": summary, "trades": tdf, "equity": eq,
                "files": {"equity_csv": os.path.join(self.cfg.out_dir, "equity_curve.csv"),
                          "trades_csv": os.path.join(self.cfg.out_dir, "trades.csv"),
                          "summary_json": os.path.join(self.cfg.out_dir, "summary.json")}}

    def _bucket_of_trade(self, t: Position) -> Optional[str]:
        d = pd.Timestamp(t.ex_date)
        for i,(lo,hi) in enumerate(self.cfg.bucket_ranges):
            if lo <= d.day <= hi:
                return ["A(1-10)","B(11-20)","C(21-31)"][i] if i < 3 else f"B{i}"
        return None
# ===== end of class definitions =====

cfg = Config(
    start_date=START,
    end_date=END,
    prices_dir="data/prices",
    cal_csv="dividend_rotation_calendar.csv",
    out_dir="bucket_results",
    buy_days_before_ex=25,
    sell_days_after_ex=40,
    use_take_profit=False,
    use_stop_loss=False
)

bt = BucketRotation(cfg)
res = bt.run()

print("Summary:", res["summary"])   # 成績サマリー
res["trades"].head()                # トレード例（先頭5行）

#25,40で2倍くらい

import matplotlib.pyplot as plt
res["equity"]["equity"].plot()
plt.title("Equity Curve (10 years)")
plt.xlabel("Date"); plt.ylabel("Equity")
plt.show()
