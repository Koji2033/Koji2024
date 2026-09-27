import datetime as dt
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="株価チャート・売買シミュレーター",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinanceの返却形式差を吸収し、OHLCVを単一列に整える。"""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # yfinanceがMultiIndexを返すケースに対応
    if isinstance(out.columns, pd.MultiIndex):
        # 典型例: level 0 = Price, level 1 = Ticker
        if ticker in out.columns.get_level_values(-1):
            try:
                out = out.xs(ticker, axis=1, level=-1)
            except Exception:
                pass
        if isinstance(out.columns, pd.MultiIndex):
            # 残った階層から OHLCV を含む側を優先
            for level in range(out.columns.nlevels):
                vals = set(map(str, out.columns.get_level_values(level)))
                if {"Open", "High", "Low", "Close"}.issubset(vals):
                    out.columns = out.columns.get_level_values(level)
                    break

    wanted = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    keep = [c for c in wanted if c in out.columns]
    out = out[keep].copy()

    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Yahoo Finance経由で価格データを取得。endはyfinance仕様上exclusiveなので1日加算。"""
    end_exclusive = end + dt.timedelta(days=1)
    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    return normalize_ohlcv(raw, ticker)


def add_indicators(df: pd.DataFrame, sma_list: list[int]) -> pd.DataFrame:
    out = df.copy()
    for n in sorted(set(sma_list)):
        out[f"SMA{n}"] = out["Close"].rolling(n).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))
    out["RSI14"] = out["RSI14"].fillna(100).clip(0, 100)
    return out


def stock_chart(df: pd.DataFrame, ticker: str, sma_list: list[int]) -> go.Figure:
    rows = 3
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.68, 0.18, 0.14],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="株価",
        ),
        row=1,
        col=1,
    )

    for n in sorted(set(sma_list)):
        col = f"SMA{n}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], mode="lines", name=f"SMA {n}"),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="出来高"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI 14"),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", row=3, col=1)

    fig.update_layout(
        title=f"{ticker} 株価チャート",
        height=760,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="価格", row=1, col=1)
    fig.update_yaxes(title_text="出来高", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    return fig


def backtest_ma(
    df: pd.DataFrame,
    initial_cash: float,
    short_n: int,
    long_n: int,
    commission_pct: float,
    slippage_pct: float,
):
    """
    MAクロス:
      - t日の終値まででクロスを判定
      - t+1日の始値で約定
      - 現物・全額売買、空売りなし
    """
    d = df.copy()
    d["short_ma"] = d["Close"].rolling(short_n).mean()
    d["long_ma"] = d["Close"].rolling(long_n).mean()
    d["signal"] = (d["short_ma"] > d["long_ma"]).astype(int)

    # 当日終値シグナルを翌営業日の始値で執行
    d["exec_signal"] = d["signal"].shift(1).fillna(0).astype(int)

    cash = float(initial_cash)
    shares = 0
    equity = []
    trades = []
    entry_total_cost = None
    entry_date = None
    entry_price = None

    commission_rate = commission_pct / 100.0
    slip_rate = slippage_pct / 100.0

    prev_exec_signal = 0

    for date, row in d.iterrows():
        target = int(row["exec_signal"])
        open_px = float(row["Open"])
        close_px = float(row["Close"])

        if target == 1 and prev_exec_signal == 0 and shares == 0:
            buy_px = open_px * (1 + slip_rate)
            per_share_cost = buy_px * (1 + commission_rate)
            qty = math.floor(cash / per_share_cost) if per_share_cost > 0 else 0

            if qty > 0:
                gross = qty * buy_px
                fee = gross * commission_rate
                total = gross + fee
                cash -= total
                shares = qty
                entry_total_cost = total
                entry_date = date
                entry_price = buy_px
                trades.append(
                    {
                        "日付": date.date().isoformat(),
                        "売買": "買",
                        "株数": qty,
                        "約定価格": round(buy_px, 4),
                        "手数料": round(fee, 2),
                        "実現損益": np.nan,
                    }
                )

        elif target == 0 and prev_exec_signal == 1 and shares > 0:
            sell_px = open_px * (1 - slip_rate)
            gross = shares * sell_px
            fee = gross * commission_rate
            proceeds = gross - fee
            cash += proceeds
            pnl = proceeds - (entry_total_cost or 0)

            trades.append(
                {
                    "日付": date.date().isoformat(),
                    "売買": "売",
                    "株数": shares,
                    "約定価格": round(sell_px, 4),
                    "手数料": round(fee, 2),
                    "実現損益": round(pnl, 2),
                }
            )
            shares = 0
            entry_total_cost = None
            entry_date = None
            entry_price = None

        market_value = shares * close_px
        equity.append(cash + market_value)
        prev_exec_signal = target

    d["Equity"] = equity

    final_equity = float(d["Equity"].iloc[-1]) if not d.empty else initial_cash
    total_return = (final_equity / initial_cash - 1) * 100 if initial_cash else np.nan

    # Buy & Hold比較: 初日終値→最終日終値（単純価格リターン）
    benchmark = (d["Close"].iloc[-1] / d["Close"].iloc[0] - 1) * 100 if len(d) >= 2 else np.nan

    peak = d["Equity"].cummax()
    drawdown = d["Equity"] / peak - 1
    max_dd = float(drawdown.min() * 100) if len(drawdown) else np.nan

    trade_df = pd.DataFrame(trades)
    sell_rows = trade_df[trade_df["売買"] == "売"] if not trade_df.empty else pd.DataFrame()
    wins = int((sell_rows["実現損益"] > 0).sum()) if not sell_rows.empty else 0
    closed = len(sell_rows)
    win_rate = wins / closed * 100 if closed else np.nan

    metrics = {
        "final_equity": final_equity,
        "total_return": total_return,
        "benchmark": benchmark,
        "max_drawdown": max_dd,
        "closed_trades": closed,
        "win_rate": win_rate,
        "open_position": shares,
    }
    return d, trade_df, metrics


def manual_trade_sim(
    df: pd.DataFrame,
    buy_date,
    sell_date,
    shares: int,
    commission_pct: float,
    slippage_pct: float,
):
    if buy_date >= sell_date:
        raise ValueError("売却日は購入日より後の日付を選んでください。")
    if shares <= 0:
        raise ValueError("株数は1以上にしてください。")

    idx = df.index
    buy_candidates = idx[idx.date >= buy_date]
    sell_candidates = idx[idx.date >= sell_date]
    if len(buy_candidates) == 0 or len(sell_candidates) == 0:
        raise ValueError("指定日以降の取引日が価格データ内にありません。")

    bd = buy_candidates[0]
    sd = sell_candidates[0]
    if bd >= sd:
        raise ValueError("実際の取引日ベースで売却日が購入日より後になるよう設定してください。")

    fee_rate = commission_pct / 100.0
    slip_rate = slippage_pct / 100.0
    buy_px = float(df.loc[bd, "Open"]) * (1 + slip_rate)
    sell_px = float(df.loc[sd, "Open"]) * (1 - slip_rate)

    buy_gross = buy_px * shares
    buy_fee = buy_gross * fee_rate
    sell_gross = sell_px * shares
    sell_fee = sell_gross * fee_rate
    invested = buy_gross + buy_fee
    proceeds = sell_gross - sell_fee
    pnl = proceeds - invested
    ret = pnl / invested * 100 if invested else np.nan

    return {
        "購入取引日": bd.date().isoformat(),
        "売却取引日": sd.date().isoformat(),
        "購入約定価格": buy_px,
        "売却約定価格": sell_px,
        "購入総額": invested,
        "売却受取額": proceeds,
        "損益": pnl,
        "収益率": ret,
    }


st.title("📈 株価チャート・売買シミュレーター")
st.caption(
    "価格データ取得: Yahoo Finance（yfinance）。教育・検証用です。実売買注文は行いません。"
)

with st.sidebar:
    st.header("基本設定")
    ticker = st.text_input(
        "銘柄コード",
        value="7203.T",
        help="日本株は例: 7203.T、米国株は例: AAPL",
    ).strip().upper()

    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 2)
    start = st.date_input("開始日", value=default_start)
    end = st.date_input("終了日", value=today)

    st.subheader("チャート")
    sma_list = st.multiselect(
        "移動平均",
        options=[5, 10, 20, 25, 50, 60, 75, 100, 200],
        default=[5, 20, 60],
    )

    st.subheader("売買コスト")
    commission_pct = st.number_input(
        "片道手数料 (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.01,
        format="%.3f",
    )
    slippage_pct = st.number_input(
        "スリッページ (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.05,
        step=0.01,
        format="%.3f",
    )

    reload_button = st.button("価格データを再取得", use_container_width=True)
    if reload_button:
        load_prices.clear()

if not ticker:
    st.warning("銘柄コードを入力してください。")
    st.stop()
if start >= end:
    st.error("終了日は開始日より後にしてください。")
    st.stop()

with st.spinner("価格データを取得しています…"):
    try:
        prices = load_prices(ticker, start, end)
    except Exception as e:
        st.error(f"価格データ取得に失敗しました: {e}")
        st.stop()

if prices.empty:
    st.error(
        "価格データを取得できませんでした。銘柄コード・期間を確認してください。"
        " 日本株は通常「7203.T」のように .T を付けます。"
    )
    st.stop()

data = add_indicators(prices, sma_list)

latest = data.iloc[-1]
prev = data.iloc[-2] if len(data) >= 2 else latest
delta = latest["Close"] - prev["Close"]
delta_pct = delta / prev["Close"] * 100 if prev["Close"] else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("終値", f"{latest['Close']:,.2f}", f"{delta:+,.2f}")
m2.metric("前日比", f"{delta_pct:+.2f}%")
m3.metric("出来高", f"{latest['Volume']:,.0f}")
m4.metric("RSI(14)", f"{latest['RSI14']:.1f}")

tab1, tab2, tab3 = st.tabs(["チャート", "自動売買バックテスト", "手動売買シミュレーション"])

with tab1:
    fig = stock_chart(data, ticker, sma_list)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    with st.expander("価格データ"):
        view = data.reset_index().rename(columns={"index": "Date"})
        st.dataframe(view.tail(250), use_container_width=True, hide_index=True)
        st.download_button(
            "価格データCSVをダウンロード",
            data=view.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{ticker}_prices.csv",
            mime="text/csv",
        )

with tab2:
    st.subheader("移動平均クロス戦略")
    st.write(
        "短期移動平均が長期移動平均を上抜けたら買い、下抜けたら売り。"
        " シグナルは終値で判定し、翌取引日の始値で約定します。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        initial_cash = st.number_input(
            "初期資金",
            min_value=10_000.0,
            value=1_000_000.0,
            step=100_000.0,
            format="%.0f",
        )
    with c2:
        short_n = st.number_input("短期MA", min_value=2, max_value=200, value=20, step=1)
    with c3:
        long_n = st.number_input("長期MA", min_value=3, max_value=400, value=60, step=1)

    if short_n >= long_n:
        st.warning("短期MAは長期MAより小さくしてください。")
    elif len(data) <= long_n + 2:
        st.warning("選択期間が短すぎます。開始日を古くしてください。")
    else:
        bt, trades, metrics = backtest_ma(
            data,
            initial_cash=initial_cash,
            short_n=int(short_n),
            long_n=int(long_n),
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )

        a, b, c, d, e = st.columns(5)
        a.metric("最終資産", f"{metrics['final_equity']:,.0f}")
        b.metric("戦略収益率", f"{metrics['total_return']:+.2f}%")
        c.metric("単純保有リターン", f"{metrics['benchmark']:+.2f}%")
        d.metric("最大DD", f"{metrics['max_drawdown']:.2f}%")
        e.metric(
            "勝率",
            "―" if np.isnan(metrics["win_rate"]) else f"{metrics['win_rate']:.1f}%",
            f"決済 {metrics['closed_trades']}回",
        )

        eq_fig = go.Figure()
        eq_fig.add_trace(
            go.Scatter(
                x=bt.index,
                y=bt["Equity"],
                mode="lines",
                name="戦略資産",
            )
        )
        benchmark_equity = initial_cash * bt["Close"] / bt["Close"].iloc[0]
        eq_fig.add_trace(
            go.Scatter(
                x=bt.index,
                y=benchmark_equity,
                mode="lines",
                name="単純保有（比較）",
            )
        )
        eq_fig.update_layout(
            title="資産推移",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h"),
            margin=dict(l=30, r=20, t=55, b=20),
        )
        st.plotly_chart(eq_fig, use_container_width=True, config={"displaylogo": False})

        if metrics["open_position"] > 0:
            st.info(
                f"期間終了時点で {metrics['open_position']:,} 株を保有したままです。"
                " 最終資産は最終終値で時価評価しています。"
            )

        st.markdown("#### 売買履歴")
        if trades.empty:
            st.write("期間内に売買シグナルはありませんでした。")
        else:
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button(
                "売買履歴CSVをダウンロード",
                data=trades.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{ticker}_ma_backtest_trades.csv",
                mime="text/csv",
            )

with tab3:
    st.subheader("指定日の1往復売買を試算")
    st.write(
        "指定日が休場日の場合は、その日以降で最初の取引日の始値を使います。"
        " 購入・売却とも始値約定として計算します。"
    )

    min_date = data.index.min().date()
    max_date = data.index.max().date()
    span = (max_date - min_date).days
    default_buy = min_date + dt.timedelta(days=max(1, int(span * 0.35)))
    default_sell = min_date + dt.timedelta(days=max(2, int(span * 0.70)))
    if default_sell > max_date:
        default_sell = max_date

    x1, x2, x3 = st.columns(3)
    with x1:
        buy_date = st.date_input(
            "購入日",
            value=default_buy,
            min_value=min_date,
            max_value=max_date,
            key="manual_buy",
        )
    with x2:
        sell_date = st.date_input(
            "売却日",
            value=default_sell,
            min_value=min_date,
            max_value=max_date,
            key="manual_sell",
        )
    with x3:
        shares = st.number_input(
            "株数",
            min_value=1,
            value=100,
            step=1,
        )

    try:
        result = manual_trade_sim(
            data,
            buy_date=buy_date,
            sell_date=sell_date,
            shares=int(shares),
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("購入総額", f"{result['購入総額']:,.0f}")
        r2.metric("売却受取額", f"{result['売却受取額']:,.0f}")
        r3.metric("損益", f"{result['損益']:+,.0f}")
        r4.metric("収益率", f"{result['収益率']:+.2f}%")

        details = pd.DataFrame(
            [
                ["購入取引日", result["購入取引日"]],
                ["売却取引日", result["売却取引日"]],
                ["購入約定価格", f"{result['購入約定価格']:,.4f}"],
                ["売却約定価格", f"{result['売却約定価格']:,.4f}"],
                ["株数", f"{int(shares):,}"],
            ],
            columns=["項目", "値"],
        )
        st.dataframe(details, use_container_width=True, hide_index=True)
    except ValueError as e:
        st.warning(str(e))

st.divider()
st.caption(
    "注意: yfinance/Yahoo Financeのデータは遅延・欠損・修正があり得ます。"
    " 本アプリは学習・検証用で、投資助言や実際の注文機能ではありません。"
)
