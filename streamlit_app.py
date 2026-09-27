import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


SMA_COLORS = [
    "#E53935",  # 赤
    "#43A047",  # 緑
    "#1E88E5",  # 青
    "#FDD835",  # 黄
    "#FB8C00",  # オレンジ
    "#EF9A9A",  # 薄い赤
    "#A5D6A7",  # 薄い緑
    "#90CAF9",  # 薄い青
    "#FFCC80",  # 薄いオレンジ
]

UP_COLOR = "#E53935"  # 赤
DOWN_COLOR = "#1E88E5"  # 青
UNCHANGED_COLOR = "#9E9E9E"  # 灰色


st.set_page_config(
    page_title="株価チャート・手動売買シミュレーター",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Price data
# -----------------------------
def normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance return formats to a single OHLCV table."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        if ticker in out.columns.get_level_values(-1):
            try:
                out = out.xs(ticker, axis=1, level=-1)
            except Exception:
                pass
        if isinstance(out.columns, pd.MultiIndex):
            for level in range(out.columns.nlevels):
                vals = set(map(str, out.columns.get_level_values(level)))
                if {"Open", "High", "Low", "Close"}.issubset(vals):
                    out.columns = out.columns.get_level_values(level)
                    break

    wanted = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    keep = [c for c in wanted if c in out.columns]
    out = out[keep].copy()

    for c in wanted:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in out.columns:
        out["Volume"] = 0
    return out


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download daily prices. yfinance end date is exclusive."""
    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    return normalize_ohlcv(raw, ticker)


def aggregate_prices(daily: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Aggregate only rows already visible to the user.
    Weekly/monthly labels use the last actual visible trading day.
    """
    if daily.empty or timeframe == "日足":
        return daily.copy()

    work = daily.copy()
    if timeframe == "週足":
        groups = work.index.to_period("W-FRI")
    else:
        groups = work.index.to_period("M")

    records = []
    dates = []
    for _, g in work.groupby(groups):
        if g.empty:
            continue
        records.append(
            {
                "Open": float(g["Open"].iloc[0]),
                "High": float(g["High"].max()),
                "Low": float(g["Low"].min()),
                "Close": float(g["Close"].iloc[-1]),
                "Volume": float(g["Volume"].sum()),
            }
        )
        dates.append(g.index[-1])

    return pd.DataFrame(records, index=pd.DatetimeIndex(dates))


def add_indicators(df: pd.DataFrame, sma_list: list[int]) -> pd.DataFrame:
    out = df.copy()
    for n in dict.fromkeys(sma_list):
        out[f"SMA{n}"] = out["Close"].rolling(n).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = (100 - (100 / (1 + rs))).clip(0, 100)
    return out


def volume_colors(close: pd.Series) -> list[str]:
    """Color each volume bar by its close compared with the previous bar."""
    change = close.diff()
    return [
        UP_COLOR if value > 0 else DOWN_COLOR if value < 0 else UNCHANGED_COLOR
        for value in change
    ]


def stock_chart(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    sma_list: list[int],
    view_start: int,
    view_end: int,
    price_range: tuple[float, float],
    volume_range: tuple[float, float],
    x_revision: str,
    price_y_revision: str,
    volume_y_revision: str,
) -> go.Figure:
    """
    Load the whole available history into Plotly, but show only view_start:view_end.
    A categorical x-axis removes weekend/holiday gaps while keeping all loaded bars
    immediately available for zoom-out and pan.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.68, 0.18, 0.14],
    )

    x = [d.strftime("%Y-%m-%d") for d in df.index]

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="株価",
            increasing=dict(
                line=dict(color=UP_COLOR),
                fillcolor=UP_COLOR,
            ),
            decreasing=dict(
                line=dict(color=DOWN_COLOR),
                fillcolor=DOWN_COLOR,
            ),
        ),
        row=1,
        col=1,
    )

    for color_index, n in enumerate(dict.fromkeys(sma_list)):
        col = f"SMA{n}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[col],
                    mode="lines",
                    name=f"SMA {n}",
                    line=dict(
                        color=SMA_COLORS[color_index % len(SMA_COLORS)],
                        width=2,
                    ),
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Bar(
            x=x,
            y=df["Volume"],
            name="出来高",
            marker_color=volume_colors(df["Close"]),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df["RSI14"], mode="lines", name="RSI 14"),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", row=3, col=1)

    fig.update_layout(
        title=f"{ticker} {timeframe}チャート",
        height=760,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=60, b=20),
        dragmode="pan",
    )

    # Category axes use zero-based category serial positions for numeric ranges.
    # This makes the viewport an exact N-bar window with no market-closure gaps.
    x_range = [view_start - 0.5, view_end + 0.5]
    for row in (1, 2, 3):
        fig.update_xaxes(
            type="category",
            range=x_range,
            uirevision=x_revision,
            row=row,
            col=1,
        )

    if len(x) > 0:
        visible_count = max(1, view_end - view_start + 1)
        step = max(1, visible_count // 12)
        tick_positions = list(range(view_start, view_end + 1, step))
        tickvals = [x[i] for i in tick_positions]
        ticktext = [
            pd.Timestamp(x[i]).strftime("%Y/%m/%d" if timeframe == "日足" else "%Y/%m")
            for i in tick_positions
        ]
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            row=3,
            col=1,
        )

    fig.update_yaxes(
        title_text="価格",
        range=list(price_range),
        uirevision=price_y_revision,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="出来高",
        range=list(volume_range),
        uirevision=volume_y_revision,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        uirevision=f"{x_revision}-rsi",
        row=3,
        col=1,
    )
    return fig


# -----------------------------
# Manual trading simulator
# -----------------------------
def empty_sim_state() -> dict:
    return {
        "active": False,
        "ticker": None,
        "current_idx": None,
        "start_idx": None,
        "initial_cash": 1_000_000.0,
        "cash": 1_000_000.0,
        "spot_qty": 0,
        "spot_avg": 0.0,
        "margin_long_qty": 0,
        "margin_long_avg": 0.0,
        "margin_short_qty": 0,
        "margin_short_avg": 0.0,
        "realized_pnl": 0.0,
        "pending_order": None,
        "order_log": [],
        "trade_log": [],
    }


if "sim" not in st.session_state:
    st.session_state.sim = empty_sim_state()
if "chart_cursor" not in st.session_state:
    st.session_state.chart_cursor = None
if "chart_cursor_signature" not in st.session_state:
    st.session_state.chart_cursor_signature = None
if "chart_scale_signature" not in st.session_state:
    st.session_state.chart_scale_signature = None
if "chart_price_range" not in st.session_state:
    st.session_state.chart_price_range = None
if "chart_volume_range" not in st.session_state:
    st.session_state.chart_volume_range = None
if "chart_scale_reset_token" not in st.session_state:
    st.session_state.chart_scale_reset_token = 0
if "chart_scale_context" not in st.session_state:
    st.session_state.chart_scale_context = None


def weighted_average(old_qty: int, old_avg: float, add_qty: int, add_price: float) -> float:
    total_qty = old_qty + add_qty
    if total_qty <= 0:
        return 0.0
    return (old_qty * old_avg + add_qty * add_price) / total_qty


def fill_price_for_order(order: dict, row: pd.Series) -> tuple[bool, float | None, str]:
    """Determine next-day execution using only that day's OHLC."""
    side = order["side"]
    method = order["method"]
    slip = float(order["slippage_pct"]) / 100.0

    open_px = float(row["Open"])
    high_px = float(row["High"])
    low_px = float(row["Low"])
    close_px = float(row["Close"])

    if method == "寄り成り":
        px = open_px * (1 + slip if side == "買い" else 1 - slip)
        return True, px, "寄り成り"

    if method == "引け成り":
        px = close_px * (1 + slip if side == "買い" else 1 - slip)
        return True, px, "引け成り"

    limit_px = float(order["limit_price"])
    if side == "買い":
        if open_px <= limit_px:
            return True, open_px, "指値（寄付で有利約定）"
        if low_px <= limit_px:
            return True, limit_px, "指値"
        return False, None, "指値に届かず失効"

    if open_px >= limit_px:
        return True, open_px, "指値（寄付で有利約定）"
    if high_px >= limit_px:
        return True, limit_px, "指値"
    return False, None, "指値に届かず失効"


def apply_spot_trade(sim: dict, side: str, qty: int, price: float, fee: float) -> tuple[bool, str]:
    if side == "買い":
        total = price * qty + fee
        if sim["cash"] + 1e-9 < total:
            return False, "現金不足のため約定できませんでした。"
        sim["cash"] -= total
        sim["realized_pnl"] -= fee
        sim["spot_avg"] = weighted_average(sim["spot_qty"], sim["spot_avg"], qty, price)
        sim["spot_qty"] += qty
        return True, "現物買い"

    if sim["spot_qty"] < qty:
        return False, "現物保有株数を超える売却のため約定できませんでした。"

    gross = price * qty
    sim["cash"] += gross - fee
    sim["realized_pnl"] += (price - sim["spot_avg"]) * qty - fee
    sim["spot_qty"] -= qty
    if sim["spot_qty"] == 0:
        sim["spot_avg"] = 0.0
    return True, "現物売り"


def apply_margin_trade(sim: dict, side: str, qty: int, price: float, fee: float) -> tuple[bool, str]:
    """
    Simplified margin accounting:
    opposite orders close existing opposite positions first, then excess opens a new position.
    No margin requirement, interest, stock-loan fee, or forced liquidation.
    """
    remaining = qty
    actions = []
    fee_per_share = fee / qty if qty else 0.0

    if side == "買い":
        close_qty = min(remaining, sim["margin_short_qty"])
        if close_qty > 0:
            allocated_fee = fee_per_share * close_qty
            pnl = (sim["margin_short_avg"] - price) * close_qty - allocated_fee
            sim["cash"] += pnl
            sim["realized_pnl"] += pnl
            sim["margin_short_qty"] -= close_qty
            if sim["margin_short_qty"] == 0:
                sim["margin_short_avg"] = 0.0
            remaining -= close_qty
            actions.append(f"信用売り返済 {close_qty}株")

        if remaining > 0:
            allocated_fee = fee_per_share * remaining
            sim["cash"] -= allocated_fee
            sim["realized_pnl"] -= allocated_fee
            sim["margin_long_avg"] = weighted_average(
                sim["margin_long_qty"], sim["margin_long_avg"], remaining, price
            )
            sim["margin_long_qty"] += remaining
            actions.append(f"信用買い新規 {remaining}株")

    else:
        close_qty = min(remaining, sim["margin_long_qty"])
        if close_qty > 0:
            allocated_fee = fee_per_share * close_qty
            pnl = (price - sim["margin_long_avg"]) * close_qty - allocated_fee
            sim["cash"] += pnl
            sim["realized_pnl"] += pnl
            sim["margin_long_qty"] -= close_qty
            if sim["margin_long_qty"] == 0:
                sim["margin_long_avg"] = 0.0
            remaining -= close_qty
            actions.append(f"信用買い返済 {close_qty}株")

        if remaining > 0:
            allocated_fee = fee_per_share * remaining
            sim["cash"] -= allocated_fee
            sim["realized_pnl"] -= allocated_fee
            sim["margin_short_avg"] = weighted_average(
                sim["margin_short_qty"], sim["margin_short_avg"], remaining, price
            )
            sim["margin_short_qty"] += remaining
            actions.append(f"信用売り新規 {remaining}株")

    return True, " / ".join(actions)


def process_pending_order(sim: dict, execution_date: pd.Timestamp, row: pd.Series) -> None:
    order = sim.get("pending_order")
    if not order:
        return

    filled, price, reason = fill_price_for_order(order, row)
    log_base = {
        "注文日": order["order_date"],
        "執行日": execution_date.date().isoformat(),
        "区分": order["account_type"],
        "売買": order["side"],
        "株数": order["qty"],
        "方法": order["method"],
        "指値": order.get("limit_price"),
    }

    if not filled or price is None:
        sim["order_log"].append(
            {**log_base, "状態": "失効", "約定価格": np.nan, "備考": reason}
        )
        sim["pending_order"] = None
        return

    fee_rate = float(order["commission_pct"]) / 100.0
    fee = float(price) * int(order["qty"]) * fee_rate

    if order["account_type"] == "現物":
        ok, action = apply_spot_trade(
            sim, order["side"], int(order["qty"]), float(price), fee
        )
    else:
        ok, action = apply_margin_trade(
            sim, order["side"], int(order["qty"]), float(price), fee
        )

    if ok:
        sim["order_log"].append(
            {
                **log_base,
                "状態": "約定",
                "約定価格": round(float(price), 4),
                "備考": f"{reason} / {action}",
            }
        )
        sim["trade_log"].append(
            {
                "日付": execution_date.date().isoformat(),
                "区分": order["account_type"],
                "内容": action,
                "株数": int(order["qty"]),
                "約定価格": round(float(price), 4),
                "手数料": round(fee, 2),
                "確定損益累計": round(sim["realized_pnl"], 2),
            }
        )
    else:
        sim["order_log"].append(
            {
                **log_base,
                "状態": "取消",
                "約定価格": np.nan,
                "備考": action,
            }
        )

    sim["pending_order"] = None


def unrealized_pnl(sim: dict, close_px: float) -> tuple[float, float, float, float]:
    spot = (close_px - sim["spot_avg"]) * sim["spot_qty"]
    mlong = (close_px - sim["margin_long_avg"]) * sim["margin_long_qty"]
    mshort = (sim["margin_short_avg"] - close_px) * sim["margin_short_qty"]
    return spot, mlong, mshort, spot + mlong + mshort


def fmt_price(v: float) -> str:
    return f"{v:,.2f}"


# -----------------------------
# Sidebar / load data
# -----------------------------
st.title("📈 株価チャート・手動売買シミュレーター")
st.caption("価格データ: Yahoo Finance（yfinance）。実注文は行わない学習・検証用アプリです。")

with st.sidebar:
    with st.expander("基本設定", expanded=True):
        ticker = st.text_input(
            "銘柄コード",
            value="7203.T",
            help="日本株は例: 7203.T、米国株は例: AAPL",
        ).strip().upper()

        today = dt.date.today()
        default_start = today - dt.timedelta(days=365 * 2)
        start = st.date_input("データ開始日", value=default_start)
        end = st.date_input("データ終了日", value=today)

        timeframe = st.radio("足種", ["日足", "週足", "月足"], horizontal=True)

        st.subheader("チャート")
        sma_list = st.multiselect(
            "移動平均",
            options=[5, 10, 20, 25, 50, 60, 75, 100, 200],
            default=[5, 20, 60],
        )
        auto_price_y = st.toggle(
            "価格の縦軸を自動調整",
            value=True,
            help="ONにすると、表示中のローソク足が収まる範囲に縦軸を自動調整します。",
        )
        auto_volume_y = st.toggle(
            "出来高の縦軸を自動調整",
            value=True,
            help="ONにすると、表示中の出来高に合わせて縦軸を自動調整します。",
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
            "成行スリッページ (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="寄り成り・引け成りに適用。指値では価格改善ルールを優先します。",
        )

        if st.button("価格データを再取得", use_container_width=True):
            load_prices.clear()
            st.rerun()

if not ticker:
    st.warning("銘柄コードを入力してください。")
    st.stop()
if start >= end:
    st.error("データ終了日は開始日より後にしてください。")
    st.stop()

with st.spinner("価格データを取得しています…"):
    try:
        prices = load_prices(ticker, start, end)
    except Exception as e:
        st.error(f"価格データ取得に失敗しました: {e}")
        st.stop()

if prices.empty:
    st.error("価格データを取得できませんでした。銘柄コード・期間を確認してください。")
    st.stop()

sim = st.session_state.sim
if sim["active"] and sim["ticker"] != ticker:
    st.session_state.sim = empty_sim_state()
    sim = st.session_state.sim
    st.warning("銘柄コードが変更されたため、手動売買シミュレーションを終了しました。")

# While a simulation is active, the cutoff applies to the entire app.
if sim["active"]:
    sim_idx = min(int(sim["current_idx"]), len(prices) - 1)
    visible_daily = prices.iloc[: sim_idx + 1].copy()
    cutoff_date = visible_daily.index[-1]
else:
    visible_daily = prices.copy()
    cutoff_date = visible_daily.index[-1]

display_bars = add_indicators(aggregate_prices(visible_daily, timeframe), sma_list)

tab_chart, tab_sim = st.tabs(["チャート", "手動売買シミュレーション"])

with tab_chart:
    if sim["active"]:
        st.info(
            f"シミュレーション中: {cutoff_date.date().isoformat()} までのデータだけ表示しています。"
            " 未来の価格データは非表示です。"
        )

    max_cursor = max(0, len(display_bars) - 1)
    bar_dates = [d.date() for d in display_bars.index]
    date_to_index = {d: i for i, d in enumerate(bar_dates)}

    range_key = f"chart_range::{ticker}::{timeframe}"
    shift_flag_key = f"chart_range_shifted::{ticker}::{timeframe}"
    last_range_key = f"chart_range_last::{ticker}::{timeframe}"

    default_start_idx = max(0, len(display_bars) - 60)
    default_range = (bar_dates[default_start_idx], bar_dates[-1])

    # If the instrument/timeframe changed, or simulation hides dates that had
    # previously been selected, reset to the latest 60 available bars.
    current_saved_range = st.session_state.get(range_key)
    if (
        current_saved_range is None
        or len(current_saved_range) != 2
        or current_saved_range[0] not in date_to_index
        or current_saved_range[1] not in date_to_index
        or current_saved_range[0] > current_saved_range[1]
    ):
        st.session_state[range_key] = default_range
        st.session_state[last_range_key] = default_range
        st.session_state[shift_flag_key] = False
        st.session_state.chart_scale_reset_token += 1

    saved_start_date, saved_end_date = st.session_state[range_key]
    saved_start_idx = date_to_index[saved_start_date]
    saved_end_idx = date_to_index[saved_end_date]
    saved_width = saved_end_idx - saved_start_idx + 1

    if sim["active"]:
        nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 1.25, 1, 2.4])
    else:
        nav1, nav2, nav3, nav5 = st.columns([1, 1, 1, 3])
        nav4 = None

    with nav1:
        if st.button(
            "◀ 1足戻す",
            disabled=saved_start_idx <= 0,
            use_container_width=True,
        ):
            new_start_idx = max(0, saved_start_idx - 1)
            new_end_idx = new_start_idx + saved_width - 1
            st.session_state[range_key] = (
                bar_dates[new_start_idx],
                bar_dates[new_end_idx],
            )
            st.session_state[shift_flag_key] = True
            st.rerun()

    with nav2:
        if st.button(
            "1足進める ▶",
            disabled=saved_end_idx >= max_cursor,
            use_container_width=True,
        ):
            new_end_idx = min(max_cursor, saved_end_idx + 1)
            new_start_idx = new_end_idx - saved_width + 1
            st.session_state[range_key] = (
                bar_dates[new_start_idx],
                bar_dates[new_end_idx],
            )
            st.session_state[shift_flag_key] = True
            st.rerun()

    with nav3:
        if st.button(
            "最新へ",
            disabled=saved_end_idx >= max_cursor,
            use_container_width=True,
        ):
            new_end_idx = max_cursor
            new_start_idx = max(0, new_end_idx - saved_width + 1)
            st.session_state[range_key] = (
                bar_dates[new_start_idx],
                bar_dates[new_end_idx],
            )
            st.session_state[shift_flag_key] = False
            st.session_state.chart_scale_reset_token += 1
            st.rerun()

    if sim["active"] and nav4 is not None:
        with nav4:
            current_idx_for_chart = min(int(sim["current_idx"]), len(prices) - 1)
            if st.button(
                "次の取引日へ ▶",
                type="primary",
                use_container_width=True,
                disabled=current_idx_for_chart >= len(prices) - 1,
                help="未来を1取引日だけ開示し、現在の表示幅を保ったまま右へ進めます。",
            ):
                next_idx = current_idx_for_chart + 1
                next_date = prices.index[next_idx]
                next_row = prices.iloc[next_idx]

                old_bar_count = len(display_bars)
                process_pending_order(sim, next_date, next_row)
                sim["current_idx"] = next_idx

                next_visible = prices.iloc[: next_idx + 1].copy()
                next_bars = aggregate_prices(next_visible, timeframe)
                new_bar_count = len(next_bars)
                added_bars = max(0, new_bar_count - old_bar_count)
                next_bar_dates = [d.date() for d in next_bars.index]

                # Keep the selected width and translate the whole window.
                # Even when weekly/monthly bar count does not increase, the
                # latest aggregate bar's label date changes as a new day is revealed.
                new_end_idx = min(
                    len(next_bar_dates) - 1,
                    saved_end_idx + added_bars,
                )
                new_start_idx = max(0, new_end_idx - saved_width + 1)
                st.session_state[range_key] = (
                    next_bar_dates[new_start_idx],
                    next_bar_dates[new_end_idx],
                )
                st.session_state[shift_flag_key] = True
                st.rerun()

    with nav5:
        st.write(
            f"表示: **{saved_start_date.isoformat()} ～ {saved_end_date.isoformat()}**　"
            f"({saved_width}足 / 読込済み {len(display_bars)}足)"
        )

    selected_range = st.select_slider(
        "表示範囲",
        options=bar_dates,
        value=st.session_state[range_key],
        key=range_key,
        help="左右のハンドルを個別に動かして表示開始日・終了日を調整できます。",
    )

    start_date_selected, end_date_selected = selected_range
    window_start = date_to_index[start_date_selected]
    cursor = date_to_index[end_date_selected]
    window_size = cursor - window_start + 1
    window_view = display_bars.iloc[window_start : cursor + 1].copy()

    previous_range = st.session_state.get(last_range_key)
    shifted_programmatically = bool(st.session_state.get(shift_flag_key, False))
    range_changed = previous_range != selected_range

    # Manual range changes recalculate the vertical scale. One-bar navigation
    # keeps the existing vertical magnification, matching the sliding-chart behavior.
    if range_changed and not shifted_programmatically:
        st.session_state.chart_scale_reset_token += 1

    st.session_state[last_range_key] = selected_range
    st.session_state[shift_flag_key] = False

    scale_context = (ticker, timeframe)
    if st.session_state.chart_scale_context != scale_context:
        st.session_state.chart_scale_context = scale_context
        st.session_state.chart_price_range = None
        st.session_state.chart_volume_range = None

    scale_signature = (
        ticker,
        timeframe,
        window_size,
        st.session_state.chart_scale_reset_token,
        window_start if auto_price_y or auto_volume_y else None,
        cursor if auto_price_y or auto_volume_y else None,
        auto_price_y,
        auto_volume_y,
    )
    if st.session_state.chart_scale_signature != scale_signature:
        st.session_state.chart_scale_signature = scale_signature

        if auto_price_y or st.session_state.chart_price_range is None:
            low = float(window_view["Low"].min())
            high = float(window_view["High"].max())
            price_span = max(high - low, abs(high) * 0.01, 1e-6)
            price_pad = price_span * 0.06
            st.session_state.chart_price_range = (low - price_pad, high + price_pad)

        if auto_volume_y or st.session_state.chart_volume_range is None:
            vol_max = float(window_view["Volume"].max()) if len(window_view) else 0.0
            st.session_state.chart_volume_range = (0.0, max(1.0, vol_max * 1.12))

    shown_start = display_bars.index[window_start]
    shown_end = display_bars.index[cursor]
    st.caption(
        f"選択範囲: {shown_start.date().isoformat()} ～ {shown_end.date().isoformat()} "
        f"（{window_size}足）"
    )

    if not window_view.empty:
        latest = display_bars.iloc[cursor]
        prev = display_bars.iloc[cursor - 1] if cursor >= 1 else latest
        delta = latest["Close"] - prev["Close"]
        delta_pct = delta / prev["Close"] * 100 if prev["Close"] else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("終値", fmt_price(float(latest["Close"])), f"{delta:+,.2f}")
        m2.metric("前足比", f"{delta_pct:+.2f}%")
        m3.metric("出来高", f"{latest['Volume']:,.0f}")
        rsi = latest.get("RSI14", np.nan)
        m4.metric("RSI(14)", "―" if pd.isna(rsi) else f"{float(rsi):.1f}")

        st.plotly_chart(
            stock_chart(
                display_bars,
                ticker,
                timeframe,
                sma_list,
                view_start=window_start,
                view_end=cursor,
                price_range=st.session_state.chart_price_range,
                volume_range=st.session_state.chart_volume_range,
                x_revision=f"{ticker}-{timeframe}-{window_start}-{cursor}",
                price_y_revision=(
                    f"{ticker}-{timeframe}-price-auto-{window_start}-{cursor}"
                    if auto_price_y
                    else f"{ticker}-{timeframe}-price-fixed"
                ),
                volume_y_revision=(
                    f"{ticker}-{timeframe}-volume-auto-{window_start}-{cursor}"
                    if auto_volume_y
                    else f"{ticker}-{timeframe}-volume-fixed"
                ),
            ),
            use_container_width=True,
            key="main_stock_chart",
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "doubleClick": "reset",
            },
        )

        st.caption(
            "左右2つのハンドルで表示期間を自由に調整できます。"
            " 縦軸自動調整がONの場合は表示範囲に追従し、OFFの場合は現在の倍率を保ちます。"
            " チャート上の1本指ドラッグはパン（移動）が既定です。"
        )

        with st.expander("表示中の価格データ"):
            view = window_view.reset_index().rename(columns={"index": "Date"})
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button(
                "表示中データCSVをダウンロード",
                data=view.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{ticker}_{timeframe}_visible_prices.csv",
                mime="text/csv",
            )

with tab_sim:
    sim = st.session_state.sim

    if not sim["active"]:
        st.subheader("シミュレーション開始")
        st.write(
            "開始すると、指定した取引日より後の価格はアプリ全体で非表示になります。"
            " 開始後はチャートタブの「次の取引日へ」で1日ずつ進めます。"
        )

        s1, s2 = st.columns(2)
        with s1:
            sim_start_date = st.date_input(
                "開始日",
                value=max(
                    prices.index.min().date(),
                    prices.index.max().date() - dt.timedelta(days=180),
                ),
                min_value=prices.index.min().date(),
                max_value=prices.index.max().date(),
                key="sim_start_date",
            )
        with s2:
            initial_cash = st.number_input(
                "初期資金",
                min_value=10_000.0,
                value=1_000_000.0,
                step=100_000.0,
                format="%.0f",
                key="sim_initial_cash",
            )

        if st.button("▶ シミュレーション開始", type="primary", use_container_width=True):
            candidates = np.where(prices.index.date >= sim_start_date)[0]
            if len(candidates) == 0:
                st.error("開始日以降の取引日がありません。")
            else:
                idx = int(candidates[0])
                new_sim = empty_sim_state()
                new_sim.update(
                    {
                        "active": True,
                        "ticker": ticker,
                        "current_idx": idx,
                        "start_idx": idx,
                        "initial_cash": float(initial_cash),
                        "cash": float(initial_cash),
                    }
                )
                st.session_state.sim = new_sim
                st.session_state.chart_cursor = None
                st.session_state.chart_cursor_signature = None
                st.session_state.chart_scale_reset_token += 1
                st.rerun()

    else:
        current_idx = min(int(sim["current_idx"]), len(prices) - 1)
        current_date = prices.index[current_idx]
        current_row = prices.iloc[current_idx]
        current_close = float(current_row["Close"])

        top1, top2, top3 = st.columns([2, 2, 1])
        with top1:
            st.markdown(f"### 現在日: {current_date.date().isoformat()}")
        with top2:
            if current_idx < len(prices) - 1:
                next_date = prices.index[current_idx + 1].date().isoformat()
                st.caption(f"次の取引日: {next_date}")
            else:
                st.caption("取得済みデータの最終取引日です。")
        with top3:
            if st.button("終了", use_container_width=True):
                st.session_state.sim = empty_sim_state()
                st.session_state.chart_cursor = None
                st.session_state.chart_cursor_signature = None
                st.session_state.chart_scale_reset_token += 1
                st.rerun()

        spot_u, long_u, short_u, total_u = unrealized_pnl(sim, current_close)
        net_assets = sim["cash"] + sim["spot_qty"] * current_close + long_u + short_u

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("現物", f"{sim['spot_qty']:,}株")
        r2.metric("信用買い", f"{sim['margin_long_qty']:,}株")
        r3.metric("信用売り", f"{sim['margin_short_qty']:,}株")
        r4.metric("現金", f"{sim['cash']:,.0f}")

        p1, p2, p3 = st.columns(3)
        p1.metric("確定損益", f"{sim['realized_pnl']:+,.0f}")
        p2.metric("含み損益", f"{total_u:+,.0f}")
        p3.metric("参考純資産", f"{net_assets:,.0f}")

        pos_rows = []
        if sim["spot_qty"]:
            pos_rows.append(
                ["現物", sim["spot_qty"], sim["spot_avg"], current_close, spot_u]
            )
        if sim["margin_long_qty"]:
            pos_rows.append(
                [
                    "信用買い",
                    sim["margin_long_qty"],
                    sim["margin_long_avg"],
                    current_close,
                    long_u,
                ]
            )
        if sim["margin_short_qty"]:
            pos_rows.append(
                [
                    "信用売り",
                    sim["margin_short_qty"],
                    sim["margin_short_avg"],
                    current_close,
                    short_u,
                ]
            )
        if pos_rows:
            st.dataframe(
                pd.DataFrame(
                    pos_rows,
                    columns=["区分", "株数", "平均建値", "現在値", "含み損益"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### 注文入力")
        if sim["pending_order"] is not None:
            po = sim["pending_order"]
            limit_text = (
                ""
                if po.get("limit_price") is None
                else f" / 指値 {po['limit_price']:,.2f}"
            )
            st.warning(
                f"翌取引日へ発注済み: {po['account_type']} {po['side']} "
                f"{po['qty']:,}株 / {po['method']}{limit_text}"
            )
            if st.button("未執行注文を取消"):
                sim["pending_order"] = None
                st.rerun()
        else:
            o1, o2, o3, o4 = st.columns(4)
            with o1:
                account_type = st.selectbox("取引区分", ["現物", "信用"])
            with o2:
                side = st.selectbox("売買", ["買い", "売り"])
            with o3:
                qty = st.number_input("株数", min_value=1, value=100, step=1)
            with o4:
                method = st.selectbox("売買方法", ["寄り成り", "引け成り", "指値"])

            limit_price = None
            if method == "指値":
                limit_price = st.number_input(
                    "指値",
                    min_value=0.01,
                    value=float(round(current_close, 2)),
                    step=0.5,
                    format="%.2f",
                )

            if current_idx >= len(prices) - 1:
                st.info("次の取引日データがないため、新規注文は出せません。")
            elif st.button("翌取引日に注文", type="primary", use_container_width=True):
                sim["pending_order"] = {
                    "order_date": current_date.date().isoformat(),
                    "account_type": account_type,
                    "side": side,
                    "qty": int(qty),
                    "method": method,
                    "limit_price": float(limit_price) if limit_price is not None else None,
                    "commission_pct": float(commission_pct),
                    "slippage_pct": float(slippage_pct),
                }
                st.rerun()

        st.caption(
            "注文は入力日の翌取引日にのみ執行判定します。"
            " 信用取引は簡易モデルで、委託保証金率・金利・貸株料・追証・強制決済は未実装です。"
        )

        if sim["order_log"]:
            st.markdown("#### 注文履歴")
            st.dataframe(
                pd.DataFrame(sim["order_log"]),
                use_container_width=True,
                hide_index=True,
            )

        if sim["trade_log"]:
            st.markdown("#### 約定履歴")
            trades = pd.DataFrame(sim["trade_log"])
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button(
                "約定履歴CSVをダウンロード",
                data=trades.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{ticker}_manual_sim_trades.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "注意: Yahoo Finance/yfinanceのデータは遅延・欠損・修正があり得ます。"
    " 本アプリは学習・検証用で、投資助言や証券会社への注文機能ではありません。"
)

