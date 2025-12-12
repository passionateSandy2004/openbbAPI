"""
Analysis service that wraps the existing logic for fetching data and computing indicators.

Functions here are imported by the FastAPI interface so the API file stays minimal.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Reuse the existing logic from main.py
from main import (
    obb,
    obb_technical_available,
    ta,
    compute_indicators,
    fetch_historical_zerodha,
    fetch_historical_yfinance,
    fetch_profile_openbb,
    fetch_profile_yfinance,
    kite,
    openbb_available,
)

DEFAULT_LOOKBACK_DAYS = 90

# Indicator catalog: name -> config
INDICATOR_CATALOG: Dict[str, Dict[str, Any]] = {
    # Core (already used)
    "sma": {"kind": "obb", "fn": "sma", "params": {"length": 20}, "min_bars": 20},
    "ema": {"kind": "obb", "fn": "ema", "params": {"length": 20}, "min_bars": 20},
    "rsi": {"kind": "obb", "fn": "rsi", "params": {"length": 14}, "min_bars": 14},
    "bbands": {"kind": "obb", "fn": "bbands", "params": {"length": 20}, "min_bars": 20},
    # Additional OpenBB technicals
    "atr": {"kind": "obb", "fn": "atr", "params": {"length": 14}, "min_bars": 14},
    "adx": {"kind": "obb", "fn": "adx", "params": {"length": 14}, "min_bars": 14},
    "obv": {"kind": "obb", "fn": "obv", "params": {}, "min_bars": 2},
    "vwap": {"kind": "obb", "fn": "vwap", "params": {}, "min_bars": 1},
    "kc": {"kind": "obb", "fn": "kc", "params": {"length": 20}, "min_bars": 20},
    "hma": {"kind": "obb", "fn": "hma", "params": {"length": 20}, "min_bars": 20},
    "wma": {"kind": "obb", "fn": "wma", "params": {"length": 20}, "min_bars": 20},
    "fib": {"kind": "obb", "fn": "fib", "params": {}, "min_bars": 1},
    "demark": {"kind": "obb", "fn": "demark", "params": {}, "min_bars": 1},
    "relative_rotation": {"kind": "obb", "fn": "relative_rotation", "params": {}, "min_bars": 1},
    "cg": {"kind": "obb", "fn": "cg", "params": {"length": 10}, "min_bars": 10},
    "clenow": {"kind": "obb", "fn": "clenow", "params": {"lookback": 20}, "min_bars": 20},
    "aroon": {"kind": "obb", "fn": "aroon", "params": {"length": 14}, "min_bars": 14},
    "fisher": {"kind": "obb", "fn": "fisher", "params": {"length": 9}, "min_bars": 9},
    "cci": {"kind": "obb", "fn": "cci", "params": {"length": 20}, "min_bars": 20},
    "donchian": {"kind": "obb", "fn": "donchian", "params": {"length": 20}, "min_bars": 20},
    "ichimoku": {"kind": "obb", "fn": "ichimoku", "params": {}, "min_bars": 52},
    "stoch": {"kind": "obb", "fn": "stoch", "params": {"fastk": 14, "fastd": 3}, "min_bars": 14},
    "adosc": {"kind": "obb", "fn": "adosc", "params": {}, "min_bars": 2},
    "ad": {"kind": "obb", "fn": "ad", "params": {}, "min_bars": 2},
    "cones": {"kind": "obb", "fn": "cones", "params": {}, "min_bars": 2},
    "zlma": {"kind": "obb", "fn": "zlma", "params": {"length": 20}, "min_bars": 20},
    # MACD via pandas_ta (OpenBB MACD caused duplicate columns)
    "macd": {"kind": "ta", "fn": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}, "min_bars": 26},
}


def _extract_values_from_result(result: Any, preferred_name: Optional[str] = None) -> Optional[Tuple[List[str], pd.DataFrame]]:
    """
    Try to convert an OpenBB result (OBBject/DataFrame/Series) to a DataFrame of indicators.
    Returns tuple (column_names, dataframe) or None if extraction fails.
    """
    result_df = None
    if result is None:
        return None

    if hasattr(result, "to_df"):
        try:
            result_df = result.to_df()
        except Exception as e:
            print(f"Error converting to_df: {e}")
            return None
    elif isinstance(result, pd.DataFrame):
        result_df = result
    elif isinstance(result, pd.Series):
        # Convert series to single-column DataFrame
        col_name = preferred_name or "indicator"
        result_df = pd.DataFrame({col_name: result})
    else:
        return None

    if result_df is None or result_df.empty:
        return None

    cols = list(result_df.columns)

    # If preferred name provided, try to pick matching column
    if preferred_name:
        for c in cols:
            if c.upper() == preferred_name.upper():
                return [c], result_df[[c]]
        for c in cols:
            if preferred_name.upper() in c.upper():
                return [c], result_df[[c]]

    # Otherwise, return all columns
    return cols, result_df


def _apply_obb_indicator(out: pd.DataFrame, name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, str]:
    """Apply a single OpenBB indicator to the dataframe."""
    min_bars = cfg.get("min_bars", 1)
    if len(out) < min_bars:
        return out, False, f"insufficient data (need {min_bars})"

    fn_name = cfg.get("fn")
    params = cfg.get("params", {}) or {}
    if not obb_technical_available or obb is None:
        return out, False, "OpenBB technical extension not available"

    fn = getattr(obb.technical, fn_name, None)
    if fn is None:
        return out, False, f"OpenBB function {fn_name} not found"

    # Build lowercase OHLCV for OpenBB
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in out.columns for col in required_cols):
        return out, False, "Missing OHLCV columns"
    out_lower = out[required_cols].copy()
    out_lower.rename(columns={c: c.lower() for c in required_cols}, inplace=True)

    try:
        result = fn(data=out_lower, **params)
        extracted = _extract_values_from_result(result, preferred_name=name.upper())
        if extracted is None:
            return out, False, "no data returned"
        cols, result_df = extracted

        # Add indicator columns (skip price-like columns)
        price_cols = {"open", "high", "low", "close", "volume"}
        added = 0
        for col in cols:
            if col.lower() in price_cols:
                continue
            series = result_df[col]
            if len(series) == len(out):
                out[col] = series.values
                added += 1
        if added == 0:
            return out, False, "no indicator columns added"
        return out, True, "ok"
    except Exception as e:
        return out, False, f"{type(e).__name__}: {e}"


def _apply_ta_indicator(out: pd.DataFrame, name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, str]:
    """Apply pandas_ta indicator (currently only macd)"""
    min_bars = cfg.get("min_bars", 1)
    if len(out) < min_bars:
        return out, False, f"insufficient data (need {min_bars})"
    if ta is None:
        return out, False, "pandas_ta not available"
    try:
        if name == "macd":
            macd = ta.macd(out["Close"])
            if macd is not None and hasattr(macd, "columns"):
                for c in macd.columns:
                    out[c] = macd[c]
                return out, True, "ok"
            return out, False, "macd returned no columns"
        return out, False, "ta indicator not implemented"
    except Exception as e:
        return out, False, f"{type(e).__name__}: {e}"


def compute_selected_indicators(df: pd.DataFrame, indicators: List[str]) -> Tuple[pd.DataFrame, List[str], List[Dict[str, str]]]:
    """
    Compute only the requested indicators.
    Returns (df_with_indicators, computed_list, skipped_list)
    skipped_list entries: {"name": "...", "reason": "..."}
    """
    if df is None or df.empty:
        raise ValueError("No data to compute indicators.")
    out = df.copy()
    computed: List[str] = []
    skipped: List[Dict[str, str]] = []

    for ind in indicators:
        ind_l = ind.lower()
        cfg = INDICATOR_CATALOG.get(ind_l)
        if cfg is None:
            skipped.append({"name": ind, "reason": "unknown indicator"})
            continue
        kind = cfg.get("kind", "obb")
        min_bars = cfg.get("min_bars", 1)
        if len(out) < min_bars:
            skipped.append({"name": ind, "reason": f"insufficient data (need {min_bars})"})
            continue

        if kind == "obb":
            out, ok, reason = _apply_obb_indicator(out, ind_l, cfg)
        elif kind == "ta":
            out, ok, reason = _apply_ta_indicator(out, ind_l, cfg)
        else:
            ok, reason = False, "unsupported kind"

        if ok:
            computed.append(ind)
        else:
            skipped.append({"name": ind, "reason": reason})

    return out, computed, skipped


def _format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _serialize_df(df: pd.DataFrame) -> list[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    # Reset index to keep date in the payload
    if out.index.name is not None:
        out = out.reset_index()
    # Convert datetime to isoformat
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out.to_dict(orient="records")


def analyze_ticker(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    exchange: str = "NSE",
    use_yfinance_fallback: bool = True,
    indicators: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fetch historical data (Zerodha first, then optional yfinance fallback),
    compute indicators, and return profile + data in serializable form.
    """
    end_dt = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_dt = datetime.fromisoformat(start) if start else end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    start_str = _format_date(start_dt)
    end_str = _format_date(end_dt)

    df = None
    profile = None

    # Zerodha first (NSE/BSE)
    if kite is not None:
        try:
            df = fetch_historical_zerodha(ticker, start_str, end_str, exchange=exchange)
        except Exception as e:
            print(f"Zerodha fetch failed in service: {e}")

    # Optional fallback to yfinance (e.g., for non-Indian tickers)
    if (df is None or df.empty) and use_yfinance_fallback:
        try:
            df = fetch_historical_yfinance(ticker, start_str, end_str)
        except Exception as e:
            print(f"yfinance fetch failed in service: {e}")

    if df is None or df.empty:
        raise ValueError("No historical data available from Zerodha or fallback sources.")

    # Profile: prefer OpenBB if available, else yfinance
    if openbb_available:
        try:
            profile = fetch_profile_openbb(ticker)
        except Exception as e:
            print(f"OpenBB profile fetch failed: {e}")

    if profile is None:
        try:
            profile = fetch_profile_yfinance(ticker)
        except Exception as e:
            print(f"yfinance profile fetch failed: {e}")
            profile = None

    # Compute indicators
    computed_meta: List[str] = []
    skipped_meta: List[Dict[str, str]] = []
    if indicators:
        df_with_ind, computed_meta, skipped_meta = compute_selected_indicators(df, indicators)
    else:
        df_with_ind = compute_indicators(df)
        # No explicit meta when using default compute_indicators

    return {
        "ticker": ticker,
        "exchange": exchange,
        "start": start_str,
        "end": end_str,
        "profile": profile,
        "rows": len(df_with_ind) if df_with_ind is not None else 0,
        "data": _serialize_df(df_with_ind),
        "computed_indicators": computed_meta,
        "skipped_indicators": skipped_meta,
    }


