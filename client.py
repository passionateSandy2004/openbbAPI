"""
Simple client to call the FastAPI analysis service.

Usage:
    python client.py --ticker RELIANCE --start 2024-11-01 --end 2024-12-10 --exchange NSE
"""

import argparse
import json
import sys

import requests


def _post_json_follow_redirects(url: str, payload: dict, timeout: int = 60, max_hops: int = 3) -> requests.Response:
    """
    Railway often redirects http -> https. Some redirect codes can cause clients
    to switch POST -> GET automatically, which then returns 405.
    We disable auto-redirects and follow redirects manually while preserving POST.
    """
    current = url
    for _ in range(max_hops + 1):
        resp = requests.post(current, json=payload, timeout=timeout, allow_redirects=False)
        if resp.status_code in (301, 302, 303, 307, 308) and "location" in resp.headers:
            current = resp.headers["location"]
            continue
        return resp
    return resp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., RELIANCE)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (optional)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (optional)")
    parser.add_argument("--exchange", default="NSE", help="Exchange (NSE or BSE)")
    parser.add_argument(
        "--use_yfinance_fallback",
        action="store_true",
        default=False,
        help="Use yfinance fallback if Zerodha is unavailable",
    )
    parser.add_argument(
        "--indicators",
        nargs="*",
        help="List of indicators to compute (e.g., rsi atr vwap ichimoku macd). If omitted, default set is used.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        default=False,
        help="Print all rows of data (default prints only first 5)",
    )
    parser.add_argument(
        "--save-json",
        help="Optional path to save full JSON response",
    )
    parser.add_argument(
        "--save-csv",
        help="Optional path to save data rows as CSV",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate an interactive HTML chart (requires plotly).",
    )
    parser.add_argument(
        "--plot-file",
        default="chart.html",
        help="Output HTML file for --plot (default: chart.html)",
    )
    parser.add_argument(
        "--plot-cols",
        nargs="*",
        help="Optional list of columns to plot as overlays (e.g., SMA_20 EMA_20 VWAP). If omitted, auto-detects common indicator columns.",
    )
    parser.add_argument(
        "--base_url",
        default="https://openbbapi-production.up.railway.app",
        help="Base URL of the FastAPI service",
    )
    args = parser.parse_args()

    payload = {
        "ticker": args.ticker,
        "start": args.start,
        "end": args.end,
        "exchange": args.exchange,
        "use_yfinance_fallback": args.use_yfinance_fallback,
        "indicators": args.indicators,
    }

    base = (args.base_url or "").rstrip("/")
    url = f"{base}/analyze"
    try:
        resp = _post_json_follow_redirects(url, payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", []) or []

        # Metadata
        print("Ticker:", data.get("ticker"))
        print("Exchange:", data.get("exchange"))
        print("Date Range:", data.get("start"), "->", data.get("end"))
        print("Rows:", data.get("rows"))
        print("Computed indicators:", data.get("computed_indicators"))
        print("Skipped indicators:", data.get("skipped_indicators"))

        # Profile summary
        profile = data.get("profile")
        if profile:
            print("Profile:", json.dumps(profile, indent=2))

        # Show rows
        if args.show_all:
            print(json.dumps(rows, indent=2))
        else:
            print("First 5 rows:")
            print(json.dumps(rows[:5], indent=2))

        # Save outputs if requested
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved JSON to {args.save_json}")

        if args.save_csv:
            try:
                import pandas as pd

                pd.DataFrame(rows).to_csv(args.save_csv, index=False)
                print(f"Saved CSV to {args.save_csv}")
            except Exception as e:
                print(f"Failed to save CSV: {e}", file=sys.stderr)

        if args.plot:
            try:
                import pandas as pd
                import plotly.graph_objects as go

                df = pd.DataFrame(rows)
                if df.empty:
                    print("No data rows returned; skipping plot.", file=sys.stderr)
                    return

                # Pick a datetime-like x axis
                x_col = None
                for candidate in ["date", "datetime", "time", "index", "Date", "Datetime", "Time", "Index"]:
                    if candidate in df.columns:
                        x_col = candidate
                        break
                if x_col is None:
                    # Try any column that parses as datetime for most rows
                    for c in df.columns:
                        parsed = pd.to_datetime(df[c], errors="coerce")
                        if parsed.notna().sum() >= max(3, int(0.8 * len(df))):
                            df[c] = parsed
                            x_col = c
                            break
                if x_col is None:
                    raise ValueError("Could not find a datetime column in response rows.")

                df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
                df = df.sort_values(x_col)

                has_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])
                fig = go.Figure()

                if has_ohlc:
                    fig.add_trace(
                        go.Candlestick(
                            x=df[x_col],
                            open=df["Open"],
                            high=df["High"],
                            low=df["Low"],
                            close=df["Close"],
                            name="OHLC",
                        )
                    )
                elif "Close" in df.columns:
                    fig.add_trace(go.Scatter(x=df[x_col], y=df["Close"], mode="lines", name="Close"))
                else:
                    raise ValueError("Response rows missing OHLC/Close columns; cannot plot price.")

                # Determine overlay columns
                overlay_cols = list(args.plot_cols) if args.plot_cols else []
                if not overlay_cols:
                    # Auto-detect common indicator columns present in the response
                    prefixes = ("SMA_", "EMA_", "BB", "VWAP", "ATR_", "RSI_", "MACD", "ICH_")
                    for c in df.columns:
                        if c in ("Open", "High", "Low", "Close", "Volume", x_col):
                            continue
                        if c.startswith(prefixes):
                            overlay_cols.append(c)

                # Plot overlays (best-effort; skip non-numeric)
                for c in overlay_cols:
                    if c not in df.columns:
                        continue
                    if not pd.api.types.is_numeric_dtype(df[c]):
                        # Try coercion
                        y = pd.to_numeric(df[c], errors="coerce")
                    else:
                        y = df[c]
                    if y.notna().sum() == 0:
                        continue
                    fig.add_trace(go.Scatter(x=df[x_col], y=y, mode="lines", name=c))

                fig.update_layout(
                    title=f"{data.get('ticker')} ({data.get('exchange')}) {data.get('start')} → {data.get('end')}",
                    xaxis_title="Date",
                    yaxis_title="Price / Indicators",
                    legend_title="Series",
                    template="plotly_white",
                    xaxis_rangeslider_visible=False,
                )
                fig.write_html(args.plot_file, include_plotlyjs="cdn")
                print(f"Saved chart to {args.plot_file}")
            except ImportError:
                print("Plotting requires plotly. Install it with: pip install plotly", file=sys.stderr)
            except Exception as e:
                print(f"Failed to generate plot: {e}", file=sys.stderr)
    except requests.HTTPError as e:
        status = e.response.status_code
        body = e.response.text
        print(f"HTTP error: {status} {body}")
        if status == 405 and base.startswith("http://"):
            print("Hint: Your URL may be redirecting http->https. Try --base_url https://<your-app>.up.railway.app")
    except Exception as e:
        print(f"Error calling API: {e}")


if __name__ == "__main__":
    main()


