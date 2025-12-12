"""
Simple client to call the FastAPI analysis service.

Usage:
    python client.py --ticker RELIANCE --start 2024-11-01 --end 2024-12-10 --exchange NSE
"""

import argparse
import json
import sys

import requests


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
        "--base_url",
        default="http://localhost:8080",
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

    url = f"{args.base_url}/analyze"
    try:
        resp = requests.post(url, json=payload, timeout=60)
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
    except requests.HTTPError as e:
        print(f"HTTP error: {e.response.status_code} {e.response.text}")
    except Exception as e:
        print(f"Error calling API: {e}")


if __name__ == "__main__":
    main()


