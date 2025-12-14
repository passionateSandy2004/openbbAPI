"""
OpenBB-compatible dashboard script (Python)

What this does:
- Fetches NSE historical data from Zerodha KiteConnect API
- Uses OpenBB extensions (openbb-technical) to analyze the data with technical indicators
- Falls back to pandas_ta or yfinance if OpenBB extensions or Zerodha unavailable
- Computes common technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Saves CSVs and a sample Plotly chart (so you can view in browser)

How to use:
1. Install dependencies (recommended in a virtualenv):
   pip install openbb openbb-technical
   pip install kiteconnect yfinance pandas pandas_ta plotly

2. Set environment variables for Zerodha API (optional):
   export KITE_API_KEY="your_api_key"
   export KITE_ACCESS_TOKEN="your_access_token"
   
   Or use default credentials (hardcoded for testing)

3. Run this script: python main.py --ticker RELIANCE --start 2023-01-01 --end 2024-12-31

Notes:
- Priority: Zerodha (NSE) → OpenBB Extensions Analysis → pandas_ta → yfinance
- For Zerodha: Use NSE/BSE ticker symbols (e.g., RELIANCE, TCS, INFY)
- OpenBB extensions are used for analysis, not data fetching
- NSE data from Zerodha is analyzed by OpenBB technical extensions
- This is intended as a starting point for building an Overview/Technical/Financials dashboard similar to OpenBB Pro using open-source tools.

"""

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd

# Try to import OpenBB v4 Platform and extensions
obb = None  # always define for importers (analysis_service imports this)
obb_available = False
obb_technical_available = False
try:
    from openbb import obb
    obb_available = True
    # Check if technical extension is available
    try:
        # Try to access technical module to verify extension is installed
        _ = obb.technical
        obb_technical_available = True
        print("OpenBB Platform v4 with technical extension found.")
    except (AttributeError, ImportError):
        obb_technical_available = False
        print("OpenBB Platform v4 found, but technical extension not available.")
except Exception:
    obb_available = False
    obb_technical_available = False
    print("OpenBB Platform v4 not found -- will use fallback libraries.")

# Try to import legacy OpenBB SDK (for backward compatibility)
openbb_available = False
try:
    from openbb_terminal.sdk import openbb  # type: ignore
    openbb_available = True
    if not obb_available:
        print("Legacy OpenBB SDK found.")
except Exception:
    openbb_available = False

# Fallback libraries
try:
    from kiteconnect import KiteConnect
    kiteconnect_available = True
except Exception:
    KiteConnect = None
    kiteconnect_available = False

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import pandas_ta as ta
except Exception:
    ta = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# Zerodha KiteConnect initialization (optional)
kite = None
if kiteconnect_available:
    try:
        api_key = os.getenv('KITE_API_KEY', '6ej0i7fspon9w0yv')  # Default from your notebook
        access_token = os.getenv('KITE_ACCESS_TOKEN', 'U4SNuRV6Q2HjQ4ZZHpegjKTEIFkVJBGG')  # Default from your notebook
        if api_key and access_token:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            print("Zerodha KiteConnect initialized.")
    except Exception as e:
        print(f"Zerodha KiteConnect initialization failed: {e}")
        kite = None


def fetch_profile_openbb(ticker: str):
    """Placeholder: how you'd fetch profile from OpenBB SDK if available.
    Replace this body with the real openbb call if you have the SDK installed.
    e.g.: profile = openbb.stocks.profile(ticker)
    """
    try:
        # This is SDK-specific and may need adjustments.
        profile = openbb.stocks.profile(ticker)
        return profile
    except Exception as e:
        print("OpenBB profile fetch failed:", e)
        return None


def fetch_historical_openbb(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Placeholder for OpenBB historical price fetch. Replace with SDK call.
    Example (pseudocode):
        df = openbb.stocks.price.historical(ticker, start_date=start, end_date=end)
    """
    try:
        df = openbb.stocks.price.historical(ticker, start_date=start, end_date=end)
        return df
    except Exception as e:
        print("OpenBB historical fetch failed:", e)
        return None


def fetch_profile_yfinance(ticker: str):
    if yf is None:
        raise ImportError("yfinance not installed")
    t = yf.Ticker(ticker)
    info = t.get_info()
    # pick a subset usable for a profile box
    profile = {
        "longName": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "fullTimeEmployees": info.get("fullTimeEmployees"),
        "exchange": info.get("exchange") or info.get("exchangeTimezoneName"),
        "country": info.get("country"),
        "shortName": info.get("shortName"),
        "longBusinessSummary": info.get("longBusinessSummary"),
    }
    return profile


def find_instrument_token(ticker: str, exchange: str = "NSE") -> int:
    """Find instrument token for a given ticker symbol from Zerodha."""
    if kite is None:
        raise ImportError("KiteConnect not initialized")
    
    try:
        # Get all instruments for the exchange
        instruments = kite.instruments(exchange)
        df_instruments = pd.DataFrame(instruments)
        
        # Search by tradingsymbol (case-insensitive)
        matching = df_instruments[
            df_instruments['tradingsymbol'].str.upper() == ticker.upper()
        ]
        
        if not matching.empty:
            token = matching.iloc[0]['instrument_token']
            print(f"Found instrument token {token} for {ticker} on {exchange}")
            return token
        else:
            raise ValueError(f"Instrument {ticker} not found on {exchange}")
    except Exception as e:
        raise ValueError(f"Error finding instrument token for {ticker}: {e}")


def fetch_historical_zerodha(ticker: str, start: str, end: str, exchange: str = "NSE") -> pd.DataFrame:
    """Fetch historical data from Zerodha KiteConnect API."""
    if kite is None:
        raise ImportError("KiteConnect not initialized")
    
    try:
        # Find instrument token
        instrument_token = find_instrument_token(ticker, exchange=exchange)
        
        # Parse dates
        from_date = datetime.strptime(start, '%Y-%m-%d')
        to_date = datetime.strptime(end, '%Y-%m-%d')
        
        # Determine interval based on date range
        days_diff = (to_date - from_date).days
        if days_diff <= 60:
            interval = "day"  # Use day for short periods
        else:
            interval = "day"  # Can also use "day" for longer periods
        
        # Fetch historical data
        historical_data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=0,
            oi=0
        )
        
        if not historical_data:
            warnings.warn("Zerodha returned empty historical data")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        if df.empty:
            warnings.warn("Zerodha returned empty DataFrame")
            return pd.DataFrame()
        
        # Rename columns to match expected format (Open, High, Low, Close, Volume)
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        # Only rename columns that exist
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # Set date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif df.index.name == 'date' or df.index.dtype == 'datetime64[ns]':
            # Date is already the index
            df.index = pd.to_datetime(df.index)
        
        # Select only OHLCV columns (if they exist)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in df.columns]
        if len(available_cols) < 4:  # At least need OHLC
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        df = df[available_cols]
        
        if df.empty:
            warnings.warn("Downloaded historical data is empty after processing")
        
        return df
        
    except Exception as e:
        print(f"Zerodha historical fetch failed: {e}")
        return None


def apply_openbb_technical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Apply OpenBB technical analysis extensions to DataFrame.
    
    Uses OpenBB v4 Platform technical extension functions that accept DataFrames.
    Falls back to pandas_ta if OpenBB extensions not available.
    """
    if df is None or df.empty:
        return df
    
    out = df.copy()
    data_length = len(out)
    
    # Check if we have enough data points (need at least 50 for some indicators)
    if data_length < 20:
        print(f"Warning: Data length ({data_length}) is too short for technical analysis. Need at least 20 data points.")
        return out
    
    # Try OpenBB technical extensions first
    if obb_technical_available:
        try:
            # Check available methods on obb.technical (for debugging)
            tech_methods = [method for method in dir(obb.technical) if not method.startswith('_')]
            # Uncomment for debugging: print(f"Available OpenBB technical methods: {tech_methods}")
            
            # Create a lowercase version for OpenBB (it expects lowercase column names)
            # Rename columns to lowercase for OpenBB functions
            # Only use OHLCV columns, don't copy indicator columns
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            out_lower = out[ohlcv_cols].copy()
            column_mapping = {col: col.lower() for col in out_lower.columns}
            out_lower.rename(columns=column_mapping, inplace=True)
            
            # Helper function to extract values from OpenBB result
            def extract_values(result, default_col_name=None):
                """Extract values from OpenBB result (handles OBBject, DataFrame, Series)"""
                if result is None:
                    return None
                
                result_df = None
                # Check if it's an OBBject with to_df method
                if hasattr(result, 'to_df'):
                    try:
                        result_df = result.to_df()
                    except Exception as e:
                        print(f"Error converting OBBject to DataFrame: {e}")
                        return None
                # Check if it's already a DataFrame
                elif isinstance(result, pd.DataFrame):
                    result_df = result
                # Check if it's a Series
                elif isinstance(result, pd.Series):
                    return result.values
                else:
                    return None
                
                if result_df is None or result_df.empty:
                    return None
                
                # Try to find the right column - check multiple possible names
                if default_col_name:
                    # Try exact match
                    if default_col_name in result_df.columns:
                        return result_df[default_col_name].values
                    # Try case-insensitive match
                    for col in result_df.columns:
                        if col.upper() == default_col_name.upper():
                            return result_df[col].values
                    # Try partial match (e.g., 'SMA' in 'close_SMA_20')
                    for col in result_df.columns:
                        if default_col_name.upper() in col.upper():
                            return result_df[col].values
                
                # If no match found, return the last column (usually the indicator, not the price)
                # Skip price columns (open, high, low, close, volume)
                price_cols = ['open', 'high', 'low', 'close', 'volume']
                indicator_cols = [col for col in result_df.columns if col.lower() not in price_cols]
                if indicator_cols:
                    return result_df[indicator_cols[-1]].values
                
                # Fallback to last column
                if len(result_df.columns) > 0:
                    return result_df.iloc[:, -1].values
                
                return None
            
            # SMA - Simple Moving Average
            if data_length >= 20:
                try:
                    if hasattr(obb.technical, 'sma'):
                        result = obb.technical.sma(data=out_lower, length=20)
                        values = extract_values(result, 'SMA')
                        if values is not None and len(values) == len(out):
                            out['SMA_20'] = values
                        else:
                            print(f"OpenBB SMA: Expected {len(out)} values, got {len(values) if values is not None else 0}")
                except Exception as e:
                    print(f"OpenBB SMA failed: {e}")
            
            # RSI - needs at least 14 data points
            if data_length >= 14:
                try:
                    if hasattr(obb.technical, 'rsi'):
                        result = obb.technical.rsi(data=out_lower, length=14)
                        values = extract_values(result, 'RSI')
                        if values is not None and len(values) == len(out):
                            out['RSI_14'] = values
                        else:
                            print(f"OpenBB RSI: Expected {len(out)} values, got {len(values) if values is not None else 0}")
                except Exception as e:
                    print(f"OpenBB RSI failed: {e}")
            
            # MACD - skipping OpenBB MACD due to duplicate column issues; will fall back to pandas_ta
            # (pandas_ta will compute MACD later if missing)
            
            # Bollinger Bands
            if data_length >= 20:
                try:
                    if hasattr(obb.technical, 'bbands'):
                        result = obb.technical.bbands(data=out_lower, length=20)
                        if result is not None:
                            result_df = None
                            if hasattr(result, 'to_df'):
                                try:
                                    result_df = result.to_df()
                                except Exception as e:
                                    print(f"BBands to_df error: {e}")
                            elif isinstance(result, pd.DataFrame):
                                result_df = result
                            
                            if result_df is not None and isinstance(result_df, pd.DataFrame):
                                # Only add indicator columns, skip price columns
                                price_cols = ['open', 'high', 'low', 'close', 'volume']
                                for col in result_df.columns:
                                    if col.lower() not in price_cols:
                                        if len(result_df[col].values) == len(out):
                                            out[col] = result_df[col].values
                except Exception as e:
                    print(f"OpenBB Bollinger Bands failed: {e}")
            
            # EMA - needs at least 20 data points
            if data_length >= 20:
                try:
                    if hasattr(obb.technical, 'ema'):
                        result = obb.technical.ema(data=out_lower, length=20)
                        values = extract_values(result, 'EMA')
                        if values is not None and len(values) == len(out):
                            out['EMA_20'] = values
                        else:
                            print(f"OpenBB EMA: Expected {len(out)} values, got {len(values) if values is not None else 0}")
                except Exception as e:
                    print(f"OpenBB EMA failed: {e}")
            
            # Check if any indicators were successfully added
            indicators_added = any(col in out.columns for col in ['SMA_20', 'EMA_20', 'RSI_14', 'MACD'])
            if indicators_added:
                print("Applied OpenBB technical analysis extensions.")
            else:
                print("OpenBB technical analysis attempted but no indicators were added.")
            
            return out
            
        except Exception as e:
            print(f"OpenBB technical analysis failed, falling back to pandas_ta: {e}")
            # Fall through to pandas_ta
    
    # If OpenBB extensions not available or failed, return unchanged
    # (pandas_ta will be applied in compute_indicators)
    return out


def fetch_historical_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance not installed")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        warnings.warn("Downloaded historical data is empty")
    df.index = pd.to_datetime(df.index)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a standard set of indicators and append to dataframe.
    
    Priority: OpenBB extensions > pandas_ta > manual computation
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # First try OpenBB technical extensions
    if obb_technical_available:
        out = apply_openbb_technical_analysis(out)
    
    # Fill in missing indicators with pandas_ta or manual computation
    # Use pandas_ta if available
    if ta is not None:
        try:
            if 'SMA_20' not in out.columns:
                out['SMA_20'] = ta.sma(out['Close'], length=20)
            if 'EMA_20' not in out.columns:
                out['EMA_20'] = ta.ema(out['Close'], length=20)
            if 'RSI_14' not in out.columns:
                out['RSI_14'] = ta.rsi(out['Close'], length=14)
            if not any('MACD' in col for col in out.columns):
                macd = ta.macd(out['Close'])
                # macd returns multiple columns
                if macd is not None and hasattr(macd, 'columns'):
                    for c in macd.columns:
                        out[c] = macd[c]
            if not any('BB' in col or 'BB_' in col for col in out.columns):
                bbands = ta.bbands(out['Close'], length=20)
                if bbands is not None and hasattr(bbands, 'columns'):
                    for c in bbands.columns:
                        out[c] = bbands[c]
        except Exception as e:
            print('pandas_ta failed to compute indicators:', e)
    else:
        # Manual simple implementations
        out['SMA_20'] = out['Close'].rolling(window=20).mean()
        out['EMA_20'] = out['Close'].ewm(span=20, adjust=False).mean()
        # RSI
        delta = out['Close'].diff()
        gain = delta.clip(lower=0).fillna(0)
        loss = -1 * delta.clip(upper=0).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss.replace(0, 1e-9))
        out['RSI_14'] = 100 - (100 / (1 + rs))
        # MACD
        ema12 = out['Close'].ewm(span=12, adjust=False).mean()
        ema26 = out['Close'].ewm(span=26, adjust=False).mean()
        out['MACD'] = ema12 - ema26
        out['MACD_Signal'] = out['MACD'].ewm(span=9, adjust=False).mean()
        # Bollinger Bands
        m = out['Close'].rolling(window=20).mean()
        s = out['Close'].rolling(window=20).std()
        out['BB_upper'] = m + 2 * s
        out['BB_lower'] = m - 2 * s

    return out


def save_outputs(ticker: str, profile: dict, df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if profile is not None:
        pd.Series(profile).to_json(os.path.join(out_dir, f"{ticker}_profile.json"), orient='index')
    if df is not None and not df.empty:
        df.to_csv(os.path.join(out_dir, f"{ticker}_historical_with_indicators.csv"))
    print(f"Saved outputs to {out_dir}")


def plot_ohlc_with_indicators(df: pd.DataFrame, ticker: str, filename: str):
    if go is None:
        print("Plotly not installed; skipping chart generation")
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name=f"{ticker} OHLC"
    ))
    # add SMA 20 if present
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'))
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20'))

    fig.update_layout(title=f"{ticker} OHLC with indicators", xaxis_rangeslider_visible=False)
    fig.write_html(filename)
    print(f"Chart written to {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True, help='Ticker symbol (e.g., RELIANCE, TCS for Zerodha; AMZN for yfinance)')
    parser.add_argument('--start', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--out', default='output', help='Output directory')
    parser.add_argument('--exchange', default='NSE', help='Exchange for Zerodha (NSE or BSE)')
    args = parser.parse_args()

    ticker = args.ticker
    start = args.start
    end = args.end
    exchange = args.exchange

    profile = None
    df = None

    # Priority: Zerodha (NSE) → OpenBB Extensions Analysis → pandas_ta → yfinance
    # Fetch from Zerodha first (NSE data)
    if kite is not None:
        try:
            print("Fetching historical data via Zerodha KiteConnect (NSE)...")
            df = fetch_historical_zerodha(ticker, start, end, exchange=exchange)
            if df is not None and not df.empty:
                print("Data fetched successfully from Zerodha (NSE)")
        except Exception as e:
            print(f"Zerodha path failed: {e}")

    # Fall back to yfinance if Zerodha failed
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        print("Fetching historical data via yfinance fallback...")
        try:
            df = fetch_historical_yfinance(ticker, start, end)
        except Exception as e:
            print(f"yfinance fallback also failed: {e}")
            raise ValueError("All data sources failed. Please check your configuration.")

    # Try to get profile from OpenBB or yfinance
    if profile is None:
        if openbb_available:
            try:
                profile = fetch_profile_openbb(ticker)
            except Exception as e:
                print(f"OpenBB profile fetch failed: {e}")

    if profile is None:
        try:
            profile = fetch_profile_yfinance(ticker)
        except Exception as e:
            print('Profile fetch fallback failed:', e)

    df_with_ind = compute_indicators(df)
    save_outputs(ticker, profile, df_with_ind, args.out)
    plot_ohlc_with_indicators(df_with_ind, ticker, os.path.join(args.out, f"{ticker}_chart.html"))


if __name__ == '__main__':
    main()
