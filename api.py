"""
FastAPI interface for the trading analysis backend.

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Example request payload:
{
  "ticker": "RELIANCE",
  "start": "2024-11-01",
  "end": "2024-12-10",
  "exchange": "NSE",
  "use_yfinance_fallback": true,
  "indicators": ["rsi", "atr", "vwap", "ichimoku", "macd"]
}
"""

from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from analysis_service import analyze_ticker

app = FastAPI(title="Trading Analysis Backend", version="1.0.0")


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g., RELIANCE")
    start: Optional[str] = Field(None, description="Start date YYYY-MM-DD (defaults to 90 days lookback)")
    end: Optional[str] = Field(None, description="End date YYYY-MM-DD (defaults to today)")
    exchange: str = Field("NSE", description="Exchange for Zerodha (NSE or BSE)")
    use_yfinance_fallback: bool = Field(
        True, description="Use yfinance fallback if Zerodha data is unavailable"
    )
    indicators: Optional[List[str]] = Field(
        None,
        description="List of indicator names to compute (e.g., ['rsi','atr','bbands']). If omitted, uses default set.",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    try:
        result = analyze_ticker(
            ticker=req.ticker,
            start=req.start,
            end=req.end,
            exchange=req.exchange,
            use_yfinance_fallback=req.use_yfinance_fallback,
            indicators=req.indicators,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

