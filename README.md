# Trading Analysis Backend API

FastAPI-based trading analysis service that fetches stock data from Zerodha KiteConnect API (NSE/BSE) or yfinance fallback, and computes technical indicators using OpenBB extensions.

## Features

- **Data Sources**: Zerodha KiteConnect (NSE/BSE) with yfinance fallback
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, VWAP, Ichimoku, and 30+ more
- **OpenBB Integration**: Uses OpenBB Platform v4 technical extensions when available
- **RESTful API**: FastAPI with automatic OpenAPI documentation

## API Endpoints

### Health Check
```
GET /health
```
Returns: `{"status": "ok"}`

### Analyze Ticker
```
POST /analyze
```

**Request Body:**
```json
{
  "ticker": "RELIANCE",
  "start": "2024-11-01",
  "end": "2024-12-10",
  "exchange": "NSE",
  "use_yfinance_fallback": true,
  "indicators": ["rsi", "atr", "vwap", "ichimoku", "macd"]
}
```

**Response:**
```json
{
  "ticker": "RELIANCE",
  "exchange": "NSE",
  "start": "2024-11-01",
  "end": "2024-12-10",
  "profile": {...},
  "rows": 237,
  "data": [...],
  "computed_indicators": ["rsi", "atr", "vwap", "ichimoku", "macd"],
  "skipped_indicators": []
}
```

## Available Indicators

- `sma`, `ema`, `rsi`, `macd`, `bbands`
- `atr`, `adx`, `obv`, `vwap`, `kc` (Keltner Channels)
- `hma`, `wma`, `fib`, `demark`, `aroon`
- `fisher`, `cci`, `donchian`, `ichimoku`
- `stoch`, `adosc`, `ad`, `cones`, `zlma`
- And more...

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Installation

1. Clone the repository and navigate to BBbackend:
```bash
cd BBbackend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your Zerodha credentials
```

5. Run the server:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

6. Access the API:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Testing the API

Use the included client:
```bash
python client.py --ticker RELIANCE --start 2024-11-01 --end 2024-12-10 --exchange NSE
```

Or use curl:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE",
    "start": "2024-11-01",
    "end": "2024-12-10",
    "exchange": "NSE",
    "indicators": ["rsi", "macd"]
  }'
```

## Railway Deployment

### Option 1: Deploy via Railway CLI

1. Install Railway CLI:
```bash
npm i -g @railway/cli
```

2. Login to Railway:
```bash
railway login
```

3. Initialize and deploy:
```bash
cd BBbackend
railway init
railway up
```

4. Set environment variables in Railway dashboard:
   - `KITE_API_KEY` (optional)
   - `KITE_ACCESS_TOKEN` (optional)
   - `PORT` is automatically set by Railway

### Option 2: Deploy via Railway Dashboard

1. Go to [Railway](https://railway.app)
2. Create a new project
3. Connect your GitHub repository
4. Select the `BBbackend` directory as the root
5. Railway will automatically detect the Dockerfile
6. Add environment variables in the Variables tab
7. Deploy!

### Option 3: Deploy with Docker

1. Build the Docker image:
```bash
docker build -t trading-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -e KITE_API_KEY=your_key \
  -e KITE_ACCESS_TOKEN=your_token \
  trading-api
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `KITE_API_KEY` | Zerodha KiteConnect API key | No | Uses test credentials |
| `KITE_ACCESS_TOKEN` | Zerodha KiteConnect access token | No | Uses test credentials |
| `PORT` | Server port | No | 8000 (Railway sets automatically) |

## Project Structure

```
BBbackend/
├── api.py                 # FastAPI application
├── analysis_service.py    # Business logic for analysis
├── main.py                # Data fetching and indicator computation
├── client.py              # Test client (not needed for deployment)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── railway.json           # Railway deployment config
├── .env.example           # Environment variables template
└── README.md             # This file
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### OpenBB Installation Issues
If OpenBB fails to install, the API will fall back to pandas_ta for technical indicators. This is handled gracefully.

### Zerodha Connection Issues
If Zerodha API fails, set `use_yfinance_fallback: true` in your request to use yfinance as fallback.

### Port Issues on Railway
Railway automatically sets the `PORT` environment variable. The Dockerfile and railway.json handle this automatically.

## License

This project is for educational and personal use.

