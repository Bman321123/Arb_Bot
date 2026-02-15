# OddsJam-Style Arbitrage Scanner

Scans **HardRock Bet** (via The Odds API) and **Kalshi** for profitable arbitrage opportunities.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or: python3 -m pip install -r requirements.txt
   ```

2. **Set your API key**
   - Get an API key from [The Odds API](https://the-odds-api.com/).
   - Either:
     - Copy `.env.example` to `.env` and set `ODDS_API_KEY=your_key`, or
     - Export: `export ODDS_API_KEY=your_key`
   - Kalshi markets/orderbook are public; `KALSHI_API_KEY` is optional (for future trading).

## Run

```bash
python arbitrage_scanner.py
```

By default it scans **NCAAB** (college basketball). To scan another sport, edit `main()` in `arbitrage_scanner.py` and change:

```python
sport = Sport.NCAAB   # or Sport.NBA, Sport.NFL, Sport.NHL, Sport.MLB
```

## Fixes applied (v3.0)

- **Odds API**: Uses region `us2` so HardRock Bet (`hardrockbet`) is included in results.
- **Kalshi**: Keeps `api.elections.kalshi.com` for public market/orderbook data; orderbook parsing uses `orderbook.orderbook.yes` as `[[price, quantity], ...]`.
- **Config**: API keys from environment only; optional `.env` via `python-dotenv`.
- **Safety**: No default API key in code; clear error if `ODDS_API_KEY` is missing.

## Output

- Logs to console and `arbitrage_scanner.log`.
- Reports arbitrage opportunities with stakes, payouts, and profit %.
