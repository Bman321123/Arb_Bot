"""
OddsJam-Style Arbitrage Betting Tool
Scans HardRock Bet (via The Odds API) and Kalshi for profitable arbitrage opportunities.
Author: Professional Trading Systems Developer
Version: 3.0
"""

import os
import re
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from rapidfuzz import fuzz, process

# --- CONFIGURATION ---
# ODDS_API_KEY: from env, or use key below if not set
ODDS_API_KEY = (os.environ.get("ODDS_API_KEY") or "e726408c4318682dffc023e260baeb9b").strip()
KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY", "").strip()
# Kalshi public market data (no auth required for markets/orderbook)
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
ODDS_API_URL = "https://api.the-odds-api.com/v4"

# HardRock Bet is in region "us2" per The Odds API
ODDS_REGIONS_FOR_HARDROCK = "us2"

# Trading Parameters
BANKROLL = 1000.00
MIN_PROFIT_PCT = 0.01
MAX_STAKE_PER_ARB = 100.00
KELLY_FRACTION = 0.1  # Conservative Kelly sizing

# Fee Structure
KALSHI_TAKER_FEE = 0.01  # 1% taker fee
HARDROCK_MARGIN = 0.0    # Vig already in odds

# Rate Limiting
ODDS_API_DELAY = 1.0
KALSHI_API_DELAY = 0.5

# Matching Parameters
FUZZY_MATCH_THRESHOLD = 60
MIN_MATCH_CONFIDENCE = 50
# Jaccard (word-set overlap): reject fuzzy match if overlap below this (reduces false alarms)
JACCARD_MIN = 0.2

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arbitrage_scanner.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# --- DATA MODELS ---

class Sport(Enum):
    """Supported sports"""
    NFL = "americanfootball_nfl"
    NBA = "basketball_nba"
    NCAAB = "basketball_ncaab"
    NHL = "icehockey_nhl"
    MLB = "baseball_mlb"


@dataclass
class ArbitrageOpportunity:
    """Represents a profitable arbitrage opportunity"""
    game_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str

    # HardRock side
    hr_team: str
    hr_odds: float
    hr_stake: float
    hr_payout: float
    hr_implied_prob: float

    # Kalshi side
    kalshi_ticker: str
    kalshi_title: str
    kalshi_no_price_cents: int
    kalshi_stake: float
    kalshi_payout: float
    kalshi_implied_prob: float

    # Arbitrage metrics
    total_investment: float
    guaranteed_profit: float
    profit_percentage: float
    match_confidence: int

    timestamp: str = datetime.now().isoformat()


# --- UTILITY FUNCTIONS ---

def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds"""
    if american_odds > 0:
        return (american_odds / 100) + 1
    return (100 / abs(american_odds)) + 1


def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability (0-1)"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds"""
    if decimal_odds >= 2.0:
        return (decimal_odds - 1) * 100
    return -100 / (decimal_odds - 1)


def calculate_kelly_stake(bankroll: float, edge: float, odds: float) -> float:
    """Calculate Kelly Criterion stake size"""
    if edge <= 0 or odds <= 1:
        return 0.0
    kelly = (edge * odds - (1 - edge)) / (odds - 1)
    return max(0, min(bankroll * kelly * KELLY_FRACTION, MAX_STAKE_PER_ARB))


def normalize_team_name(name: str) -> str:
    """Normalize team names for better matching"""
    replacements = {
        "St.": "State",
        "St ": "State ",
        "&": "and",
        "'": "",
    }
    normalized = name.strip()
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized.lower()


def tokenize_to_set(text: str) -> set:
    """Normalize and tokenize into set of words (alphanumeric). Used for Jaccard."""
    normalized = normalize_team_name(text)
    words = re.findall(r"[a-z0-9]+", normalized)
    return set(w for w in words if len(w) > 1)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard = |A ∩ B| / |A ∪ B|. Returns 0 if both sets empty."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def jaccard_string_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two strings (word-set overlap)."""
    return jaccard_similarity(tokenize_to_set(a), tokenize_to_set(b))


def fuzzy_match_teams(
    team_name: str,
    candidates: List[str],
    threshold: int = FUZZY_MATCH_THRESHOLD
) -> Optional[Tuple[str, int]]:
    """
    Find best fuzzy match for team name.
    Returns (matched_string, confidence_score) or None.
    """
    if not candidates:
        return None

    normalized_team = normalize_team_name(team_name)
    normalized_candidates = [normalize_team_name(c) for c in candidates]

    # Create mapping back to original
    candidate_map = dict(zip(normalized_candidates, candidates))

    result = process.extractOne(
        normalized_team,
        normalized_candidates,
        scorer=fuzz.token_sort_ratio
    )

    if result and result[1] >= threshold:
        original_match = candidate_map.get(result[0])
        return (original_match, result[1]) if original_match else None

    return None


# --- API CLIENTS ---

class OddsAPIClient:
    """Client for The Odds API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = ODDS_API_URL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ArbitrageScanner/3.0"
        })

    def get_odds(
        self,
        sport: Sport,
        bookmakers: Optional[List[str]] = None,
        regions: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch odds for a sport. Use regions=us2 for HardRock Bet."""
        url = f"{self.base_url}/sports/{sport.value}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions or ODDS_REGIONS_FOR_HARDROCK,
            "markets": "h2h",
            "oddsFormat": "american",
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            time.sleep(ODDS_API_DELAY)

            data = response.json()
            logger.info(f"Fetched {len(data)} games for {sport.name}")
            return data if isinstance(data, list) else []

        except requests.RequestException as e:
            logger.error(f"Odds API error: {e}")
            return []

    def get_remaining_requests(self) -> Optional[int]:
        """Check remaining API requests"""
        try:
            response = self.session.get(
                f"{self.base_url}/sports",
                params={"apiKey": self.api_key},
                timeout=10
            )
            return int(response.headers.get("x-requests-remaining", 0))
        except Exception:
            return None


class KalshiAPIClient:
    """Client for Kalshi API (public market data; no auth required for markets/orderbook)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = KALSHI_API_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ArbitrageScanner/3.0"
        })

        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def get_markets(
        self,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 1000
    ) -> List[Dict]:
        """Fetch Kalshi markets (public endpoint, no auth required)."""
        url = f"{self.base_url}/markets"
        params = {"status": status, "limit": limit}

        if series_ticker:
            params["series_ticker"] = series_ticker

        try:
            response = self.session.get(url, params=params, timeout=15)

            if not response.ok:
                logger.warning(
                    f"Kalshi markets: {response.status_code} - {response.text[:200]}"
                )
                return []

            time.sleep(KALSHI_API_DELAY)
            data = response.json()
            markets = data.get("markets", [])
            logger.info(f"Fetched {len(markets)} Kalshi markets")
            return markets

        except requests.RequestException as e:
            logger.error(f"Kalshi markets error: {e}")
            return []

    def get_orderbook(self, ticker: str) -> Optional[Dict]:
        """Fetch orderbook for a market (public endpoint)."""
        url = f"{self.base_url}/markets/{ticker}/orderbook"

        try:
            response = self.session.get(url, timeout=10)

            if not response.ok:
                return None

            time.sleep(KALSHI_API_DELAY)
            return response.json()

        except requests.RequestException:
            return None

    def get_no_bin_price_cents(self, orderbook: Dict) -> Optional[int]:
        """
        Return the Buy-It-Now (BIN) price to BUY No contracts, in cents.

        BIN = lowest No ASK = price at which you can immediately buy No.
        Per Kalshi: Best No ASK = 100 - (best Yes BID). API returns bids only.
        Supports orderbook.yes (cents) and orderbook_fp.yes_dollars (dollar strings).
        """
        def extract_yes_bid_prices_cents(ob: Dict) -> List[int]:
            prices: List[int] = []
            # Legacy: orderbook.yes = [[price_cents, quantity], ...]
            for level in ob.get("yes", []):
                if isinstance(level, (list, tuple)) and len(level) >= 1:
                    try:
                        p = int(level[0])
                        if 1 <= p <= 99:
                            prices.append(p)
                    except (ValueError, TypeError):
                        pass
                elif isinstance(level, dict) and "price" in level:
                    try:
                        p = int(level["price"])
                        if 1 <= p <= 99:
                            prices.append(p)
                    except (ValueError, TypeError):
                        pass
            if prices:
                return prices
            # orderbook_fp: yes_dollars = [["0.42", count], ...]
            for level in ob.get("yes_dollars", []):
                if isinstance(level, (list, tuple)) and len(level) >= 1:
                    try:
                        cents = int(round(float(str(level[0]).strip()) * 100))
                        if 1 <= cents <= 99:
                            prices.append(cents)
                    except (ValueError, TypeError):
                        pass
            return prices

        ob_legacy = orderbook.get("orderbook") or {}
        ob_fp = orderbook.get("orderbook_fp") or {}
        yes_prices = extract_yes_bid_prices_cents(ob_legacy) or extract_yes_bid_prices_cents(ob_fp)

        if not yes_prices:
            return None

        best_yes_bid = max(yes_prices)
        no_ask_cents = 100 - best_yes_bid  # BIN for No = lowest No ask

        return no_ask_cents if no_ask_cents > 0 else None


# --- ARBITRAGE CALCULATOR ---

class ArbitrageCalculator:
    """Calculate arbitrage opportunities with fees"""

    @staticmethod
    def calculate(
        hr_american_odds: float,
        kalshi_no_price_cents: int,
        total_investment: float = MAX_STAKE_PER_ARB
    ) -> Optional[Dict[str, float]]:
        """
        Calculate arbitrage opportunity with fees.
        Returns stakes and profit if opportunity exists, else None.
        """
        # Convert to probabilities
        hr_prob = american_to_implied_prob(hr_american_odds)
        kalshi_no_prob = kalshi_no_price_cents / 100.0

        # Apply fees
        hr_prob_with_margin = min(1.0, hr_prob * (1 + HARDROCK_MARGIN))
        kalshi_no_prob_with_fee = min(1.0, kalshi_no_prob * (1 + KALSHI_TAKER_FEE))

        # Check for arbitrage
        total_prob = hr_prob_with_margin + kalshi_no_prob_with_fee

        if total_prob >= 1.0:
            return None  # No arbitrage

        # Calculate stakes (proportional to probabilities)
        hr_stake = total_investment * (hr_prob_with_margin / total_prob)
        kalshi_stake = total_investment * (kalshi_no_prob_with_fee / total_prob)

        # Calculate payouts
        hr_decimal_odds = american_to_decimal(hr_american_odds)
        hr_payout = hr_stake * hr_decimal_odds

        # Kalshi: buying No contracts at price, each pays $1 if No wins
        cost_per_contract = (kalshi_no_price_cents / 100.0) * (1 + KALSHI_TAKER_FEE)
        num_contracts = kalshi_stake / cost_per_contract if cost_per_contract > 0 else 0
        kalshi_payout = num_contracts * 1.0

        # Profit is guaranteed (same in both scenarios)
        profit_if_hr_wins = hr_payout - total_investment
        profit_if_kalshi_wins = kalshi_payout - total_investment
        guaranteed_profit = min(profit_if_hr_wins, profit_if_kalshi_wins)
        profit_pct = (guaranteed_profit / total_investment) * 100

        # Check minimum profit threshold
        if profit_pct < MIN_PROFIT_PCT:
            return None

        return {
            "hr_stake": round(hr_stake, 2),
            "kalshi_stake": round(kalshi_stake, 2),
            "total_investment": round(total_investment, 2),
            "hr_payout": round(hr_payout, 2),
            "kalshi_payout": round(kalshi_payout, 2),
            "guaranteed_profit": round(guaranteed_profit, 2),
            "profit_pct": round(profit_pct, 2),
        }


# --- ARBITRAGE SCANNER ---

class ArbitrageScanner:
    """Main arbitrage scanning engine"""

    def __init__(self):
        self.odds_client = OddsAPIClient(ODDS_API_KEY)
        self.kalshi_client = KalshiAPIClient(KALSHI_API_KEY)
        self.opportunities: List[ArbitrageOpportunity] = []

    def get_kalshi_series_for_sport(self, sport: Sport) -> Optional[str]:
        """Map sport to Kalshi series ticker"""
        mapping = {
            Sport.NCAAB: "KXNCAAMBGAME",  # NCAA Men's Basketball
            Sport.NBA: "KXNBAGAME",        # NBA
            Sport.NFL: "KXNFLGAME",        # NFL
            # Add more as needed
        }
        return mapping.get(sport)

    def scan(self, sport: Sport) -> List[ArbitrageOpportunity]:
        """
        Scan for arbitrage opportunities for a given sport.
        """
        logger.info(f"Starting scan for {sport.name}...")
        opportunities = []

        # Fetch odds: use us2 region for HardRock Bet
        hr_games = self.odds_client.get_odds(
            sport,
            bookmakers=["hardrockbet"],
            regions=ODDS_REGIONS_FOR_HARDROCK,
        )

        series_ticker = self.get_kalshi_series_for_sport(sport)
        if not series_ticker:
            logger.warning(f"No Kalshi series mapping for {sport.name}")
            return opportunities

        kalshi_markets = self.kalshi_client.get_markets(series_ticker=series_ticker)

        if not hr_games or not kalshi_markets:
            logger.info("Insufficient data for arbitrage scan")
            return opportunities

        # Build Kalshi market lookup: match by title (e.g. "Team A vs Team B" or "Team A wins")
        kalshi_lookup = {
            m.get("title", ""): m.get("ticker", "")
            for m in kalshi_markets
            if m.get("title") and m.get("ticker")
        }
        kalshi_titles = list(kalshi_lookup.keys())

        # Scan each HardRock game
        for game in hr_games:
            game_opps = self._scan_game(game, kalshi_titles, kalshi_lookup, sport)
            opportunities.extend(game_opps)

        logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        self.opportunities = opportunities
        return opportunities

    def _scan_game(
        self,
        game: Dict,
        kalshi_titles: List[str],
        kalshi_lookup: Dict[str, str],
        sport: Sport
    ) -> List[ArbitrageOpportunity]:
        """Scan a single game for arbitrage opportunities"""
        opportunities = []

        # Extract game info
        game_id = game.get("id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence_time = game.get("commence_time", "")

        # Get HardRock bookmaker
        bookmakers = game.get("bookmakers", [])
        hr_book = next(
            (b for b in bookmakers if b.get("key") == "hardrockbet"),
            bookmakers[0] if bookmakers else None
        )

        if not hr_book:
            return opportunities

        # Get head-to-head market
        markets = hr_book.get("markets", [])
        h2h_market = next((m for m in markets if m.get("key") == "h2h"), None)

        if not h2h_market:
            return opportunities

        outcomes = h2h_market.get("outcomes", [])

        # Check each outcome for arbitrage
        for outcome in outcomes:
            team_name = outcome.get("name", "")
            hr_odds = outcome.get("price")

            if hr_odds is None:
                continue

            # Match with Kalshi market (fuzzy match on full title, e.g. "Duke wins" vs "Duke")
            match_result = fuzzy_match_teams(team_name, kalshi_titles)
            if not match_result:
                continue

            matched_title, match_confidence = match_result

            # Jaccard gate: require real word overlap to avoid false alarms
            if jaccard_string_similarity(team_name, matched_title) < JACCARD_MIN:
                continue

            # Same-outcome check: HardRock team name must appear in Kalshi title (avoid wrong team)
            team_tokens = tokenize_to_set(team_name)
            title_tokens = tokenize_to_set(matched_title)
            if team_tokens and not (team_tokens & title_tokens):
                continue

            kalshi_ticker = kalshi_lookup.get(matched_title)

            if not kalshi_ticker:
                continue

            # Get Kalshi orderbook
            orderbook = self.kalshi_client.get_orderbook(kalshi_ticker)
            if not orderbook:
                continue

            kalshi_no_price = self.kalshi_client.get_no_bin_price_cents(orderbook)
            if not kalshi_no_price or kalshi_no_price <= 0:
                continue

            # Calculate arbitrage
            arb_calc = ArbitrageCalculator.calculate(
                float(hr_odds),
                kalshi_no_price,
                min(BANKROLL * 0.1, MAX_STAKE_PER_ARB)
            )

            if not arb_calc:
                continue

            # Create opportunity object
            opportunity = ArbitrageOpportunity(
                game_id=game_id,
                sport=sport.name,
                home_team=home_team,
                away_team=away_team,
                commence_time=commence_time,
                hr_team=team_name,
                hr_odds=float(hr_odds),
                hr_stake=arb_calc["hr_stake"],
                hr_payout=arb_calc["hr_payout"],
                hr_implied_prob=american_to_implied_prob(float(hr_odds)),
                kalshi_ticker=kalshi_ticker,
                kalshi_title=matched_title,
                kalshi_no_price_cents=kalshi_no_price,
                kalshi_stake=arb_calc["kalshi_stake"],
                kalshi_payout=arb_calc["kalshi_payout"],
                kalshi_implied_prob=kalshi_no_price / 100.0,
                total_investment=arb_calc["total_investment"],
                guaranteed_profit=arb_calc["guaranteed_profit"],
                profit_percentage=arb_calc["profit_pct"],
                match_confidence=match_confidence
            )

            opportunities.append(opportunity)
            self._log_opportunity(opportunity)

        return opportunities

    def _log_opportunity(self, opp: ArbitrageOpportunity):
        """Log arbitrage opportunity in readable format"""
        logger.info("=" * 80)
        logger.info("ARBITRAGE OPPORTUNITY FOUND")
        logger.info("=" * 80)
        logger.info(f"Game: {opp.home_team} vs {opp.away_team}")
        logger.info(f"Sport: {opp.sport} | Commence: {opp.commence_time}")
        logger.info(f"Match Confidence: {opp.match_confidence}%")
        logger.info("")
        logger.info(f"HardRock Side:")
        logger.info(f"  Team: {opp.hr_team}")
        logger.info(f"  Odds: {opp.hr_odds:+.0f}")
        logger.info(f"  Stake: ${opp.hr_stake:.2f}")
        logger.info(f"  Payout: ${opp.hr_payout:.2f}")
        logger.info("")
        logger.info(f"Kalshi Side:")
        logger.info(f"  Market: {opp.kalshi_title}")
        logger.info(f"  Ticker: {opp.kalshi_ticker}")
        logger.info(f"  No BIN (Ask): {opp.kalshi_no_price_cents}¢")
        logger.info(f"  Stake: ${opp.kalshi_stake:.2f}")
        logger.info(f"  Payout: ${opp.kalshi_payout:.2f}")
        logger.info("")
        logger.info(f"Total Investment: ${opp.total_investment:.2f}")
        logger.info(f"Guaranteed Profit: ${opp.guaranteed_profit:.2f} ({opp.profit_percentage:.2f}%)")
        logger.info("=" * 80)

    def export_to_dict(self) -> List[Dict]:
        """Export opportunities to dictionary format"""
        return [asdict(opp) for opp in self.opportunities]


# --- MAIN EXECUTION ---

def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("OddsJam-Style Arbitrage Scanner v3.0")
    logger.info("=" * 80)

    if not ODDS_API_KEY or ODDS_API_KEY.strip() == "":
        logger.error("Please set ODDS_API_KEY (e.g. export ODDS_API_KEY=your_key or use .env)")
        return

    # Initialize scanner
    scanner = ArbitrageScanner()

    # Check API limits
    remaining = scanner.odds_client.get_remaining_requests()
    if remaining is not None:
        logger.info(f"Odds API requests remaining: {remaining}")

    # Scan for opportunities (default: NCAAB; change Sport.NBA, Sport.NFL, etc. as needed)
    sport = Sport.NCAAB
    opportunities = scanner.scan(sport)

    if opportunities:
        logger.info(f"\n✓ Found {len(opportunities)} profitable arbitrage opportunities!")

        # Sort by profit percentage
        sorted_opps = sorted(
            opportunities,
            key=lambda x: x.profit_percentage,
            reverse=True
        )

        logger.info("\nTop Opportunities:")
        for i, opp in enumerate(sorted_opps[:5], 1):
            logger.info(
                f"{i}. {opp.hr_team} | "
                f"${opp.guaranteed_profit:.2f} ({opp.profit_percentage:.2f}%) | "
                f"Confidence: {opp.match_confidence}%"
            )
    else:
        logger.info("\nNo arbitrage opportunities found in this scan.")

    logger.info("\nScan complete.")


if __name__ == "__main__":
    main()
