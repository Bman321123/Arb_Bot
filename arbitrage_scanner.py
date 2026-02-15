"""
Sportsbook-vs-Sportsbook Arbitrage Scanner
Uses The Odds API to fetch odds from multiple books (DraftKings, FanDuel, HardRock, BetMGM, Kalshi)
and finds arbitrage opportunities between them (best odds per side across books).
Version: 4.0
"""

import os
import re
import time
import logging
from datetime import datetime, timezone
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
# Multi-book scan: top books + regions (us, us2, us_ex for Kalshi exchange)
ODDS_BOOKMAKERS = ["draftkings", "fanduel", "hardrockbet", "betmgm", "kalshi"]
ODDS_REGIONS_MULTI = "us,us2,us_ex"

# Trading Parameters
BANKROLL = 1000.00
MIN_PROFIT_PCT = 1.0  # Minimum 1% profit to report an arb (profit_pct is in percent, e.g. 1.0 = 1%)
MAX_STAKE_PER_ARB = 1000.00
KELLY_FRACTION = 0.1  # Conservative Kelly sizing

# Fee Structure
KALSHI_TAKER_FEE = 0.01  # 1% taker fee
HARDROCK_MARGIN = 0.0    # Vig already in odds

# Rate Limiting
ODDS_API_DELAY = 1.0
KALSHI_API_DELAY = 0.5

# Continuous scan: seconds between full multi-sport scans (0 = start next cycle as soon as current one finishes).
# Odds API allows 30 req/sec; we make 1 request per sport with ODDS_API_DELAY between them, so we stay under limit.
SCAN_INTERVAL_SEC = 15

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


# Sports to scan in one run (one Odds API request per sport)
SPORTS_TO_SCAN: List[Sport] = [
    Sport.NFL,
    Sport.NBA,
    Sport.NCAAB,
    Sport.NHL,
    Sport.MLB,
]


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


@dataclass
class BookVsBookOpportunity:
    """Sportsbook-vs-sportsbook arbitrage (no Kalshi API)."""
    game_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str
    # Side 1 (e.g. home)
    side1_team: str
    side1_book: str
    side1_odds: float
    side1_stake: float
    side1_payout: float
    # Side 2 (e.g. away)
    side2_team: str
    side2_book: str
    side2_odds: float
    side2_stake: float
    side2_payout: float
    total_investment: float
    guaranteed_profit: float
    profit_pct: float
    timestamp: str = datetime.now().isoformat()


@dataclass
class TotalsOpportunity:
    """Over/Under (totals) arbitrage between sportsbooks."""
    game_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str
    point: float  # e.g. 234.5
    over_book: str
    over_odds: float
    over_stake: float
    over_payout: float
    under_book: str
    under_odds: float
    under_stake: float
    under_payout: float
    total_investment: float
    guaranteed_profit: float
    profit_pct: float
    timestamp: str = datetime.now().isoformat()


@dataclass
class SpreadsOpportunity:
    """Point spread (handicap) arbitrage between sportsbooks."""
    game_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str
    home_point: float   # e.g. -3.5
    away_point: float   # e.g. +3.5
    home_book: str
    home_odds: float
    home_stake: float
    home_payout: float
    away_book: str
    away_odds: float
    away_stake: float
    away_payout: float
    total_investment: float
    guaranteed_profit: float
    profit_pct: float
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


def align_outcomes_to_home_away(
    home_team: str,
    away_team: str,
    outcomes: List[Dict],
) -> Optional[Tuple[float, float]]:
    """
    Map a book's h2h outcomes to (home_odds, away_odds) using canonical game names.
    outcomes = [{"name": "Team A", "price": -110}, {"name": "Team B", "price": 100}]
    Returns (home_odds, away_odds) or None if alignment fails.
    """
    if not outcomes or len(outcomes) != 2:
        return None
    candidates = [home_team, away_team]
    matched: Dict[str, float] = {}  # "home" or "away" -> american odds
    for out in outcomes:
        name = out.get("name", "")
        price = out.get("price")
        if price is None:
            return None
        result = fuzzy_match_teams(name, candidates, threshold=50)
        if not result:
            return None
        which_team, _ = result
        if which_team == home_team:
            matched["home"] = float(price)
        else:
            matched["away"] = float(price)
    if "home" in matched and "away" in matched:
        return (matched["home"], matched["away"])
    return None


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
        markets: str = "h2h",
    ) -> List[Dict]:
        """Fetch odds for a sport. markets: 'h2h', 'totals', or 'h2h,totals' (each market costs quota)."""
        url = f"{self.base_url}/sports/{sport.value}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions or ODDS_REGIONS_FOR_HARDROCK,
            "markets": markets,
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
        """Check remaining API requests (legacy). Prefer get_usage() for full picture."""
        usage = self.get_usage()
        return usage.get("remaining") if usage else None

    def get_usage(self) -> Optional[Dict[str, int]]:
        """
        Get Odds API usage for current period (e.g. month).
        GET /sports does not count against quota. Returns dict with:
        - remaining: requests left this period
        - used: requests used this period
        """
        try:
            response = self.session.get(
                f"{self.base_url}/sports",
                params={"apiKey": self.api_key},
                timeout=10
            )
            remaining = response.headers.get("x-requests-remaining")
            used = response.headers.get("x-requests-used")
            out = {}
            if remaining is not None:
                try:
                    out["remaining"] = int(remaining)
                except ValueError:
                    pass
            if used is not None:
                try:
                    out["used"] = int(used)
                except ValueError:
                    pass
            return out if out else None
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

    @staticmethod
    def calculate_books_vs_books(
        odds_side1: float,
        odds_side2: float,
        total_investment: float = MAX_STAKE_PER_ARB,
    ) -> Optional[Dict[str, float]]:
        """
        Two-way arb between sportsbooks: best odds on side1 and best on side2.
        Returns stakes and profit if impl_1 + impl_2 < 1, else None.
        """
        impl_1 = american_to_implied_prob(odds_side1)
        impl_2 = american_to_implied_prob(odds_side2)
        if impl_1 + impl_2 >= 1.0:
            return None
        total_impl = impl_1 + impl_2
        stake1 = total_investment * (impl_1 / total_impl)
        stake2 = total_investment * (impl_2 / total_impl)
        payout1 = stake1 * american_to_decimal(odds_side1)
        payout2 = stake2 * american_to_decimal(odds_side2)
        guaranteed_profit = min(payout1, payout2) - total_investment
        profit_pct = (guaranteed_profit / total_investment) * 100
        if profit_pct < MIN_PROFIT_PCT:
            return None
        return {
            "side1_stake": round(stake1, 2),
            "side2_stake": round(stake2, 2),
            "side1_payout": round(payout1, 2),
            "side2_payout": round(payout2, 2),
            "total_investment": round(total_investment, 2),
            "guaranteed_profit": round(guaranteed_profit, 2),
            "profit_pct": round(profit_pct, 2),
        }


def is_live(commence_time: str) -> bool:
    """True if game has already started (commence_time in the past, UTC)."""
    if not commence_time:
        return False
    try:
        start = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return start <= datetime.now(timezone.utc)
    except (ValueError, TypeError):
        return False


def format_bet_slip(opp: BookVsBookOpportunity, number: Optional[int] = None, live: Optional[bool] = None) -> str:
    """
    Format one arbitrage opportunity as an easy-to-read bet slip:
    - The two sportsbooks
    - Recommended bet amount per side, total cost
    - Payout if each side wins
    - Guaranteed profit
    """
    title = f"  ARB #{number}  " if number is not None else "  ARBITRAGE BET SLIP  "
    if live is not None:
        title = title.strip() + f"  —  {'LIVE' if live else 'PREGAME'}  "
    lines = [
        "",
        "=" * 60,
        title.center(60),
        "=" * 60,
        f"  {opp.sport}  ·  {opp.home_team}  vs  {opp.away_team}",
        f"  Game time: {opp.commence_time}",
        "",
        "  SPORTSBOOKS USED:",
        f"    1) {opp.side1_book.upper()}",
        f"    2) {opp.side2_book.upper()}",
        "",
        "  PLACE THESE TWO BETS:",
        "",
        f"  BET 1 — {opp.side1_book.upper()}",
        f"    · Bet ${opp.side1_stake:.2f} on: {opp.side1_team}",
        f"    · Odds: {opp.side1_odds:+.0f}",
        f"    · If this wins, you get: ${opp.side1_payout:.2f}",
        "",
        f"  BET 2 — {opp.side2_book.upper()}",
        f"    · Bet ${opp.side2_stake:.2f} on: {opp.side2_team}",
        f"    · Odds: {opp.side2_odds:+.0f}",
        f"    · If this wins, you get: ${opp.side2_payout:.2f}",
        "",
        "  SUMMARY:",
        f"    Total cost (both bets):  ${opp.total_investment:.2f}",
        f"    If {opp.side1_team} wins:  you receive ${opp.side1_payout:.2f}",
        f"    If {opp.side2_team} wins:  you receive ${opp.side2_payout:.2f}",
        f"    Guaranteed profit:       ${opp.guaranteed_profit:.2f}  ({opp.profit_pct:.2f}%)",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_totals_bet_slip(opp: TotalsOpportunity, number: Optional[int] = None, live: Optional[bool] = None) -> str:
    """Format an over/under arbitrage opportunity as a bet slip."""
    title = f"  TOTALS ARB #{number}  " if number is not None else "  OVER/UNDER ARBITRAGE  "
    if live is not None:
        title = title.strip() + f"  —  {'LIVE' if live else 'PREGAME'}  "
    lines = [
        "",
        "=" * 60,
        title.center(60),
        "=" * 60,
        f"  {opp.sport}  ·  {opp.home_team}  vs  {opp.away_team}",
        f"  Total points line: {opp.point}",
        f"  Game time: {opp.commence_time}",
        "",
        "  SPORTSBOOKS USED:",
        f"    1) {opp.over_book.upper()} (Over)",
        f"    2) {opp.under_book.upper()} (Under)",
        "",
        "  PLACE THESE TWO BETS:",
        "",
        f"  BET 1 — {opp.over_book.upper()} (OVER {opp.point})",
        f"    · Bet ${opp.over_stake:.2f} on: Over {opp.point}",
        f"    · Odds: {opp.over_odds:+.0f}",
        f"    · If Over hits, you get: ${opp.over_payout:.2f}",
        "",
        f"  BET 2 — {opp.under_book.upper()} (UNDER {opp.point})",
        f"    · Bet ${opp.under_stake:.2f} on: Under {opp.point}",
        f"    · Odds: {opp.under_odds:+.0f}",
        f"    · If Under hits, you get: ${opp.under_payout:.2f}",
        "",
        "  SUMMARY:",
        f"    Total cost (both bets):  ${opp.total_investment:.2f}",
        f"    If Over {opp.point}:  you receive ${opp.over_payout:.2f}",
        f"    If Under {opp.point}:  you receive ${opp.under_payout:.2f}",
        f"    Guaranteed profit:       ${opp.guaranteed_profit:.2f}  ({opp.profit_pct:.2f}%)",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_spreads_bet_slip(opp: SpreadsOpportunity, number: Optional[int] = None, live: Optional[bool] = None) -> str:
    """Format a point-spread arbitrage opportunity as a bet slip."""
    title = f"  SPREADS ARB #{number}  " if number is not None else "  SPREADS ARBITRAGE  "
    if live is not None:
        title = title.strip() + f"  —  {'LIVE' if live else 'PREGAME'}  "
    home_spread = f"{opp.home_point:+.1f}" if opp.home_point != int(opp.home_point) else f"{int(opp.home_point):+d}"
    away_spread = f"{opp.away_point:+.1f}" if opp.away_point != int(opp.away_point) else f"{int(opp.away_point):+d}"
    lines = [
        "",
        "=" * 60,
        title.center(60),
        "=" * 60,
        f"  {opp.sport}  ·  {opp.home_team}  vs  {opp.away_team}",
        f"  Spread: {opp.home_team} {home_spread}  /  {opp.away_team} {away_spread}",
        f"  Game time: {opp.commence_time}",
        "",
        "  SPORTSBOOKS USED:",
        f"    1) {opp.home_book.upper()} ({opp.home_team} {home_spread})",
        f"    2) {opp.away_book.upper()} ({opp.away_team} {away_spread})",
        "",
        "  PLACE THESE TWO BETS:",
        "",
        f"  BET 1 — {opp.home_book.upper()}",
        f"    · Bet ${opp.home_stake:.2f} on: {opp.home_team} {home_spread}",
        f"    · Odds: {opp.home_odds:+.0f}",
        f"    · If {opp.home_team} covers, you get: ${opp.home_payout:.2f}",
        "",
        f"  BET 2 — {opp.away_book.upper()}",
        f"    · Bet ${opp.away_stake:.2f} on: {opp.away_team} {away_spread}",
        f"    · Odds: {opp.away_odds:+.0f}",
        f"    · If {opp.away_team} covers, you get: ${opp.away_payout:.2f}",
        "",
        "  SUMMARY:",
        f"    Total cost (both bets):  ${opp.total_investment:.2f}",
        f"    If {opp.home_team} covers:  you receive ${opp.home_payout:.2f}",
        f"    If {opp.away_team} covers:  you receive ${opp.away_payout:.2f}",
        f"    Guaranteed profit:         ${opp.guaranteed_profit:.2f}  ({opp.profit_pct:.2f}%)",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


# --- ARBITRAGE SCANNER ---

class ArbitrageScanner:
    """Scans The Odds API for sportsbook-vs-sportsbook arbitrage (no Kalshi API)."""

    def __init__(self):
        self.odds_client = OddsAPIClient(ODDS_API_KEY)
        self.opportunities: List[BookVsBookOpportunity] = []
        self.totals_opportunities: List[TotalsOpportunity] = []
        self.spreads_opportunities: List[SpreadsOpportunity] = []

    def scan(self, sport: Sport) -> Tuple[List[BookVsBookOpportunity], List[TotalsOpportunity], List[SpreadsOpportunity]]:
        """
        Fetch odds (h2h + totals + spreads) from multiple books; find moneyline, over/under, and spread arbs.
        Returns (moneyline_opportunities, totals_opportunities, spreads_opportunities).
        """
        logger.info(f"Starting multi-book scan for {sport.name} (h2h + totals + spreads)...")
        opportunities: List[BookVsBookOpportunity] = []
        totals_opps: List[TotalsOpportunity] = []
        spreads_opps: List[SpreadsOpportunity] = []

        games = self.odds_client.get_odds(
            sport,
            bookmakers=ODDS_BOOKMAKERS,
            regions=ODDS_REGIONS_MULTI,
            markets="h2h,totals,spreads",
        )
        if not games:
            logger.info("No games returned from Odds API")
            return opportunities, totals_opps, spreads_opps

        for game in games:
            opp = self._scan_game_books_vs_books(game, sport)
            if opp:
                opportunities.append(opp)
                self._log_book_vs_book_opportunity(opp)
            for topp in self._scan_game_totals(game, sport):
                totals_opps.append(topp)
                logger.info(format_totals_bet_slip(topp, live=is_live(topp.commence_time)))
            for sopp in self._scan_game_spreads(game, sport):
                spreads_opps.append(sopp)
                logger.info(format_spreads_bet_slip(sopp, live=is_live(sopp.commence_time)))

        logger.info(f"Found {len(opportunities)} moneyline + {len(totals_opps)} totals + {len(spreads_opps)} spreads arbitrage opportunities")
        self.opportunities = opportunities
        self.totals_opportunities = totals_opps
        self.spreads_opportunities = spreads_opps
        return opportunities, totals_opps, spreads_opps

    def scan_multiple_sports(
        self,
        sports: Optional[List[Sport]] = None,
    ) -> Tuple[List[BookVsBookOpportunity], List[TotalsOpportunity], List[SpreadsOpportunity]]:
        """
        Scan multiple sports in one run. One API request per sport; results are
        combined. Returns (moneyline_opps, totals_opps, spreads_opps).
        """
        to_scan = sports if sports is not None else SPORTS_TO_SCAN
        all_h2h: List[BookVsBookOpportunity] = []
        all_totals: List[TotalsOpportunity] = []
        all_spreads: List[SpreadsOpportunity] = []
        for sport in to_scan:
            h2h_opps, totals_opps, spreads_opps = self.scan(sport)
            all_h2h.extend(h2h_opps)
            all_totals.extend(totals_opps)
            all_spreads.extend(spreads_opps)
        self.opportunities = all_h2h
        self.totals_opportunities = all_totals
        self.spreads_opportunities = all_spreads
        return all_h2h, all_totals, all_spreads

    def _scan_game_totals(
        self,
        game: Dict,
        sport: Sport,
    ) -> List[TotalsOpportunity]:
        """
        For one game, collect each book's totals (Over/Under) market; group by point line;
        for each line find best Over and best Under across books; if arb, return opportunity.
        """
        game_id = game.get("id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence_time = game.get("commence_time", "")
        bookmakers = game.get("bookmakers", [])

        # Collect (point, book_key, over_odds, under_odds) per book; point rounded to 0.5 for grouping
        by_point: Dict[float, List[Tuple[str, float, float]]] = {}  # point -> [(book, over_odds, under_odds)]

        for book in bookmakers:
            bk_key = book.get("key", "")
            markets = book.get("markets", [])
            totals_market = next((m for m in markets if m.get("key") == "totals"), None)
            if not totals_market:
                continue
            outcomes = totals_market.get("outcomes", [])
            over_odds = None
            under_odds = None
            point = None
            for out in outcomes:
                name = (out.get("name") or "").strip().lower()
                price = out.get("price")
                pt = out.get("point")
                if price is None:
                    continue
                if name == "over":
                    over_odds = float(price)
                    if pt is not None:
                        point = float(pt)
                elif name == "under":
                    under_odds = float(price)
                    if pt is not None and point is None:
                        point = float(pt)
            if over_odds is None or under_odds is None or point is None:
                continue
            # Group by point rounded to 1 decimal (so 234.5 is one line)
            key = round(point, 1)
            if key not in by_point:
                by_point[key] = []
            by_point[key].append((bk_key, over_odds, under_odds))

        result: List[TotalsOpportunity] = []
        total_stake = min(BANKROLL * 0.1, MAX_STAKE_PER_ARB)

        for point_key, book_list in by_point.items():
            if len(book_list) < 2:
                continue
            best_over = max(book_list, key=lambda x: x[1])   # (book, over, under)
            best_under = max(book_list, key=lambda x: x[2])
            over_odds = best_over[1]
            under_odds = best_under[2]
            over_book = best_over[0]
            under_book = best_under[0]

            calc = ArbitrageCalculator.calculate_books_vs_books(over_odds, under_odds, total_stake)
            if not calc:
                continue

            result.append(TotalsOpportunity(
                game_id=game_id,
                sport=sport.name,
                home_team=home_team,
                away_team=away_team,
                commence_time=commence_time,
                point=point_key,
                over_book=over_book,
                over_odds=over_odds,
                over_stake=calc["side1_stake"],
                over_payout=calc["side1_payout"],
                under_book=under_book,
                under_odds=under_odds,
                under_stake=calc["side2_stake"],
                under_payout=calc["side2_payout"],
                total_investment=calc["total_investment"],
                guaranteed_profit=calc["guaranteed_profit"],
                profit_pct=calc["profit_pct"],
            ))

        return result

    def _scan_game_spreads(
        self,
        game: Dict,
        sport: Sport,
    ) -> List[SpreadsOpportunity]:
        """
        For one game, collect each book's spreads market; align outcomes to home/away by team name;
        group by spread line (home_point); for each line find best home-spread and away-spread odds
        across books; if arb, return opportunity.
        """
        game_id = game.get("id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence_time = game.get("commence_time", "")
        bookmakers = game.get("bookmakers", [])

        # Per-book: (home_point, away_point, home_odds, away_odds); group by (round(home_point, 1), round(away_point, 1))
        by_line: Dict[Tuple[float, float], List[Tuple[str, float, float]]] = {}  # (h_pt, a_pt) -> [(book, home_odds, away_odds)]

        for book in bookmakers:
            bk_key = book.get("key", "")
            markets = book.get("markets", [])
            spreads_market = next((m for m in markets if m.get("key") == "spreads"), None)
            if not spreads_market:
                continue
            outcomes = spreads_market.get("outcomes", [])
            if len(outcomes) != 2:
                continue
            # Align to home/away by matching outcome name to game teams
            candidates = [home_team, away_team]
            home_odds = None
            away_odds = None
            home_point = None
            away_point = None
            for out in outcomes:
                name = out.get("name", "")
                price = out.get("price")
                point = out.get("point")
                if price is None:
                    continue
                result = fuzzy_match_teams(name, candidates, threshold=50)
                if not result:
                    continue
                which_team, _ = result
                if which_team == home_team:
                    home_odds = float(price)
                    home_point = float(point) if point is not None else None
                else:
                    away_odds = float(price)
                    away_point = float(point) if point is not None else None
            if home_odds is None or away_odds is None:
                continue
            # If API doesn't return point, we can't group across books; skip
            if home_point is None and away_point is None:
                continue
            if home_point is None:
                home_point = -away_point if away_point is not None else 0.0
            if away_point is None:
                away_point = -home_point
            line_key = (round(home_point, 1), round(away_point, 1))
            if line_key not in by_line:
                by_line[line_key] = []
            by_line[line_key].append((bk_key, home_odds, away_odds))

        result_list: List[SpreadsOpportunity] = []
        total_stake = min(BANKROLL * 0.1, MAX_STAKE_PER_ARB)

        for (home_pt, away_pt), book_list in by_line.items():
            if len(book_list) < 2:
                continue
            best_home = max(book_list, key=lambda x: x[1])
            best_away = max(book_list, key=lambda x: x[2])
            home_odds = best_home[1]
            away_odds = best_away[2]
            home_book = best_home[0]
            away_book = best_away[0]

            calc = ArbitrageCalculator.calculate_books_vs_books(home_odds, away_odds, total_stake)
            if not calc:
                continue

            result_list.append(SpreadsOpportunity(
                game_id=game_id,
                sport=sport.name,
                home_team=home_team,
                away_team=away_team,
                commence_time=commence_time,
                home_point=home_pt,
                away_point=away_pt,
                home_book=home_book,
                home_odds=home_odds,
                home_stake=calc["side1_stake"],
                home_payout=calc["side1_payout"],
                away_book=away_book,
                away_odds=away_odds,
                away_stake=calc["side2_stake"],
                away_payout=calc["side2_payout"],
                total_investment=calc["total_investment"],
                guaranteed_profit=calc["guaranteed_profit"],
                profit_pct=calc["profit_pct"],
            ))

        return result_list

    def _scan_game_books_vs_books(
        self,
        game: Dict,
        sport: Sport,
    ) -> Optional[BookVsBookOpportunity]:
        """
        For one game, align each book's h2h outcomes to home/away, then take
        best odds per side across books. If impl_home + impl_away < 1, return arb.
        """
        game_id = game.get("id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence_time = game.get("commence_time", "")
        bookmakers = game.get("bookmakers", [])

        # Per-book (home_odds, away_odds); key = book key
        book_odds: List[Tuple[str, float, float]] = []  # (book_key, home_odds, away_odds)

        for book in bookmakers:
            bk_key = book.get("key", "")
            markets = book.get("markets", [])
            h2h = next((m for m in markets if m.get("key") == "h2h"), None)
            if not h2h:
                continue
            outcomes = h2h.get("outcomes", [])
            aligned = align_outcomes_to_home_away(home_team, away_team, outcomes)
            if aligned:
                home_odds, away_odds = aligned
                book_odds.append((bk_key, home_odds, away_odds))

        if len(book_odds) < 2:
            return None

        # Best odds per side across all books (no favoritism: purely max odds for each side)
        best_home = max(book_odds, key=lambda x: x[1])  # (book_key, home_odds, away_odds)
        best_away = max(book_odds, key=lambda x: x[2])

        odds_home = best_home[1]
        odds_away = best_away[2]
        book_home = best_home[0]
        book_away = best_away[0]

        total = min(BANKROLL * 0.1, MAX_STAKE_PER_ARB)
        calc = ArbitrageCalculator.calculate_books_vs_books(odds_home, odds_away, total)
        if not calc:
            return None

        return BookVsBookOpportunity(
            game_id=game_id,
            sport=sport.name,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            side1_team=home_team,
            side1_book=book_home,
            side1_odds=odds_home,
            side1_stake=calc["side1_stake"],
            side1_payout=calc["side1_payout"],
            side2_team=away_team,
            side2_book=book_away,
            side2_odds=odds_away,
            side2_stake=calc["side2_stake"],
            side2_payout=calc["side2_payout"],
            total_investment=calc["total_investment"],
            guaranteed_profit=calc["guaranteed_profit"],
            profit_pct=calc["profit_pct"],
        )

    def _log_book_vs_book_opportunity(self, opp: BookVsBookOpportunity) -> None:
        """Log a book-vs-book arbitrage opportunity (detailed)."""
        logger.info(format_bet_slip(opp, live=is_live(opp.commence_time)))

    @staticmethod
    def print_bet_slips(opportunities: List[BookVsBookOpportunity]) -> None:
        """Print moneyline opportunities in easy-to-read bet slip format to console."""
        for i, opp in enumerate(opportunities, 1):
            slip = format_bet_slip(opp, number=i, live=is_live(opp.commence_time))
            print(slip)
            print()

    @staticmethod
    def print_totals_slips(opportunities: List[TotalsOpportunity]) -> None:
        """Print over/under opportunities in bet slip format to console."""
        for i, opp in enumerate(opportunities, 1):
            slip = format_totals_bet_slip(opp, number=i, live=is_live(opp.commence_time))
            print(slip)
            print()

    @staticmethod
    def print_spreads_slips(opportunities: List[SpreadsOpportunity]) -> None:
        """Print spread opportunities in bet slip format to console."""
        for i, opp in enumerate(opportunities, 1):
            slip = format_spreads_bet_slip(opp, number=i, live=is_live(opp.commence_time))
            print(slip)
            print()

    def export_to_dict(self) -> List[Dict]:
        """Export moneyline opportunities to dictionary format."""
        return [asdict(opp) for opp in self.opportunities]


# --- MAIN EXECUTION ---

def main():
    """Run continuously: scan all sports every SCAN_INTERVAL_SEC until worthwhile arbs are found. Ctrl+C to stop."""
    print()
    print("=" * 60)
    print("  ARBITRAGE SCANNER — Continuous mode")
    print("  Scans multiple sports & books until arbs appear.")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    print()

    if not ODDS_API_KEY or ODDS_API_KEY.strip() == "":
        logger.error("Please set ODDS_API_KEY (e.g. export ODDS_API_KEY=your_key or use .env)")
        return

    scanner = ArbitrageScanner()
    cycle = 0

    try:
        while True:
            cycle += 1
            usage = scanner.odds_client.get_usage()
            if usage:
                used = usage.get("used")
                remaining = usage.get("remaining")
                if used is not None and remaining is not None:
                    print(f"  Odds API this period: {used} used, {remaining} remaining")
                    logger.info(f"[Cycle {cycle}] Odds API: {used} used, {remaining} remaining this period")
                elif remaining is not None:
                    print(f"  Odds API remaining: {remaining}")
                    logger.info(f"[Cycle {cycle}] Odds API requests remaining: {remaining}")

            h2h_opps, totals_opps, spreads_opps = scanner.scan_multiple_sports()
            sorted_h2h = sorted(h2h_opps, key=lambda x: x.profit_pct, reverse=True)
            sorted_totals = sorted(totals_opps, key=lambda x: x.profit_pct, reverse=True)
            sorted_spreads = sorted(spreads_opps, key=lambda x: x.profit_pct, reverse=True)
            has_any = sorted_h2h or sorted_totals or sorted_spreads

            if has_any:
                # Split into LIVE (act fast) and PREGAME (stick around longer)
                live_h2h = [o for o in sorted_h2h if is_live(o.commence_time)]
                pregame_h2h = [o for o in sorted_h2h if not is_live(o.commence_time)]
                live_totals = [o for o in sorted_totals if is_live(o.commence_time)]
                pregame_totals = [o for o in sorted_totals if not is_live(o.commence_time)]
                live_spreads = [o for o in sorted_spreads if is_live(o.commence_time)]
                pregame_spreads = [o for o in sorted_spreads if not is_live(o.commence_time)]

                logger.info(f"Found {len(sorted_h2h)} moneyline + {len(sorted_totals)} totals + {len(sorted_spreads)} spreads arb(s). Showing bet slips.")
                print()
                print("\n  *** WORTHWHILE ARBS FOUND — PLACE THESE BETS ***\n")

                # LIVE first (act fast)
                has_live = live_h2h or live_totals or live_spreads
                if has_live:
                    print("  ==========  LIVE (act fast)  ==========\n")
                    if live_h2h:
                        print("  --- MONEYLINE (H2H) ---\n")
                        ArbitrageScanner.print_bet_slips(live_h2h)
                    if live_totals:
                        print("  --- OVER/UNDER (TOTALS) ---\n")
                        ArbitrageScanner.print_totals_slips(live_totals)
                    if live_spreads:
                        print("  --- SPREADS ---\n")
                        ArbitrageScanner.print_spreads_slips(live_spreads)

                # PREGAME (more time)
                has_pregame = pregame_h2h or pregame_totals or pregame_spreads
                if has_pregame:
                    print("  ==========  PREGAME  ==========\n")
                    if pregame_h2h:
                        print("  --- MONEYLINE (H2H) ---\n")
                        ArbitrageScanner.print_bet_slips(pregame_h2h)
                    if pregame_totals:
                        print("  --- OVER/UNDER (TOTALS) ---\n")
                        ArbitrageScanner.print_totals_slips(pregame_totals)
                    if pregame_spreads:
                        print("  --- SPREADS ---\n")
                        ArbitrageScanner.print_spreads_slips(pregame_spreads)

                print("  Next scan in", SCAN_INTERVAL_SEC, "seconds...")
            else:
                logger.info(f"[Cycle {cycle}] No arbs this round. Next scan in {SCAN_INTERVAL_SEC}s.")

            time.sleep(SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        print()
        logger.info("Stopped by user (Ctrl+C).")
        print("Scan stopped.")


if __name__ == "__main__":
    main()
