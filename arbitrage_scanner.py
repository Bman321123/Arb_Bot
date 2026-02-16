"""
Sportsbook-vs-Sportsbook Arbitrage Scanner
Uses The Odds API to fetch odds from all NY-available sportsbooks (us + us2).
Scans every book against every other; best odds per side across books. Markets: h2h, totals, spreads.
Version: 4.0
"""

import os
import re
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from rapidfuzz import fuzz, process

# --- CONFIGURATION ---
ODDS_API_KEY = (os.environ.get("ODDS_API_KEY") or "862942ef3e86e6ca99daf3e24c95bdc4").strip()
KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY", "").strip()
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
ODDS_API_URL = "https://api.the-odds-api.com/v4"

# NY-available books: Odds API "us" + "us2" state-licensed bookmakers (DraftKings, FanDuel, Caesars, BetMGM, BetRivers, Fanatics, ESPN Bet, Hard Rock, Bally).
# Caesars = williamhill_us. We scan all books against each other; best odds per side across books.
ODDS_REGIONS_FOR_HARDROCK = "us"
ODDS_BOOKMAKERS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "betrivers",
    "fanatics",
    "williamhill_us",   # Caesars
    "espnbet",          # ESPN Bet / theScore Bet (us2)
    "hardrockbet",     # us2
    "ballybet",        # us2
]
ODDS_REGIONS_MULTI = "us,us2"

BANKROLL = 1000.00
MIN_PROFIT_PCT = 1.0
MAX_STAKE_PER_ARB = 1000.00
# Max allowed on any single leg; we size so the larger leg equals this to maximize profit.
MAX_STAKE_PER_SIDE = 250.00
KELLY_FRACTION = 0.1

KALSHI_TAKER_FEE = 0.01
HARDROCK_MARGIN = 0.0

ODDS_API_DELAY = 1.0
KALSHI_API_DELAY = 0.5

SCAN_INTERVAL_SEC = 15

# Stop the program when this many credits have been used in the current period.
ODDS_API_CREDITS_LIMIT = 450

FUZZY_MATCH_THRESHOLD = 60
MIN_MATCH_CONFIDENCE = 50
JACCARD_MIN = 0.2

# Kalshi Yes/No: "Team A NO" = "Team B YES". Odds API may return team names with NO side.
KALSHI_BOOK_KEY = "kalshi"

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
    NFL = "americanfootball_nfl"
    NBA = "basketball_nba"
    NCAAB = "basketball_ncaab"
    NHL = "icehockey_nhl"
    MLB = "baseball_mlb"


SPORTS_TO_SCAN: List[Sport] = [
    Sport.NFL,
    Sport.NBA,
    Sport.NCAAB,
    Sport.NHL,
    Sport.MLB,
]


@dataclass
class BookVsBookOpportunity:
    game_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str
    side1_team: str
    side1_book: str
    side1_odds: float
    side1_stake: float
    side1_payout: float
    side2_team: str
    side2_book: str
    side2_odds: float
    side2_stake: float
    side2_payout: float
    total_investment: float
    guaranteed_profit: float
    profit_pct: float
    timestamp: str = ""
    side1_link: Optional[str] = None
    side2_link: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TotalsOpportunity:
    game_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str
    point: float
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
    timestamp: str = ""
    over_link: Optional[str] = None
    under_link: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class SpreadsOpportunity:
    game_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str
    home_point: float
    away_point: float
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
    timestamp: str = ""
    home_link: Optional[str] = None
    away_link: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# --- UTILITY FUNCTIONS ---

def american_to_decimal(american_odds: float) -> float:
    if american_odds > 0:
        return (american_odds / 100) + 1
    return (100 / abs(american_odds)) + 1


def american_to_implied_prob(american_odds: float) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds >= 2.0:
        return (decimal_odds - 1) * 100
    return -100 / (decimal_odds - 1)


def implied_prob_to_american(implied_prob: float) -> float:
    if implied_prob <= 0 or implied_prob >= 1:
        return 0.0
    return decimal_to_american(1.0 / implied_prob)


def normalize_team_name(name: str) -> str:
    replacements = {"St.": "State", "St ": "State ", "&": "and", "'": ""}
    normalized = name.strip()
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized.lower()


def tokenize_to_set(text: str) -> set:
    normalized = normalize_team_name(text)
    words = re.findall(r"[a-z0-9]+", normalized)
    return set(w for w in words if len(w) > 1)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def align_outcomes_to_home_away(
    home_team: str,
    away_team: str,
    outcomes: List[Dict],
) -> Optional[Tuple[float, float]]:
    if not outcomes or len(outcomes) != 2:
        return None
    candidates = [home_team, away_team]
    matched: Dict[str, float] = {}
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
    if not candidates:
        return None
    normalized_team = normalize_team_name(team_name)
    normalized_candidates = [normalize_team_name(c) for c in candidates]
    scores = [(c, fuzz.ratio(normalized_team, normalize_team_name(c))) for c in candidates]
    best = max(scores, key=lambda x: x[1])
    if best[1] < threshold:
        return None
    original_match = candidates[normalized_candidates.index(normalize_team_name(best[0]))]
    return (original_match, best[1])


def is_live(commence_time: str) -> bool:
    """True if game has already started (commence_time <= now UTC)."""
    try:
        start = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return start <= datetime.now(timezone.utc)
    except (ValueError, TypeError):
        return False


def is_pregame(commence_time: str) -> bool:
    """True if game start is in the future (commence_time > now UTC). Used for labeling and grouping slips."""
    try:
        start = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return start > datetime.now(timezone.utc)
    except (ValueError, TypeError):
        return False


def _outcome_link(outcome: Dict, market: Dict, book: Dict) -> Optional[str]:
    link = (outcome or {}).get("link") or (market or {}).get("link") or (book or {}).get("link")
    return (link or "").strip() or None


def _link_for_new_jersey(link: Optional[str], book_key: str) -> Optional[str]:
    link = (link or "").strip() or None
    if not link:
        return None
    key = (book_key or "").lower().strip()
    try:
        parsed = urlparse(link)
        if key == "draftkings":
            q = parse_qs(parsed.query)
            q["intendedSiteExp"] = ["US-NJ-SB"]
            new_query = urlencode(q, doseq=True)
            out = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
            return out.replace("{state}", "NJ")
        if key == "fanduel":
            q = parse_qs(parsed.query)
            q["state"] = ["NJ"]
            new_query = urlencode(q, doseq=True)
            out = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
            return out.replace("{state}", "NJ")
        if key == "betmgm":
            # NJ format per reference: https://www.nj.betmgm.com/en/sports?...
            netloc = "www.nj.betmgm.com"
            path = (parsed.path or "").strip() or "/en/sports"
            if not path.startswith("/"):
                path = "/" + path
            query = (parsed.query or "").replace("{state}", "NJ")
            out = urlunparse(("https", netloc, path, parsed.params, query, parsed.fragment))
            return out.replace("{state}", "NJ")
        if key == "hardrockbet":
            q = parse_qs(parsed.query)
            q["state"] = ["NJ"]
            new_query = urlencode(q, doseq=True)
            out = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
            return out.replace("{state}", "NJ")
    except Exception:
        pass
    return link.replace("{state}", "NJ") if link else link


def _kalshi_contract_price_cents(american_odds: float) -> int:
    p = american_to_implied_prob(american_odds)
    return round(p * 100)


def _kalshi_stake_to_contracts(stake_dollars: float, american_odds: float) -> float:
    price = american_to_implied_prob(american_odds)
    if price <= 0:
        return 0.0
    return stake_dollars / price


# --- FORMAT BET SLIPS ---

def format_bet_slip(opp: BookVsBookOpportunity, number: Optional[int] = None, live: Optional[bool] = None) -> str:
    title = f"  ARB #{number}  " if number is not None else "  ARBITRAGE BET SLIP  "
    if live is not None:
        title = title.strip() + f"  —  {'LIVE' if live else 'PREGAME'}  "
    s1_kalshi = getattr(opp, "side1_book", "") and str(opp.side1_book).lower() == KALSHI_BOOK_KEY
    s2_kalshi = getattr(opp, "side2_book", "") and str(opp.side2_book).lower() == KALSHI_BOOK_KEY

    def leg1_lines() -> List[str]:
        if s1_kalshi:
            contracts = _kalshi_stake_to_contracts(opp.side1_stake, opp.side1_odds)
            price_c = _kalshi_contract_price_cents(opp.side1_odds)
            return [
                f"  BET 1 — {opp.side1_book.upper()}",
                f"    · Buy {contracts:,.1f} contracts on: {opp.side1_team}",
                f"    · Contract price: {price_c}¢",
                f"    · If this wins, you get: ${opp.side1_payout:.2f}",
            ]
        return [
            f"  BET 1 — {opp.side1_book.upper()}",
            f"    · Bet ${opp.side1_stake:.2f} on: {opp.side1_team}",
            f"    · Odds: {opp.side1_odds:+.1f}",
            f"    · If this wins, you get: ${opp.side1_payout:.2f}",
        ]

    def leg2_lines() -> List[str]:
        if s2_kalshi:
            contracts = _kalshi_stake_to_contracts(opp.side2_stake, opp.side2_odds)
            price_c = _kalshi_contract_price_cents(opp.side2_odds)
            return [
                f"  BET 2 — {opp.side2_book.upper()}",
                f"    · Buy {contracts:,.1f} contracts on: {opp.side2_team}",
                f"    · Contract price: {price_c}¢",
                f"    · If this wins, you get: ${opp.side2_payout:.2f}",
            ]
        return [
            f"  BET 2 — {opp.side2_book.upper()}",
            f"    · Bet ${opp.side2_stake:.2f} on: {opp.side2_team}",
            f"    · Odds: {opp.side2_odds:+.1f}",
            f"    · If this wins, you get: ${opp.side2_payout:.2f}",
        ]

    lines = [
        "", "=" * 60, title.center(60), "=" * 60,
        f"  {opp.sport}  ·  {opp.home_team}  vs  {opp.away_team}",
        f"  Game time: {opp.commence_time}",
        "", "  SPORTSBOOKS USED:",
        f"    1) {opp.side1_book.upper()}",
        f"    2) {opp.side2_book.upper()}",
        "", "  PLACE THESE TWO BETS:",
        "",
        *leg1_lines(),
        *([f"    · Link: {opp.side1_link}"] if getattr(opp, "side1_link", None) else []),
        *([f"    · On Kalshi: buy {opp.side1_team} YES or {opp.side2_team} NO"] if s1_kalshi else []),
        "",
        *leg2_lines(),
        *([f"    · Link: {opp.side2_link}"] if getattr(opp, "side2_link", None) else []),
        *([f"    · On Kalshi: buy {opp.side2_team} YES or {opp.side1_team} NO"] if s2_kalshi else []),
        "",
        "  SUMMARY:",
        f"    Total cost (both bets):  ${opp.total_investment:.2f}",
        f"    If {opp.side1_team} wins:  you receive ${opp.side1_payout:.2f}",
        f"    If {opp.side2_team} wins:  you receive ${opp.side2_payout:.2f}",
        f"    Guaranteed profit:       ${opp.guaranteed_profit:.2f}  ({opp.profit_pct:.2f}%)",
        "",
        "  NOTE: Odds from API (can be 1–2 min old). Always verify on the book before placing.",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_totals_bet_slip(opp: TotalsOpportunity, number: Optional[int] = None, live: Optional[bool] = None) -> str:
    title = f"  TOTALS ARB #{number}  " if number is not None else "  OVER/UNDER ARBITRAGE  "
    if live is not None:
        title = title.strip() + f"  —  {'LIVE' if live else 'PREGAME'}  "
    over_kalshi = getattr(opp, "over_book", "") and str(opp.over_book).lower() == KALSHI_BOOK_KEY
    under_kalshi = getattr(opp, "under_book", "") and str(opp.under_book).lower() == KALSHI_BOOK_KEY
    bet1 = [
        f"  BET 1 — {opp.over_book.upper()} (OVER {opp.point})",
        f"    · Buy {_kalshi_stake_to_contracts(opp.over_stake, opp.over_odds):,.1f} contracts on: Over {opp.point}" if over_kalshi else f"    · Bet ${opp.over_stake:.2f} on: Over {opp.point}",
        f"    · Contract price: {_kalshi_contract_price_cents(opp.over_odds)}¢" if over_kalshi else f"    · Odds: {opp.over_odds:+.1f}",
        f"    · If Over hits, you get: ${opp.over_payout:.2f}",
    ]
    bet2 = [
        f"  BET 2 — {opp.under_book.upper()} (UNDER {opp.point})",
        f"    · Buy {_kalshi_stake_to_contracts(opp.under_stake, opp.under_odds):,.1f} contracts on: Under {opp.point}" if under_kalshi else f"    · Bet ${opp.under_stake:.2f} on: Under {opp.point}",
        f"    · Contract price: {_kalshi_contract_price_cents(opp.under_odds)}¢" if under_kalshi else f"    · Odds: {opp.under_odds:+.1f}",
        f"    · If Under hits, you get: ${opp.under_payout:.2f}",
    ]
    lines = [
        "", "=" * 60, title.center(60), "=" * 60,
        f"  {opp.sport}  ·  {opp.home_team}  vs  {opp.away_team}",
        f"  Total points line: {opp.point}", f"  Game time: {opp.commence_time}",
        "", "  SPORTSBOOKS USED:",
        f"    1) {opp.over_book.upper()} (Over)",
        f"    2) {opp.under_book.upper()} (Under)",
        "", "  PLACE THESE TWO BETS:", "",
        *bet1, *([f"    · Link: {opp.over_link}"] if getattr(opp, "over_link", None) else []),
        "",
        *bet2, *([f"    · Link: {opp.under_link}"] if getattr(opp, "under_link", None) else []),
        "",
        "  SUMMARY:",
        f"    Total cost (both bets):  ${opp.total_investment:.2f}",
        f"    If Over {opp.point}:  you receive ${opp.over_payout:.2f}",
        f"    If Under {opp.point}:  you receive ${opp.under_payout:.2f}",
        f"    Guaranteed profit:       ${opp.guaranteed_profit:.2f}  ({opp.profit_pct:.2f}%)",
        "",
        "  NOTE: Odds from API (can be 1–2 min old). Always verify on the book before placing.",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_spreads_bet_slip(opp: SpreadsOpportunity, number: Optional[int] = None, live: Optional[bool] = None) -> str:
    title = f"  SPREADS ARB #{number}  " if number is not None else "  SPREADS ARBITRAGE  "
    if live is not None:
        title = title.strip() + f"  —  {'LIVE' if live else 'PREGAME'}  "
    home_spread = f"{opp.home_point:+.1f}" if opp.home_point != int(opp.home_point) else f"{int(opp.home_point):+d}"
    away_spread = f"{opp.away_point:+.1f}" if opp.away_point != int(opp.away_point) else f"{int(opp.away_point):+d}"
    home_kalshi = getattr(opp, "home_book", "") and str(opp.home_book).lower() == KALSHI_BOOK_KEY
    away_kalshi = getattr(opp, "away_book", "") and str(opp.away_book).lower() == KALSHI_BOOK_KEY
    bet1 = [
        f"  BET 1 — {opp.home_book.upper()}",
        f"    · Buy {_kalshi_stake_to_contracts(opp.home_stake, opp.home_odds):,.1f} contracts on: {opp.home_team} {home_spread}" if home_kalshi else f"    · Bet ${opp.home_stake:.2f} on: {opp.home_team} {home_spread}",
        f"    · Contract price: {_kalshi_contract_price_cents(opp.home_odds)}¢" if home_kalshi else f"    · Odds: {opp.home_odds:+.1f}",
        f"    · If {opp.home_team} covers, you get: ${opp.home_payout:.2f}",
    ]
    bet2 = [
        f"  BET 2 — {opp.away_book.upper()}",
        f"    · Buy {_kalshi_stake_to_contracts(opp.away_stake, opp.away_odds):,.1f} contracts on: {opp.away_team} {away_spread}" if away_kalshi else f"    · Bet ${opp.away_stake:.2f} on: {opp.away_team} {away_spread}",
        f"    · Contract price: {_kalshi_contract_price_cents(opp.away_odds)}¢" if away_kalshi else f"    · Odds: {opp.away_odds:+.1f}",
        f"    · If {opp.away_team} covers, you get: ${opp.away_payout:.2f}",
    ]
    lines = [
        "", "=" * 60, title.center(60), "=" * 60,
        f"  {opp.sport}  ·  {opp.home_team}  vs  {opp.away_team}",
        f"  Spread: {opp.home_team} {home_spread}  /  {opp.away_team} {away_spread}",
        f"  Game time: {opp.commence_time}",
        "", "  SPORTSBOOKS USED:",
        f"    1) {opp.home_book.upper()} ({opp.home_team} {home_spread})",
        f"    2) {opp.away_book.upper()} ({opp.away_team} {away_spread})",
        "", "  PLACE THESE TWO BETS:", "",
        *bet1, *([f"    · Link: {opp.home_link}"] if getattr(opp, "home_link", None) else []),
        "", *bet2, *([f"    · Link: {opp.away_link}"] if getattr(opp, "away_link", None) else []),
        "",
        "  SUMMARY:",
        f"    Total cost (both bets):  ${opp.total_investment:.2f}",
        f"    If {opp.home_team} covers:  you receive ${opp.home_payout:.2f}",
        f"    If {opp.away_team} covers:  you receive ${opp.away_payout:.2f}",
        f"    Guaranteed profit:         ${opp.guaranteed_profit:.2f}  ({opp.profit_pct:.2f}%)",
        "",
        "  NOTE: Odds from API (can be 1–2 min old). Always verify on the book before placing.",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


# --- ODDS API CLIENT ---

class OddsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.base_url = ODDS_API_URL

    def get_odds(
        self,
        sport: Sport,
        bookmakers: Optional[List[str]] = None,
        regions: Optional[str] = None,
        markets: str = "h2h",
    ) -> List[Dict]:
        url = f"{self.base_url}/sports/{sport.value}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions or ODDS_REGIONS_FOR_HARDROCK,
            "markets": markets,
            "oddsFormat": "american",
            "includeLinks": "true",
            "includeSids": "true",
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

    def get_usage(self) -> Optional[Dict[str, int]]:
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
                out["remaining"] = int(remaining)
            if used is not None:
                out["used"] = int(used)
            return out if out else None
        except Exception:
            return None


# --- ARBITRAGE CALCULATOR ---

class ArbitrageCalculator:
    @staticmethod
    def calculate_books_vs_books(
        odds_side1: float,
        odds_side2: float,
        total_investment: Optional[float] = None,
        max_stake_per_side: Optional[float] = None,
    ) -> Optional[Dict]:
        impl_1 = american_to_implied_prob(odds_side1)
        impl_2 = american_to_implied_prob(odds_side2)
        total_impl = impl_1 + impl_2
        if total_impl >= 1.0:
            return None
        # Size so the larger leg equals max_stake_per_side to maximize dollar profit (profit % is fixed by odds)
        if max_stake_per_side is not None and max_stake_per_side > 0:
            total_investment = max_stake_per_side * (total_impl / max(impl_1, impl_2))
        if total_investment is None or total_investment <= 0:
            return None
        stake1 = total_investment * (impl_1 / total_impl)
        stake2 = total_investment * (impl_2 / total_impl)
        # Round so stakes sum exactly to total_investment (avoids $99.99 vs $100.00 confusion)
        stake1 = round(stake1, 2)
        stake2 = round(total_investment - stake1, 2)
        payout1 = stake1 * american_to_decimal(odds_side1)
        payout2 = stake2 * american_to_decimal(odds_side2)
        guaranteed = min(payout1, payout2) - total_investment
        profit_pct = (guaranteed / total_investment) * 100
        if profit_pct < MIN_PROFIT_PCT:
            return None
        return {
            "side1_stake": stake1,
            "side2_stake": stake2,
            "total_investment": round(total_investment, 2),
            "side1_payout": round(payout1, 2),
            "side2_payout": round(payout2, 2),
            "guaranteed_profit": round(guaranteed, 2),
            "profit_pct": round(profit_pct, 2),
        }


# --- ARBITRAGE SCANNER ---

class ArbitrageScanner:
    def __init__(self):
        self.odds_client = OddsAPIClient(ODDS_API_KEY)
        self.opportunities: List[BookVsBookOpportunity] = []
        self.totals_opportunities: List[TotalsOpportunity] = []
        self.spreads_opportunities: List[SpreadsOpportunity] = []

    def scan(self, sport: Sport) -> Tuple[List[BookVsBookOpportunity], List[TotalsOpportunity], List[SpreadsOpportunity]]:
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

        # Scan all games (live and pregame); slips are labeled LIVE vs PREGAME and grouped accordingly
        for game in games:
            opp = self._scan_game_books_vs_books(game, sport)
            if opp:
                opportunities.append(opp)
                logger.info(format_bet_slip(opp, live=is_live(opp.commence_time)))
            for topp in self._scan_game_totals(game, sport):
                totals_opps.append(topp)
                logger.info(format_totals_bet_slip(topp, live=is_live(topp.commence_time)))
            for sopp in self._scan_game_spreads(game, sport):
                spreads_opps.append(sopp)
                logger.info(format_spreads_bet_slip(sopp, live=is_live(sopp.commence_time)))

        logger.info(f"Found {len(opportunities)} moneyline + {len(totals_opps)} totals + {len(spreads_opps)} spreads")
        self.opportunities = opportunities
        self.totals_opportunities = totals_opps
        self.spreads_opportunities = spreads_opps
        return opportunities, totals_opps, spreads_opps

    def scan_multiple_sports(
        self,
        sports: Optional[List[Sport]] = None,
    ) -> Tuple[List[BookVsBookOpportunity], List[TotalsOpportunity], List[SpreadsOpportunity]]:
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
        game_id = game.get("id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence_time = game.get("commence_time", "")
        bookmakers = game.get("bookmakers", [])

        by_point: Dict[float, List[Tuple[str, float, float, Optional[str], Optional[str]]]] = {}

        for book in bookmakers:
            bk_key = book.get("key", "")
            markets = book.get("markets", [])
            totals_market = next((m for m in markets if m.get("key") == "totals"), None)
            if not totals_market:
                continue
            outcomes = totals_market.get("outcomes", [])
            over_odds = None
            under_odds = None
            over_link = None
            under_link = None
            point = None
            for out in outcomes:
                name = (out.get("name") or "").strip().lower()
                price = out.get("price")
                pt = out.get("point")
                if price is None:
                    continue
                link = _outcome_link(out, totals_market, book)
                if name == "over":
                    over_odds = float(price)
                    over_link = link
                    if pt is not None:
                        point = float(pt)
                elif name == "under":
                    under_odds = float(price)
                    under_link = link
                    if pt is not None and point is None:
                        point = float(pt)
            if over_odds is None or under_odds is None or point is None:
                continue
            if (bk_key or "").lower() == KALSHI_BOOK_KEY:
                over_odds, under_odds = under_odds, over_odds
                over_link, under_link = under_link, over_link
            key = round(point, 1)
            if key not in by_point:
                by_point[key] = []
            by_point[key].append((bk_key, over_odds, under_odds, over_link, under_link))

        result: List[TotalsOpportunity] = []

        for point_key, book_list in by_point.items():
            if len(book_list) < 2:
                continue
            best_over = max(book_list, key=lambda x: x[1])
            best_under = max(book_list, key=lambda x: x[2])
            over_odds = best_over[1]
            under_odds = best_under[2]
            over_book = best_over[0]
            under_book = best_under[0]
            over_link = best_over[3] if len(best_over) > 3 else None
            under_link = best_under[4] if len(best_under) > 4 else None

            calc = ArbitrageCalculator.calculate_books_vs_books(
                over_odds, under_odds, max_stake_per_side=MAX_STAKE_PER_SIDE
            )
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
                over_link=_link_for_new_jersey(over_link, over_book),
                under_link=_link_for_new_jersey(under_link, under_book),
            ))

        return result

    def _scan_game_spreads(
        self,
        game: Dict,
        sport: Sport,
    ) -> List[SpreadsOpportunity]:
        game_id = game.get("id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence_time = game.get("commence_time", "")
        bookmakers = game.get("bookmakers", [])

        by_line: Dict[Tuple[float, float], List[Tuple[str, float, float, Optional[str], Optional[str]]]] = {}

        for book in bookmakers:
            bk_key = book.get("key", "")
            markets = book.get("markets", [])
            spreads_market = next((m for m in markets if m.get("key") == "spreads"), None)
            if not spreads_market:
                continue
            outcomes = spreads_market.get("outcomes", [])
            if len(outcomes) != 2:
                continue
            candidates = [home_team, away_team]
            home_odds = None
            away_odds = None
            home_point = None
            away_point = None
            home_link = None
            away_link = None
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
                link = _outcome_link(out, spreads_market, book)
                if which_team == home_team:
                    home_odds = float(price)
                    home_point = float(point) if point is not None else None
                    home_link = link
                else:
                    away_odds = float(price)
                    away_point = float(point) if point is not None else None
                    away_link = link
            if home_odds is None or away_odds is None:
                continue
            if home_point is None and away_point is None:
                continue
            if home_point is None:
                home_point = -away_point if away_point is not None else 0.0
            if away_point is None:
                away_point = -home_point
            if (bk_key or "").lower() == KALSHI_BOOK_KEY:
                home_odds, away_odds = away_odds, home_odds
                home_link, away_link = away_link, home_link
            line_key = (round(home_point, 1), round(away_point, 1))
            if line_key not in by_line:
                by_line[line_key] = []
            by_line[line_key].append((bk_key, home_odds, away_odds, home_link, away_link))

        result_list: List[SpreadsOpportunity] = []

        for (home_pt, away_pt), book_list in by_line.items():
            if len(book_list) < 2:
                continue
            best_home = max(book_list, key=lambda x: x[1])
            best_away = max(book_list, key=lambda x: x[2])
            home_odds = best_home[1]
            away_odds = best_away[2]
            home_book = best_home[0]
            away_book = best_away[0]
            home_link = best_home[3] if len(best_home) > 3 else None
            away_link = best_away[4] if len(best_away) > 4 else None

            calc = ArbitrageCalculator.calculate_books_vs_books(
                home_odds, away_odds, max_stake_per_side=MAX_STAKE_PER_SIDE
            )
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
                home_link=_link_for_new_jersey(home_link, home_book),
                away_link=_link_for_new_jersey(away_link, away_book),
            ))

        return result_list

    def _scan_game_books_vs_books(
        self,
        game: Dict,
        sport: Sport,
    ) -> Optional[BookVsBookOpportunity]:
        game_id = game.get("id", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence_time = game.get("commence_time", "")
        bookmakers = game.get("bookmakers", [])

        book_odds: List[Tuple[str, float, float, Optional[str], Optional[str]]] = []

        for book in bookmakers:
            bk_key = book.get("key", "")
            markets = book.get("markets", [])
            h2h = next((m for m in markets if m.get("key") == "h2h"), None)
            if not h2h:
                continue
            outcomes = h2h.get("outcomes", [])
            candidates = [home_team, away_team]
            matched: Dict[str, Tuple[float, Optional[str]]] = {}
            for out in outcomes:
                name = out.get("name", "")
                price = out.get("price")
                if price is None:
                    continue
                result = fuzzy_match_teams(name, candidates, threshold=50)
                if not result:
                    continue
                which_team, _ = result
                link = _outcome_link(out, h2h, book)
                if which_team == home_team:
                    matched["home"] = (float(price), link)
                else:
                    matched["away"] = (float(price), link)
            if "home" in matched and "away" in matched:
                home_odds_val, home_link_val = matched["home"][0], matched["home"][1]
                away_odds_val, away_link_val = matched["away"][0], matched["away"][1]
                if (bk_key or "").lower() == KALSHI_BOOK_KEY:
                    impl_h = american_to_implied_prob(home_odds_val)
                    impl_a = american_to_implied_prob(away_odds_val)
                    if impl_h + impl_a > 1.001:
                        home_odds_val = implied_prob_to_american(1.0 - impl_h)
                book_odds.append((
                    bk_key,
                    home_odds_val, away_odds_val,
                    home_link_val, away_link_val,
                ))

        if len(book_odds) < 2:
            return None

        best_home = max(book_odds, key=lambda x: x[1])
        best_away = max(book_odds, key=lambda x: x[2])

        odds_home = best_home[1]
        odds_away = best_away[2]
        book_home = best_home[0]
        book_away = best_away[0]
        home_link = best_home[3] if len(best_home) > 3 else None
        away_link = best_away[4] if len(best_away) > 4 else None

        calc = ArbitrageCalculator.calculate_books_vs_books(
            odds_home, odds_away, max_stake_per_side=MAX_STAKE_PER_SIDE
        )
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
            side1_link=_link_for_new_jersey(home_link, book_home),
            side2_link=_link_for_new_jersey(away_link, book_away),
        )

    @staticmethod
    def print_bet_slips(opportunities: List[BookVsBookOpportunity]) -> None:
        for i, opp in enumerate(opportunities, 1):
            print(format_bet_slip(opp, number=i, live=is_live(opp.commence_time)))
            print()

    @staticmethod
    def print_totals_slips(opportunities: List[TotalsOpportunity]) -> None:
        for i, opp in enumerate(opportunities, 1):
            print(format_totals_bet_slip(opp, number=i, live=is_live(opp.commence_time)))
            print()

    @staticmethod
    def print_spreads_slips(opportunities: List[SpreadsOpportunity]) -> None:
        for i, opp in enumerate(opportunities, 1):
            print(format_spreads_bet_slip(opp, number=i, live=is_live(opp.commence_time)))
            print()


# --- MAIN ---

def main() -> None:
    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY not set. Set it in .env or export ODDS_API_KEY=your_key")
        return

    scanner = ArbitrageScanner()

    # Show Odds API credits at startup
    usage = scanner.odds_client.get_usage()
    if usage:
        remaining = usage.get("remaining")
        used = usage.get("used")
        if used is not None and used >= ODDS_API_CREDITS_LIMIT:
            logger.warning(f"Odds API credits already at or above limit ({used} >= {ODDS_API_CREDITS_LIMIT}). Exiting.")
            print(f"  Odds API: {used} credits used (limit {ODDS_API_CREDITS_LIMIT}). Exiting.")
            sys.exit(0)
        if remaining is not None:
            msg = f"  Odds API credits: {remaining} remaining"
            if used is not None:
                msg += f"  ({used} used this period)"
            print(msg)
        logger.info(f"Odds API at startup: {usage}")

    try:
        logger.info(f"Scanning {[s.name for s in SPORTS_TO_SCAN]} (single run)...")
        usage = scanner.odds_client.get_usage()
        if usage:
            used = usage.get("used")
            remaining = usage.get("remaining")
            if used is not None and used >= ODDS_API_CREDITS_LIMIT:
                logger.warning(f"Odds API credits at limit ({used} >= {ODDS_API_CREDITS_LIMIT}). Exiting.")
                print(f"  Odds API: {used} credits used (limit {ODDS_API_CREDITS_LIMIT}). Exiting.")
                sys.exit(0)
            if remaining is not None:
                msg = f"  Odds API credits: {remaining} remaining"
                if used is not None:
                    msg += f"  ({used} used this period)"
                print(msg)
                logger.info(f"Odds API: used={used}, remaining={remaining}")
            elif used is not None:
                logger.info(f"Odds API: used={used}")

        h2h_opps, totals_opps, spreads_opps = scanner.scan_multiple_sports()

        # Check again after scan (we just used more credits); exit if at limit
        usage = scanner.odds_client.get_usage()
        if usage and usage.get("used") is not None and usage.get("used") >= ODDS_API_CREDITS_LIMIT:
            used = usage.get("used")
            logger.warning(f"Odds API credits at limit after scan ({used} >= {ODDS_API_CREDITS_LIMIT}). Exiting.")
            print(f"  Odds API: {used} credits used (limit {ODDS_API_CREDITS_LIMIT}). Exiting.")
            sys.exit(0)
        sorted_h2h = sorted(h2h_opps, key=lambda x: x.profit_pct, reverse=True)
        sorted_totals = sorted(totals_opps, key=lambda x: x.profit_pct, reverse=True)
        sorted_spreads = sorted(spreads_opps, key=lambda x: x.profit_pct, reverse=True)
        has_any = sorted_h2h or sorted_totals or sorted_spreads

        if has_any:
            live_h2h = [o for o in sorted_h2h if is_live(o.commence_time)]
            pregame_h2h = [o for o in sorted_h2h if not is_live(o.commence_time)]
            live_totals = [o for o in sorted_totals if is_live(o.commence_time)]
            pregame_totals = [o for o in sorted_totals if not is_live(o.commence_time)]
            live_spreads = [o for o in sorted_spreads if is_live(o.commence_time)]
            pregame_spreads = [o for o in sorted_spreads if not is_live(o.commence_time)]

            logger.info(f"Found {len(sorted_h2h)} moneyline + {len(sorted_totals)} totals + {len(sorted_spreads)} spreads")
            print("\n  *** WORTHWHILE ARBS FOUND — PLACE THESE BETS ***\n")

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

            print("  Scan complete.")
        else:
            logger.info("No arbs this scan.")

    except KeyboardInterrupt:
        logger.info("Stopped by user.")


if __name__ == "__main__":
    main()
