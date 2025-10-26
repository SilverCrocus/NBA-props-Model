"""
Enhanced Bet Logger with Calibration Diagnostics

Extends CLVTracker with detailed calibration and Kelly sizing diagnostics.

New Columns:
- prob_raw: Uncalibrated probability
- prob_cal: Calibrated probability
- q_novig: No-vig fair probability (from vig removal)
- ev_pre: EV before calibration
- ev_post: EV after calibration
- kelly_base: Full Kelly (before fraction)
- kelly_mult: ECE penalty multiplier [0, 1]
- kelly_frac: Fraction applied (0.1 or 0.25)
- stake_final: Final stake after all adjustments
- ece_rolling: Rolling ECE at bet time (by side)
- clv_pct_rolling: Rolling % positive CLV (by side)

Usage:
    logger = EnhancedBetLogger()

    logger.log_bet(
        bet_id='bet_001',
        date='2025 - 01 - 15',
        player='LeBron James',
        side='OVER',
        line=35.5,
        entry_odds_dec=1.91,
        prob_raw=0.65,
        prob_cal=0.60,
        ev_pre=0.08,
        ev_post=0.05,
        kelly_diagnostics={'kelly_base': 0.055, 'kelly_mult': 0.8, ...},
        stake=87.50
    )

Author: NBA Props Model
Date: October 25, 2025
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnhancedBetLogger:
    """
    Enhanced bet logger with calibration and Kelly sizing diagnostics.

    Extends CLVTracker with detailed diagnostic columns for model validation
    and performance analysis.
    """

    # Define schema with all columns
    SCHEMA = [
        # Core bet identification
        "bet_id",
        "date",
        "game_id",
        "player",
        "market",
        "side",
        "line",
        # Odds and pricing
        "entry_odds_dec",
        "entry_odds_american",
        "entry_time",
        "close_odds_dec",
        "close_odds_american",
        "close_time",
        # Probabilities (NEW)
        "prob_raw",  # Uncalibrated probability
        "prob_cal",  # Calibrated probability
        "q_novig",  # No-vig fair probability
        # Expected value (NEW)
        "ev_pre",  # EV before calibration
        "ev_post",  # EV after calibration
        # Kelly sizing diagnostics (NEW)
        "kelly_base",  # Full Kelly fraction
        "kelly_mult",  # ECE penalty multiplier [0, 1]
        "kelly_frac",  # Fraction applied (0.1 or 0.25)
        "stake_final",  # Final stake in dollars
        # Rolling metrics at bet time (NEW)
        "ece_rolling",  # Rolling ECE (by side)
        "clv_pct_rolling",  # Rolling % positive CLV (by side)
        # Closing line value
        "close_line",
        "clv_line",
        "clv_price",
        "clv_novig",
        "clv_pct",
        "beat_closing",
        # Results
        "result",
        "won",
        "profit",
        # Metadata
        "predicted_pra",
        "player_sigma",
        "bankroll",
        "tier",
    ]

    def __init__(self, ledger_path: str = "data/enhanced_bet_ledger.csv"):
        """
        Initialize enhanced bet logger.

        Args:
            ledger_path: Path to CSV file storing bet ledger
        """
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing ledger or create new
        if self.ledger_path.exists():
            self.ledger = pd.read_csv(self.ledger_path)
            logger.info(f"✅ Loaded {len(self.ledger)} bets from ledger")

            # Add missing columns (for backward compatibility)
            for col in self.SCHEMA:
                if col not in self.ledger.columns:
                    self.ledger[col] = np.nan
                    logger.debug(f"Added missing column: {col}")
        else:
            self.ledger = pd.DataFrame(columns=self.SCHEMA)
            logger.info(f"📝 Created new bet ledger: {self.ledger_path}")

    def log_bet(
        self,
        bet_id: str,
        date: str,
        player: str,
        side: str,
        line: float,
        entry_odds_dec: float,
        prob_raw: float,
        prob_cal: float,
        q_novig: float,
        ev_pre: float,
        ev_post: float,
        kelly_diagnostics: Dict[str, float],
        stake: float,
        entry_time: Optional[str] = None,
        game_id: Optional[str] = None,
        market: str = "PRA",
        predicted_pra: Optional[float] = None,
        player_sigma: Optional[float] = None,
        bankroll: Optional[float] = None,
        tier: Optional[int] = None,
        ece_rolling: Optional[float] = None,
        clv_pct_rolling: Optional[float] = None,
    ):
        """
        Log a new bet with full diagnostics.

        Args:
            bet_id: Unique bet identifier
            date: Game date (YYYY-MM-DD)
            player: Player name
            side: 'OVER' or 'UNDER'
            line: Betting line
            entry_odds_dec: Entry odds (decimal)
            prob_raw: Raw uncalibrated probability
            prob_cal: Calibrated probability
            q_novig: No-vig fair market probability
            ev_pre: EV before calibration
            ev_post: EV after calibration
            kelly_diagnostics: Dict from calculate_kelly_side_specific()
                             Must have keys: kelly_base, kelly_mult, kelly_frac
            stake: Final stake amount
            entry_time: Bet entry timestamp (default: now)
            game_id: NBA game ID
            market: Market type (default: "PRA")
            predicted_pra: Model prediction
            player_sigma: Player uncertainty
            bankroll: Current bankroll
            tier: Strategy tier (1, 2, or 3)
            ece_rolling: Rolling ECE at bet time
            clv_pct_rolling: Rolling % positive CLV at bet time

        Raises:
            ValueError: If bet_id already exists or required fields missing
        """
        # Check if bet already exists
        if bet_id in self.ledger["bet_id"].values:
            raise ValueError(f"Bet {bet_id} already exists")

        # Validate required Kelly diagnostics
        required_kelly_keys = ["kelly_base", "kelly_mult", "kelly_frac"]
        if not all(k in kelly_diagnostics for k in required_kelly_keys):
            raise ValueError(
                f"kelly_diagnostics missing required keys: {required_kelly_keys}"
            )  # noqa: E501

        # Convert decimal odds to American
        if entry_odds_dec >= 2.0:
            entry_odds_american = int((entry_odds_dec - 1) * 100)
        else:
            entry_odds_american = int(-100 / (entry_odds_dec - 1))

        # Default entry time
        if entry_time is None:
            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create bet record
        bet_record = {
            # Core identification
            "bet_id": bet_id,
            "date": date,
            "game_id": game_id or f"{date}_{player}",
            "player": player,
            "market": market,
            "side": side.upper(),
            "line": line,
            # Entry odds
            "entry_odds_dec": entry_odds_dec,
            "entry_odds_american": entry_odds_american,
            "entry_time": entry_time,
            # Closing line (to be filled later)
            "close_odds_dec": np.nan,
            "close_odds_american": np.nan,
            "close_time": None,
            "close_line": np.nan,
            "clv_line": np.nan,
            "clv_price": np.nan,
            "clv_novig": np.nan,
            "clv_pct": np.nan,
            "beat_closing": np.nan,
            # Probabilities
            "prob_raw": prob_raw,
            "prob_cal": prob_cal,
            "q_novig": q_novig,
            # Expected value
            "ev_pre": ev_pre,
            "ev_post": ev_post,
            # Kelly sizing
            "kelly_base": kelly_diagnostics["kelly_base"],
            "kelly_mult": kelly_diagnostics["kelly_mult"],
            "kelly_frac": kelly_diagnostics["kelly_frac"],
            "stake_final": stake,
            # Rolling metrics
            "ece_rolling": ece_rolling if ece_rolling is not None else np.nan,
            "clv_pct_rolling": (
                clv_pct_rolling if clv_pct_rolling is not None else np.nan
            ),  # noqa: E501
            # Results (to be filled later)
            "result": np.nan,
            "won": np.nan,
            "profit": np.nan,
            # Metadata
            "predicted_pra": predicted_pra,
            "player_sigma": player_sigma,
            "bankroll": bankroll,
            "tier": tier,
        }

        # Append to ledger
        self.ledger = pd.concat([self.ledger, pd.DataFrame([bet_record])], ignore_index=True)
        self._save_ledger()

        ece_str = f"{ece_rolling:.3f}" if not np.isnan(ece_rolling) else "N/A"
        logger.info(
            f"✅ Logged bet: {player} {side} {line} @ {entry_odds_dec:.2f}, "
            f"EV={ev_post:.2%}, Stake=${stake:.2f}, ECE={ece_str}"
        )

    def log_closing_line(
        self,
        bet_id: str,
        close_line: float,
        close_odds_dec: float,
        close_time: str,
        opposite_close_odds_dec: Optional[float] = None,
    ):
        """
        Log closing line for a bet (same as CLVTracker).

        Args:
            bet_id: Bet identifier
            close_line: Closing line
            close_odds_dec: Closing odds (decimal)
            close_time: Closing timestamp
            opposite_close_odds_dec: Opposite side odds (for no-vig calculation)  # noqa: E501
        """
        # Find bet
        idx = self.ledger[self.ledger["bet_id"] == bet_id].index

        if len(idx) == 0:
            logger.error(f"Bet {bet_id} not found")
            return

        idx = idx[0]
        bet = self.ledger.loc[idx]

        # Convert to American odds
        if close_odds_dec >= 2.0:
            close_odds_american = int((close_odds_dec - 1) * 100)
        else:
            close_odds_american = int(-100 / (close_odds_dec - 1))

        # Calculate CLV
        entry_line = bet["line"]
        entry_odds_dec = bet["entry_odds_dec"]
        direction = 1 if bet["side"] == "OVER" else -1

        # Line CLV
        clv_line = (close_line - entry_line) * direction

        # Price CLV
        clv_price = (close_odds_dec - entry_odds_dec) * direction

        # No-vig CLV
        entry_implied = 1 / entry_odds_dec
        close_implied = 1 / close_odds_dec

        if opposite_close_odds_dec is not None:
            close_opposite_implied = 1 / opposite_close_odds_dec
            entry_opposite_implied = 1 / entry_odds_dec

            # Remove vig
            total_entry = entry_implied + entry_opposite_implied
            total_close = close_implied + close_opposite_implied

            entry_novig = entry_implied / total_entry if total_entry > 0 else 0.5  # noqa: E501
            close_novig = close_implied / total_close if total_close > 0 else 0.5  # noqa: E501

            clv_novig = (close_novig - entry_novig) * direction
            clv_pct = clv_novig
        else:
            clv_novig = (close_implied - entry_implied) * direction
            clv_pct = clv_novig

        beat_closing = 1 if clv_novig > 0 else 0

        # Update ledger
        self.ledger.loc[idx, "close_line"] = close_line
        self.ledger.loc[idx, "close_odds_dec"] = close_odds_dec
        self.ledger.loc[idx, "close_odds_american"] = close_odds_american
        self.ledger.loc[idx, "close_time"] = close_time
        self.ledger.loc[idx, "clv_line"] = clv_line
        self.ledger.loc[idx, "clv_price"] = clv_price
        self.ledger.loc[idx, "clv_novig"] = clv_novig
        self.ledger.loc[idx, "clv_pct"] = clv_pct
        self.ledger.loc[idx, "beat_closing"] = beat_closing

        self._save_ledger()

        logger.info(
            f"✅ Logged closing line: {bet['player']} - CLV = {clv_pct:+.2%} "
            f"({'✅ BEAT' if beat_closing else '❌ LOST'})"
        )

    def log_result(self, bet_id: str, result: float):
        """
        Log game result and calculate profit.

        Args:
            bet_id: Bet identifier
            result: Actual PRA value
        """
        # Find bet
        idx = self.ledger[self.ledger["bet_id"] == bet_id].index

        if len(idx) == 0:
            logger.error(f"Bet {bet_id} not found")
            return

        idx = idx[0]
        bet = self.ledger.loc[idx]

        # Determine if won
        if bet["side"] == "OVER":
            won = 1 if result > bet["line"] else 0
        else:  # UNDER
            won = 1 if result < bet["line"] else 0

        # Calculate profit
        stake = bet["stake_final"]
        if won:
            profit = stake * (bet["entry_odds_dec"] - 1)
        else:
            profit = -stake

        # Update ledger
        self.ledger.loc[idx, "result"] = result
        self.ledger.loc[idx, "won"] = won
        self.ledger.loc[idx, "profit"] = profit

        self._save_ledger()

        logger.info(
            f"✅ Logged result: {bet['player']} {result:.1f} - "
            f"{'✅ WON' if won else '❌ LOST'} (${profit:+.2f})"
        )

    def get_rolling_clv_pct(self, side: str, window: int = 50) -> float:
        """
        Get rolling % of bets that beat closing (by side).

        Args:
            side: 'OVER' or 'UNDER'
            window: Rolling window size

        Returns:
            % of bets that beat closing (0.0 - 1.0), or np.nan if insufficient data  # noqa: E501
        """
        side_bets = self.ledger[
            (self.ledger["side"] == side) & (~self.ledger["beat_closing"].isna())
        ].tail(window)

        if len(side_bets) < 10:
            return np.nan

        return side_bets["beat_closing"].mean()

    def get_summary_stats(self) -> Dict[str, float]:
        """
        Get summary statistics for the ledger.

        Returns:
            Dictionary of summary statistics
        """
        # Overall stats
        total_bets = len(self.ledger)
        bets_with_results = self.ledger[~self.ledger["won"].isna()]
        n_results = len(bets_with_results)

        if n_results == 0:
            return {
                "total_bets": total_bets,
                "n_results": 0,
                "win_rate": np.nan,
                "roi": np.nan,
                "total_profit": np.nan,
            }

        # Win rate and profit
        win_rate = bets_with_results["won"].mean()
        total_staked = bets_with_results["stake_final"].sum()
        total_profit = bets_with_results["profit"].sum()
        roi = total_profit / total_staked if total_staked > 0 else np.nan

        # CLV stats
        bets_with_clv = self.ledger[~self.ledger["clv_pct"].isna()]
        n_clv = len(bets_with_clv)
        pct_beat_closing = bets_with_clv["beat_closing"].mean() if n_clv > 0 else np.nan
        avg_clv = bets_with_clv["clv_pct"].mean() if n_clv > 0 else np.nan

        # Calibration stats
        avg_prob_raw = self.ledger["prob_raw"].mean()
        avg_prob_cal = self.ledger["prob_cal"].mean()
        avg_ev_pre = self.ledger["ev_pre"].mean()
        avg_ev_post = self.ledger["ev_post"].mean()

        # Kelly stats
        avg_kelly_mult = self.ledger["kelly_mult"].mean()
        avg_ece_rolling = self.ledger["ece_rolling"].mean()

        return {
            "total_bets": total_bets,
            "n_results": n_results,
            "win_rate": win_rate,
            "roi": roi,
            "total_staked": total_staked,
            "total_profit": total_profit,
            "n_clv": n_clv,
            "pct_beat_closing": pct_beat_closing,
            "avg_clv": avg_clv,
            "avg_prob_raw": avg_prob_raw,
            "avg_prob_cal": avg_prob_cal,
            "avg_ev_pre": avg_ev_pre,
            "avg_ev_post": avg_ev_post,
            "avg_kelly_mult": avg_kelly_mult,
            "avg_ece_rolling": avg_ece_rolling,
        }

    def print_summary(self):
        """Print formatted summary of betting performance."""
        stats = self.get_summary_stats()

        print("\n" + "=" * 80)
        print("ENHANCED BET LEDGER SUMMARY")
        print("=" * 80)

        print("\n📊 Overview:")
        print(f"   Total bets logged: {stats['total_bets']}")
        print(f"   Bets with results: {stats['n_results']}")
        print(f"   Bets with CLV: {stats['n_clv']}")

        if stats["n_results"] > 0:
            print("\n💰 Performance:")
            print(f"   Win Rate: {stats['win_rate']:.1%}")
            print(f"   Total Staked: ${stats['total_staked']:,.2f}")
            print(f"   Total Profit: ${stats['total_profit']:+,.2f}")
            print(f"   ROI: {stats['roi']:+.1%}")

        if stats["n_clv"] > 0:
            print("\n📈 Closing Line Value:")
            print(f"   % Beat Closing: {stats['pct_beat_closing']:.1%}")
            print(f"   Avg CLV: {stats['avg_clv']:+.2%}")

        print("\n🎯 Calibration:")
        print(f"   Avg Prob (raw): {stats['avg_prob_raw']:.3f}")
        print(f"   Avg Prob (cal): {stats['avg_prob_cal']:.3f}")
        print(f"   Avg EV (pre):   {stats['avg_ev_pre']:.2%}")
        print(f"   Avg EV (post):  {stats['avg_ev_post']:.2%}")

        print("\n🔧 Kelly Sizing:")
        print(f"   Avg ECE Multiplier: {stats['avg_kelly_mult']:.2f}")
        print(f"   Avg Rolling ECE: {stats['avg_ece_rolling']:.3f}")

        print("\n" + "=" * 80)

    def _save_ledger(self):
        """Save ledger to CSV."""
        # Ensure columns are in schema order
        self.ledger = self.ledger[self.SCHEMA]
        self.ledger.to_csv(self.ledger_path, index=False)


# Example usage
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("ENHANCED BET LOGGER TESTING")
    print("=" * 80)

    # Initialize logger
    logger_instance = EnhancedBetLogger(ledger_path="/tmp/enhanced_bet_ledger_test.csv")

    # Simulate Kelly diagnostics
    kelly_diag = {
        "kelly_base": 0.055,
        "kelly_mult": 0.85,
        "kelly_frac": 0.1,
        "ece_rolling": 0.042,
        "stake_before_cap": 467.50,
        "stake_final": 467.50,
    }

    # Log a bet
    print("\n" + "=" * 80)
    print("TEST 1: Log Bet")
    print("=" * 80)

    logger_instance.log_bet(
        bet_id="bet_001",
        date="2025 - 01 - 15",
        player="LeBron James",
        side="OVER",
        line=35.5,
        entry_odds_dec=1.91,
        prob_raw=0.65,
        prob_cal=0.60,
        q_novig=0.52,
        ev_pre=0.08,
        ev_post=0.05,
        kelly_diagnostics=kelly_diag,
        stake=467.50,
        predicted_pra=38.2,
        player_sigma=6.5,
        bankroll=10000,
        tier=1,
        ece_rolling=0.042,
        clv_pct_rolling=0.58,
    )

    # Log closing line
    print("\n" + "=" * 80)
    print("TEST 2: Log Closing Line")
    print("=" * 80)

    logger_instance.log_closing_line(
        bet_id="bet_001",
        close_line=36.5,
        close_odds_dec=1.87,
        close_time="2025 - 01 - 15 19:28:00",
        opposite_close_odds_dec=1.95,
    )

    # Log result
    print("\n" + "=" * 80)
    print("TEST 3: Log Result")
    print("=" * 80)

    logger_instance.log_result(bet_id="bet_001", result=42.0)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST 4: Summary Statistics")
    print("=" * 80)

    logger_instance.print_summary()

    # Inspect ledger
    print("\n" + "=" * 80)
    print("TEST 5: Ledger Inspection")
    print("=" * 80)

    print("\nLedger columns:")
    print(logger_instance.ledger.columns.tolist())

    print("\nFirst bet:")
    print(logger_instance.ledger.iloc[0].to_dict())

    print("\n" + "=" * 80)
    print("✅ ENHANCED BET LOGGER TEST COMPLETE")
    print("=" * 80)
