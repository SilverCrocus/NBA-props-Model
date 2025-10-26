"""
Closing Line Value (CLV) Tracker

Tracks CLV for betting model validation. CLV is the gold standard early signal
for model profitability - provides statistical significance in 50 bets vs 2000+
needed for ROI validation.

Research shows:
- Positive CLV bettors: +19.62% ROI (n=57k bets)
- Negative CLV bettors: -22.35% ROI
- CLV stabilizes 40 - 100x faster than P&L

Benchmarks:
- 50 - 52%: Break-even after vig
- 55 - 60%: Sharp bettor range (TARGET)
- 60 - 70%: Elite professional
- +2% avg CLV → +4% ROI

Usage:
    tracker = CLVTracker()

    # Log bet at entry
    tracker.log_bet_entry(
        bet_id="bet_001",
        date="2024 - 10 - 23",
        player="LeBron James",
        market="PRA",
        side="OVER",
        line=35.5,
        entry_odds_dec=1.91,
        entry_time="2024 - 10 - 23 19:00:00"
    )

    # Log closing line (T-2 minutes before game)
    tracker.log_closing_line(
        bet_id="bet_001",
        close_line=36.5,
        close_odds_dec=1.87,
        close_time="2024 - 10 - 23 19:28:00"
    )

    # Get CLV report
    report = tracker.get_clv_report()
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def remove_vig(over_prob, under_prob):
    """Remove vig to get fair market probabilities"""
    total = over_prob + under_prob
    if total <= 0:
        return 0.5, 0.5
    return over_prob / total, under_prob / total


class CLVTracker:
    """
    Track Closing Line Value for betting model validation

    CLV = difference between our entry price and closing market price
    Positive CLV = we got better odds than closing (good)
    Negative CLV = market moved against us (bad)

    Methods:
    - No-vig probability difference (most accurate)
    - Line movement tracking
    - Statistical significance testing
    """

    def __init__(self, ledger_path: str = "data/clv_ledger.csv"):
        """
        Initialize CLV tracker

        Args:
            ledger_path: Path to CSV file storing CLV ledger
        """
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing ledger or create new
        if self.ledger_path.exists():
            self.ledger = pd.read_csv(self.ledger_path)
            print(f"✅ Loaded {len(self.ledger)} bets from CLV ledger")
        else:
            self.ledger = pd.DataFrame(
                columns=[
                    "bet_id",
                    "date",
                    "game_id",
                    "player",
                    "market",
                    "side",
                    "line",
                    "entry_odds_dec",
                    "entry_time",
                    "close_line",
                    "close_odds_dec",
                    "close_time",
                    "direction",
                    "clv_line",
                    "clv_price",
                    "clv_novig",
                    "clv_pct",
                    "beat_closing",
                    "result",
                    "won",
                    "profit",
                ]
            )
            print(f"📝 Created new CLV ledger: {self.ledger_path}")

    def log_bet_entry(
        self,
        bet_id: str,
        date: str,
        player: str,
        market: str,
        side: str,
        line: float,
        entry_odds_dec: float,
        entry_time: str,
        game_id: Optional[str] = None,
        predicted_pra: Optional[float] = None,
        player_sigma: Optional[float] = None,
        ev: Optional[float] = None,
        stake: Optional[float] = None,
    ):
        """
        Log bet at entry time

        Args:
            bet_id: Unique bet identifier
            date: Game date (YYYY-MM-DD)
            player: Player name
            market: Market type (e.g., "PRA", "Points", "Rebounds")
            side: OVER or UNDER
            line: Betting line (e.g., 35.5)
            entry_odds_dec: Entry odds in decimal (e.g., 1.91 for -110)
            entry_time: Entry timestamp (YYYY-MM-DD HH:MM:SS)
            game_id: NBA game ID (optional)
            predicted_pra: Model prediction (optional, for analysis)
            player_sigma: Player variance (optional, for analysis)
            ev: Expected value at entry (optional, for analysis)
            stake: Bet size in dollars (optional)
        """
        # Check if bet already exists
        if bet_id in self.ledger["bet_id"].values:
            print(f"⚠️  Bet {bet_id} already exists, skipping")
            return

        # Determine direction (+1 for OVER, -1 for UNDER)
        direction = 1 if side.upper() == "OVER" else -1

        # Create new row
        new_bet = {
            "bet_id": bet_id,
            "date": date,
            "game_id": game_id or f"{date}_{player}",
            "player": player,
            "market": market,
            "side": side.upper(),
            "line": line,
            "entry_odds_dec": entry_odds_dec,
            "entry_time": entry_time,
            "direction": direction,
            # Closing line fields (to be filled later)
            "close_line": np.nan,
            "close_odds_dec": np.nan,
            "close_time": None,
            "clv_line": np.nan,
            "clv_price": np.nan,
            "clv_novig": np.nan,
            "clv_pct": np.nan,
            "beat_closing": np.nan,
            # Result fields (to be filled after game)
            "result": np.nan,
            "won": np.nan,
            "profit": np.nan,
            # Optional analysis fields
            "predicted_pra": predicted_pra,
            "player_sigma": player_sigma,
            "ev_entry": ev,
            "stake": stake,
        }

        # Append to ledger
        self.ledger = pd.concat([self.ledger, pd.DataFrame([new_bet])], ignore_index=True)
        self._save_ledger()

        print(f"✅ Logged bet entry: {player} {side} {line} @ {entry_odds_dec}")

    def log_closing_line(
        self,
        bet_id: str,
        close_line: float,
        close_odds_dec: float,
        close_time: str,
        opposite_close_odds_dec: Optional[float] = None,
    ):
        """
        Log closing line (capture T-2 minutes before game start)

        Args:
            bet_id: Bet identifier
            close_line: Closing line number
            close_odds_dec: Closing odds for our side (decimal)
            close_time: Closing line timestamp
            opposite_close_odds_dec: Closing odds for opposite side (for no-vig calc)  # noqa: E501
        """
        # Find bet
        idx = self.ledger[self.ledger["bet_id"] == bet_id].index

        if len(idx) == 0:
            print(f"❌ Bet {bet_id} not found in ledger")
            return

        idx = idx[0]

        # Get bet details
        bet = self.ledger.loc[idx]
        entry_line = bet["line"]
        entry_odds_dec = bet["entry_odds_dec"]
        direction = bet["direction"]

        # Calculate CLV components

        # 1. Line CLV (did the line move in our favor?)
        # Positive = line moved in our favor (we got better line)
        clv_line = (close_line - entry_line) * direction

        # 2. Price CLV (did odds get worse for our side?)
        # Positive = we got better price (odds got worse after our bet)
        clv_price = (close_odds_dec - entry_odds_dec) * direction

        # 3. No-vig CLV (most accurate - uses implied probabilities)
        entry_implied = 1 / entry_odds_dec
        close_implied = 1 / close_odds_dec

        # If we have opposite side odds, remove vig
        if opposite_close_odds_dec is not None:
            close_opposite_implied = 1 / opposite_close_odds_dec

            # Entry (use typical -110/-110 for opposite if not available)
            entry_opposite_implied = 1 / entry_odds_dec  # Assume symmetric

            # Remove vig
            entry_novig, _ = remove_vig(entry_implied, entry_opposite_implied)
            close_novig, _ = remove_vig(close_implied, close_opposite_implied)

            # CLV = difference in no-vig probabilities
            # Negative CLV = we got better price (our prob is lower = better
            # odds)
            clv_novig = (close_novig - entry_novig) * direction
            clv_pct = clv_novig
        else:
            # Fallback: use raw implied prob difference
            clv_novig = (close_implied - entry_implied) * direction
            clv_pct = clv_novig

        # Beat closing line? (positive CLV)
        beat_closing = 1 if clv_novig > 0 else 0

        # Update ledger
        self.ledger.loc[idx, "close_line"] = close_line
        self.ledger.loc[idx, "close_odds_dec"] = close_odds_dec
        self.ledger.loc[idx, "close_time"] = close_time
        self.ledger.loc[idx, "clv_line"] = clv_line
        self.ledger.loc[idx, "clv_price"] = clv_price
        self.ledger.loc[idx, "clv_novig"] = clv_novig
        self.ledger.loc[idx, "clv_pct"] = clv_pct
        self.ledger.loc[idx, "beat_closing"] = beat_closing

        self._save_ledger()

        beat_status = "✅ BEAT" if beat_closing else "❌ LOST"
        print(
            f"✅ Logged closing line: {bet['player']} - CLV = {clv_pct:+.2%} " f"({beat_status})"
        )  # noqa: E501

    def log_result(self, bet_id: str, result: float, stake: Optional[float] = None):
        """
        Log game result and calculate profit/loss

        Args:
            bet_id: Bet identifier
            result: Actual value (e.g., actual PRA)
            stake: Bet size (if not already logged)
        """
        # Find bet
        idx = self.ledger[self.ledger["bet_id"] == bet_id].index

        if len(idx) == 0:
            print(f"❌ Bet {bet_id} not found in ledger")
            return

        idx = idx[0]
        bet = self.ledger.loc[idx]

        # Determine if won
        if bet["side"] == "OVER":
            won = 1 if result > bet["line"] else 0
        else:  # UNDER
            won = 1 if result < bet["line"] else 0

        # Calculate profit
        if stake is None:
            stake = bet.get("stake", 1.0)  # Default to $1 if not specified

        if won:
            profit = stake * (bet["entry_odds_dec"] - 1)
        else:
            profit = -stake

        # Update ledger
        self.ledger.loc[idx, "result"] = result
        self.ledger.loc[idx, "won"] = won
        self.ledger.loc[idx, "profit"] = profit
        if "stake" not in bet or pd.isna(bet["stake"]):
            self.ledger.loc[idx, "stake"] = stake

        self._save_ledger()

        win_status = "✅ WON" if won else "❌ LOST"
        print(
            f"✅ Logged result: {
                bet['player']} {
                result:.1f} - "
            f"{win_status} (${
                profit:+.2f})"
        )

    def get_clv_report(self, min_bets: int = 10) -> Dict:
        """
        Generate CLV validation report

        Args:
            min_bets: Minimum bets required for statistical significance

        Returns:
            Dict with CLV statistics and validation results
        """
        # Filter to bets with closing lines
        clv_bets = self.ledger[~self.ledger["clv_pct"].isna()].copy()

        if len(clv_bets) < min_bets:
            return {
                "status": "INSUFFICIENT_DATA",
                "n_bets": len(clv_bets),
                "min_required": min_bets,
                "message": f"Need {
                    min_bets -
                    len(clv_bets)} more bets for validation",
            }

        # Ensure numeric types
        clv_bets["clv_pct"] = pd.to_numeric(clv_bets["clv_pct"], errors="coerce")
        clv_bets["beat_closing"] = pd.to_numeric(clv_bets["beat_closing"], errors="coerce")

        # Calculate statistics
        pct_beat_closing = clv_bets["beat_closing"].mean()
        avg_clv = clv_bets["clv_pct"].mean()
        median_clv = clv_bets["clv_pct"].median()
        std_clv = clv_bets["clv_pct"].std()

        # Statistical significance (t-test against 0)
        from scipy.stats import ttest_1samp

        t_stat, p_value = ttest_1samp(clv_bets["clv_pct"].dropna().values, 0)

        # Validation thresholds
        TARGET_PCT_BEAT = 0.55  # 55% beat closing
        TARGET_AVG_CLV = 0.02  # 2% average CLV

        passes_pct = pct_beat_closing >= TARGET_PCT_BEAT
        passes_avg = avg_clv >= TARGET_AVG_CLV
        passes_sig = p_value < 0.05 and avg_clv > 0

        # Overall validation
        if passes_pct and passes_avg and passes_sig:
            status = "PASS"
            message = "✅ Model shows positive CLV - proceed with confidence"
        elif passes_pct and avg_clv > 0:
            status = "MARGINAL"
            message = "⚠️  Positive CLV but below target - continue monitoring"
        else:
            status = "FAIL"
            message = "❌ Negative or insufficient CLV - investigate model"

        # Calculate ROI if results available
        results_bets = clv_bets[~clv_bets["won"].isna()]
        if len(results_bets) > 0:
            total_staked = results_bets["stake"].sum()
            total_profit = results_bets["profit"].sum()
            roi = total_profit / total_staked if total_staked > 0 else 0
            win_rate = results_bets["won"].mean()
        else:
            roi = None
            win_rate = None

        return {
            "status": status,
            "message": message,
            "n_bets": len(clv_bets),
            "pct_beat_closing": pct_beat_closing,
            "avg_clv": avg_clv,
            "median_clv": median_clv,
            "std_clv": std_clv,
            "t_statistic": t_stat,
            "p_value": p_value,
            "passes_pct_threshold": passes_pct,
            "passes_avg_threshold": passes_avg,
            "statistically_significant": passes_sig,
            "win_rate": win_rate,
            "roi": roi,
            "target_pct_beat": TARGET_PCT_BEAT,
            "target_avg_clv": TARGET_AVG_CLV,
        }

    def print_clv_report(self):
        """Print formatted CLV validation report"""
        report = self.get_clv_report()

        print("\n" + "=" * 80)
        print("CLV VALIDATION REPORT")
        print("=" * 80)

        if report["status"] == "INSUFFICIENT_DATA":
            # Check if we have results data to show performance report instead
            results_bets = self.ledger[~self.ledger["won"].isna()]

            if len(results_bets) >= 10:
                print(f"\n⚠️  No closing lines captured (0/{len(self.ledger)} bets)")  # noqa: E501
                print("   CLV validation requires closing line data")
                print(
                    f"\n   However, you have {
                        len(results_bets)} bets with results!"
                )
                print("   Showing PERFORMANCE REPORT instead...\n")

                self.print_performance_report()
                return
            else:
                print(f"\n⚠️  {report['message']}")
                return

        print(f"\n📊 CLV Statistics ({report['n_bets']} bets):")
        print(
            f"   % Beat Closing:     {report['pct_beat_closing']:.1%} "
            f"(target: {report['target_pct_beat']:.0%})"
        )
        print(
            f"   Average CLV:        {report['avg_clv']:+.2%} "
            f"(target: {report['target_avg_clv']:+.0%})"
        )
        print(f"   Median CLV:         {report['median_clv']:+.2%}")
        print(f"   Std Dev:            {report['std_clv']:.2%}")

        print("\n📈 Statistical Significance:")
        print(f"   T-statistic:        {report['t_statistic']:.2f}")
        print(f"   P-value:            {report['p_value']:.4f}")
        sig_status = "✅ YES" if report["statistically_significant"] else "❌ NO"
        print(f"   Significant:        {sig_status}")

        if report["win_rate"] is not None:
            print("\n💰 Betting Performance:")
            print(f"   Win Rate:           {report['win_rate']:.1%}")
            print(f"   ROI:                {report['roi']:+.1%}")

        print("\n🎯 Validation:")
        pct_status = "✅ PASS" if report["passes_pct_threshold"] else "❌ FAIL"
        avg_status = "✅ PASS" if report["passes_avg_threshold"] else "❌ FAIL"
        sig_status2 = "✅ PASS" if report["statistically_significant"] else "❌ FAIL"  # noqa: E501
        print(f"   % Beat Closing:     {pct_status}")
        print(f"   Average CLV:        {avg_status}")
        print(f"   Statistical Sig:    {sig_status2}")

        print(f"\n{report['message']}")
        print("=" * 80)

    def print_performance_report(self):
        """Print performance report based on actual bet results (no closing lines needed)"""  # noqa: E501
        results_bets = self.ledger[~self.ledger["won"].isna()].copy()

        if len(results_bets) == 0:
            print("\n⚠️  No bet results available yet")
            print("   Log results using: uv run log_results.py <date>")
            return

        print("=" * 80)
        print("BETTING PERFORMANCE REPORT")
        print("=" * 80)

        # Basic stats
        total_bets = len(results_bets)
        wins = int(results_bets["won"].sum())
        losses = total_bets - wins
        win_rate = results_bets["won"].mean()

        print(f"\n📊 Win/Loss Record ({total_bets} bets):")
        print(f"   Wins:               {wins}")
        print(f"   Losses:             {losses}")
        print(f"   Win Rate:           {win_rate:.1%}")

        # Breakeven calculation
        avg_odds = results_bets["entry_odds_dec"].mean()
        implied_prob = 1 / avg_odds
        breakeven_rate = implied_prob

        print("\n📈 Performance vs Breakeven:")
        print(f"   Avg Entry Odds:     {avg_odds:.2f} (decimal)")
        print(f"   Breakeven Rate:     {breakeven_rate:.1%}")
        print(f"   Actual Rate:        {win_rate:.1%}")

        if win_rate > breakeven_rate:
            edge = (win_rate - breakeven_rate) * 100
            print(f"   Edge:               +{edge:.1f}% ✅")
        else:
            edge = (win_rate - breakeven_rate) * 100
            print(f"   Edge:               {edge:.1f}% ⚠️")

        # P&L
        total_staked = results_bets["stake"].sum()
        total_profit = results_bets["profit"].sum()
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

        print("\n💰 Profit & Loss:")
        print(f"   Total Staked:       ${total_staked:,.2f}")
        print(f"   Total Profit:       ${total_profit:+,.2f}")
        print(f"   ROI:                {roi:+.2f}%")

        # Statistical significance test on win rate
        try:
            from scipy.stats import binomtest

            result = binomtest(wins, total_bets, breakeven_rate, alternative="greater")
            p_value = result.pvalue
        except ImportError:
            # Fallback for older scipy versions
            from scipy.stats import binom_test

            p_value = binom_test(wins, total_bets, breakeven_rate, alternative="greater")

        print("\n📊 Statistical Significance:")
        print(f"   H0: Win rate = {breakeven_rate:.1%} (breakeven)")
        print(f"   H1: Win rate > {breakeven_rate:.1%}")
        print(f"   P-value:            {p_value:.4f}")

        if p_value < 0.05:
            print(
                "   Result:             ✅ Statistically significant edge (p < 0.05)"
            )  # noqa: E501
        elif p_value < 0.10:
            print("   Result:             ⚠️  Marginally significant (p < 0.10)")  # noqa: E501
        else:
            print("   Result:             ❌ Not significant (need more bets)")

        # Sample size recommendation
        if total_bets < 50:
            needed = 50 - total_bets
            print("\n💡 Recommendation:")
            print(f"   Current sample: {total_bets} bets")
            print(f"   Need {needed} more bets for stronger validation")
        elif total_bets < 200:
            needed = 200 - total_bets
            print("\n💡 Recommendation:")
            print(f"   Good sample size! Collect {needed} more for high confidence")  # noqa: E501
        else:
            print("\n💡 Recommendation:")
            print(f"   Excellent sample size ({total_bets} bets)")

        # Breakdown by side
        print("\n📋 Breakdown by Bet Type:")
        for side in ["OVER", "UNDER"]:
            side_bets = results_bets[results_bets["side"] == side]
            if len(side_bets) > 0:
                side_wins = side_bets["won"].sum()
                side_rate = side_bets["won"].mean()
                side_profit = side_bets["profit"].sum()
                print(
                    f"   {side:6s}:  {int(side_wins)}/{len(side_bets)} wins "
                    f"({side_rate:.1%}) | Profit: ${side_profit:+.2f}"
                )

        # Overall assessment
        print("\n🎯 Overall Assessment:")

        if win_rate >= 0.55 and roi > 0:
            print("   ✅ STRONG PERFORMANCE")
            print("      - Win rate above 55%")
            print("      - Positive ROI")
            print("      - Continue current strategy")
        elif win_rate >= breakeven_rate and roi > 0:
            print("   ✅ PROFITABLE")
            print("      - Beating breakeven rate")
            print("      - Positive ROI")
            print("      - On track for long-term profit")
        elif win_rate >= breakeven_rate and roi < 0:
            print("   ⚠️  MARGINAL")
            print("      - Win rate at/above breakeven")
            print("      - Negative ROI (likely variance)")
            print("      - Need more bets to confirm edge")
        else:
            print("   ⚠️  UNDERPERFORMING")
            print("      - Win rate below breakeven")
            print("      - Review bet selection criteria")
            print("      - Consider adjusting strategy")

        print("\n" + "=" * 80)

    def _save_ledger(self):
        """Save ledger to CSV"""
        self.ledger.to_csv(self.ledger_path, index=False)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("CLV TRACKER EXAMPLE")
    print("=" * 80)

    # Initialize tracker
    tracker = CLVTracker(ledger_path="data/clv_ledger_example.csv")

    # Simulate 50 bets
    np.random.seed(42)

    for i in range(50):
        bet_id = f"bet_{i:03d}"

        # Entry
        tracker.log_bet_entry(
            bet_id=bet_id,
            date="2024 - 10 - 23",
            player=f"Player_{i % 10}",
            market="PRA",
            side=np.random.choice(["OVER", "UNDER"]),
            line=30 + np.random.uniform(-10, 10),
            entry_odds_dec=1.91,
            entry_time="2024 - 10 - 23 18:00:00",
            ev=0.05 + np.random.uniform(-0.03, 0.03),
        )

        # Closing line (simulate +2% average CLV)
        if np.random.rand() < 0.58:  # 58% beat closing
            close_odds = 1.91 - np.random.uniform(0.02, 0.08)  # Better for us
        else:
            close_odds = 1.91 + np.random.uniform(0.02, 0.08)  # Worse for us

        tracker.log_closing_line(
            bet_id=bet_id,
            close_line=30 + np.random.uniform(-10, 10),
            close_odds_dec=close_odds,
            close_time="2024 - 10 - 23 18:58:00",
            opposite_close_odds_dec=1.91,
        )

        # Result (54% win rate)
        result = 30 + np.random.uniform(-15, 15)
        tracker.log_result(bet_id=bet_id, result=result, stake=100)

    # Print report
    tracker.print_clv_report()
