#!/usr/bin/env python3
"""
Fit Side-Specific Beta Calibrator for NBA Props Betting

Trains separate calibrators for OVER and UNDER bets using beta calibration
(or Platt scaling fallback for small samples).

Usage:
    uv run python scripts/calibration/fit_beta_calibrator.py --validate
    uv run python scripts/calibration/fit_beta_calibrator.py --train-only

Requirements:
    - data/clv_ledger.csv with 100+ completed bets
    - Columns: predicted_pra, result, line, side, won, player_sigma, entry_odds_dec  # noqa: E501
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_bet_ledger():
    """Load bet ledger with completed results"""
    ledger_path = Path("data/clv_ledger.csv")

    if not ledger_path.exists():
        raise FileNotFoundError(f"Bet ledger not found: {ledger_path}")

    df = pd.read_csv(ledger_path)

    # Filter to completed bets only
    df = df[df["result"].notna() & df["won"].notna()].copy()

    print(f"✓ Loaded {len(df)} completed bets")
    print(f"   OVER:  {(df['side'] == 'OVER').sum()} bets")
    print(f"   UNDER: {(df['side'] == 'UNDER').sum()} bets")

    # Check required columns
    required = ["predicted_pra", "result", "line", "side", "won", "player_sigma"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def pra_to_probability(predictions, lines, sigmas, sides):
    """Convert PRA predictions to win probabilities using Normal CDF"""
    probs = np.zeros(len(predictions))

    for i in range(len(predictions)):
        if sides[i] == "OVER":
            probs[i] = 1 - norm.cdf(lines[i], loc=predictions[i], scale=sigmas[i])
        else:  # UNDER
            probs[i] = norm.cdf(lines[i], loc=predictions[i], scale=sigmas[i])

    return probs


def compute_ece(probs, outcomes, n_bins=5):
    """Compute Expected Calibration Error"""
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_prob = probs[mask].mean()
            bin_outcome = outcomes[mask].mean()
            bin_weight = mask.sum() / len(probs)
            ece += bin_weight * abs(bin_prob - bin_outcome)

    return ece


def fit_platt_scaling(probs, outcomes):
    """Fit Platt scaling (logistic regression on logit-transformed probs)"""
    import warnings

    from sklearn.linear_model import LogisticRegression

    warnings.filterwarnings("ignore")

    # Convert to logits
    probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(probs_clipped / (1 - probs_clipped))

    # Fit logistic regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logits.reshape(-1, 1), outcomes)

    return lr


def apply_platt_scaling(lr_model, probs):
    """Apply Platt scaling to probabilities"""
    probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    probs_cal = lr_model.predict_proba(logits.reshape(-1, 1))[:, 1]
    return probs_cal


def fit_beta_calibration(probs, outcomes):
    """Fit Beta calibration using betacal library"""
    try:
        from betacal import BetaCalibration

        # Clip probabilities to valid range
        probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)

        # Fit 3-parameter beta calibration
        bc = BetaCalibration(parameters="abm")
        bc.fit(probs_clipped, outcomes)

        return bc
    except ImportError:
        print("⚠️  betacal not installed. Install with: uv add betacal")
        print("    Falling back to Platt scaling.")
        return None


def temporal_train_test_split(df, test_fraction=0.2):
    """Split data by date (most recent = test set)"""
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - test_fraction))

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[:split_idx] = True
    test_mask = ~train_mask

    print("\n✓ Temporal split:")
    print(
        f"   Train: {
            train_mask.sum()} bets (before {
            df.iloc[split_idx]['date']})"
    )
    print(f"   Test:  {test_mask.sum()} bets (on/after {df.iloc[split_idx]['date']})")  # noqa: E501

    return train_mask, test_mask


def train_hybrid_calibrator(df, train_mask, min_samples_beta=30):
    """
    Train side-specific calibrators with hybrid approach:
    - Beta calibration if n >= min_samples_beta
    - Platt scaling if n < min_samples_beta
    """
    # Prepare data
    train_df = df[train_mask].copy()

    predictions = train_df["predicted_pra"].values
    lines = train_df["line"].values
    sigmas = train_df["player_sigma"].values
    sides = train_df["side"].values
    outcomes = train_df["won"].values

    # Convert PRA to probabilities
    probs_raw = pra_to_probability(predictions, lines, sigmas, sides)

    # Split by side
    over_mask = sides == "OVER"
    under_mask = sides == "UNDER"

    n_over = over_mask.sum()
    n_under = under_mask.sum()

    print("\nTraining calibrators:")
    print(f"  OVER:  {n_over} samples")
    print(f"  UNDER: {n_under} samples")

    calibrators = {}
    metadata = {"n_over": n_over, "n_under": n_under}

    # OVER calibrator
    if n_over >= min_samples_beta:
        bc = fit_beta_calibration(probs_raw[over_mask], outcomes[over_mask])
        if bc is not None:
            calibrators["OVER"] = ("beta", bc)
            metadata["calibrator_type_over"] = "beta"
            print(f"  → OVER: Using Beta calibration (n={n_over})")
        else:
            lr = fit_platt_scaling(probs_raw[over_mask], outcomes[over_mask])
            calibrators["OVER"] = ("platt", lr)
            metadata["calibrator_type_over"] = "platt"
            print("  → OVER: Using Platt scaling (betacal not available)")
    elif n_over >= 10:
        lr = fit_platt_scaling(probs_raw[over_mask], outcomes[over_mask])
        calibrators["OVER"] = ("platt", lr)
        metadata["calibrator_type_over"] = "platt"
        print(
            f"  → OVER: Using Platt scaling (n={n_over} < {min_samples_beta}, small sample)"
        )  # noqa: E501
    else:
        calibrators["OVER"] = None
        metadata["calibrator_type_over"] = "none"
        print(f"  → OVER: No calibration (n={n_over} < 10, insufficient data)")

    # UNDER calibrator
    if n_under >= min_samples_beta:
        bc = fit_beta_calibration(probs_raw[under_mask], outcomes[under_mask])
        if bc is not None:
            calibrators["UNDER"] = ("beta", bc)
            metadata["calibrator_type_under"] = "beta"
            print(f"  → UNDER: Using Beta calibration (n={n_under})")
        else:
            lr = fit_platt_scaling(probs_raw[under_mask], outcomes[under_mask])
            calibrators["UNDER"] = ("platt", lr)
            metadata["calibrator_type_under"] = "platt"
            print("  → UNDER: Using Platt scaling (betacal not available)")
    elif n_under >= 10:
        lr = fit_platt_scaling(probs_raw[under_mask], outcomes[under_mask])
        calibrators["UNDER"] = ("platt", lr)
        metadata["calibrator_type_under"] = "platt"
        print(
            f"  → UNDER: Using Platt scaling (n={n_under} < {min_samples_beta}, small sample)"
        )  # noqa: E501
    else:
        calibrators["UNDER"] = None
        metadata["calibrator_type_under"] = "none"
        print(f"  → UNDER: No calibration (n={n_under} < 10, insufficient data)")

    # Calculate training metrics
    probs_cal_train = apply_calibrators(calibrators, probs_raw, sides, over_mask, under_mask)

    ece_train_raw = compute_ece(probs_raw, outcomes)
    ece_train_cal = compute_ece(probs_cal_train, outcomes)
    brier_train_raw = brier_score_loss(outcomes, probs_raw)
    brier_train_cal = brier_score_loss(outcomes, probs_cal_train)

    metadata["ece_train_raw"] = ece_train_raw
    metadata["ece_train_cal"] = ece_train_cal
    metadata["brier_train_raw"] = brier_train_raw
    metadata["brier_train_cal"] = brier_train_cal

    print("\n📊 Training Metrics:")
    ece_pct = (ece_train_cal / ece_train_raw - 1) * 100
    brier_pct = (brier_train_cal / brier_train_raw - 1) * 100
    print(
        f"   ECE:   {ece_train_raw:.4f} → {ece_train_cal:.4f} " f"({ece_pct:+.1f}%)"
    )  # noqa: E501
    print(
        f"   Brier: {brier_train_raw:.4f} → {brier_train_cal:.4f} " f"({brier_pct:+.1f}%)"
    )  # noqa: E501

    return calibrators, metadata


def apply_calibrators(calibrators, probs_raw, sides, over_mask, under_mask):
    """Apply side-specific calibrators to probabilities"""
    probs_cal = probs_raw.copy()

    # OVER
    if calibrators.get("OVER") is not None:
        cal_type, cal_model = calibrators["OVER"]
        if cal_type == "beta":
            probs_cal[over_mask] = cal_model.predict(probs_raw[over_mask])
        elif cal_type == "platt":
            probs_cal[over_mask] = apply_platt_scaling(cal_model, probs_raw[over_mask])

    # UNDER
    if calibrators.get("UNDER") is not None:
        cal_type, cal_model = calibrators["UNDER"]
        if cal_type == "beta":
            probs_cal[under_mask] = cal_model.predict(probs_raw[under_mask])
        elif cal_type == "platt":
            probs_cal[under_mask] = apply_platt_scaling(cal_model, probs_raw[under_mask])

    return probs_cal


def validate_calibrator(df, calibrators, metadata, test_mask):
    """Validate calibrator on hold-out test set"""
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS (HOLD-OUT TEST SET)")
    print("=" * 70)

    # Extract test data
    test_df = df[test_mask].copy()

    predictions = test_df["predicted_pra"].values
    lines = test_df["line"].values
    sigmas = test_df["player_sigma"].values
    sides = test_df["side"].values
    outcomes = test_df["won"].values

    # Convert to probabilities
    probs_raw = pra_to_probability(predictions, lines, sigmas, sides)

    # Split by side
    over_mask = sides == "OVER"
    under_mask = sides == "UNDER"

    # Apply calibration
    probs_cal = apply_calibrators(calibrators, probs_raw, sides, over_mask, under_mask)

    # Overall metrics
    ece_raw = compute_ece(probs_raw, outcomes)
    ece_cal = compute_ece(probs_cal, outcomes)
    brier_raw = brier_score_loss(outcomes, probs_raw)
    brier_cal = brier_score_loss(outcomes, probs_cal)

    print("\n📊 Overall Metrics:")
    ece_pct = (ece_cal / ece_raw - 1) * 100
    brier_pct = (brier_cal / brier_raw - 1) * 100
    print(f"   ECE:   {ece_raw:.4f} → {ece_cal:.4f} ({ece_pct:+.1f}%)")
    print(f"   Brier: {brier_raw:.4f} → {brier_cal:.4f} ({brier_pct:+.1f}%)")

    # By side
    if over_mask.sum() > 0:
        ece_over_raw = compute_ece(probs_raw[over_mask], outcomes[over_mask])
        ece_over_cal = compute_ece(probs_cal[over_mask], outcomes[over_mask])
        wr_over = outcomes[over_mask].mean()
        print(f"\n📈 OVER ({over_mask.sum()} bets):")
        print(f"   ECE:  {ece_over_raw:.4f} → {ece_over_cal:.4f}")
        print(f"   WR:   {wr_over:.1%}")

    if under_mask.sum() > 0:
        ece_under_raw = compute_ece(probs_raw[under_mask], outcomes[under_mask])
        ece_under_cal = compute_ece(probs_cal[under_mask], outcomes[under_mask])
        wr_under = outcomes[under_mask].mean()
        print(f"\n📈 UNDER ({under_mask.sum()} bets):")
        print(f"   ECE:  {ece_under_raw:.4f} → {ece_under_cal:.4f}")
        print(f"   WR:   {wr_under:.1%}")

        # Side bias
        if over_mask.sum() > 0:
            side_gap = abs(wr_over - wr_under)
            print(f"\n⚖️  Side Bias Gap: {side_gap:.1%}")

    # Pass/fail criteria
    print("\n✅ Validation Criteria:")
    passed = True

    if ece_cal < 0.08:
        print(f"   ✅ ECE < 0.08: PASS ({ece_cal:.4f})")
    else:
        print(f"   ❌ ECE < 0.08: FAIL ({ece_cal:.4f})")
        passed = False

    if ece_cal < ece_raw:
        print("   ✅ ECE improved: PASS")
    else:
        print("   ❌ ECE improved: FAIL (got worse)")
        passed = False

    if brier_cal <= brier_raw * 1.02:  # Allow 2% Brier degradation
        print("   ✅ Brier not degraded >2%: PASS")
    else:
        print("   ❌ Brier degraded >2%: FAIL")
        passed = False

    if over_mask.sum() > 0 and under_mask.sum() > 0:
        if side_gap < 0.10:
            print(f"   ✅ Side gap < 10%: PASS ({side_gap:.1%})")
        else:
            print(f"   ⚠️  Side gap < 10%: MARGINAL ({side_gap:.1%})")

    print("=" * 70)

    return passed, {
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "brier_raw": brier_raw,
        "brier_cal": brier_cal,
    }


def save_calibrator(calibrators, metadata, save_path="models/beta_calibrator.pkl"):
    """Save calibrator to disk"""
    import pickle

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Package everything
    calibrator_package = {"calibrators": calibrators, "metadata": metadata, "version": "1.0"}

    with open(save_path, "wb") as f:
        pickle.dump(calibrator_package, f)

    print(f"\n✓ Saved calibrator to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train side-specific beta calibrator")
    parser.add_argument(
        "--validate", action="store_true", help="Run validation on hold-out set (80 / 20 split)"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Train on all data without validation"
    )
    parser.add_argument(
        "--min-samples-beta",
        type=int,
        default=30,
        help="Minimum samples for beta calibration (default: 30)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SIDE-SPECIFIC CALIBRATOR TRAINING")
    print("=" * 70)

    # Load data
    df = load_bet_ledger()

    if args.validate:
        # 80 / 20 temporal split
        train_mask, test_mask = temporal_train_test_split(df, test_fraction=0.2)

        # Train on training set
        calibrators, metadata = train_hybrid_calibrator(
            df, train_mask, min_samples_beta=args.min_samples_beta
        )

        # Validate on test set
        passed, val_metrics = validate_calibrator(df, calibrators, metadata, test_mask)

        metadata.update(val_metrics)

        if passed:
            print("\n✅ VALIDATION PASSED - Safe to deploy")
            save_calibrator(calibrators, metadata)
            return 0
        else:
            print("\n❌ VALIDATION FAILED - Do NOT deploy")
            print("    Consider:")
            print("    - Lower min_samples_beta (try 20 or 25)")
            print("    - Use Platt scaling for both sides")
            print("    - Accumulate more bet data (currently 100 bets)")
            return 1

    else:
        # Train on all data
        train_mask = np.ones(len(df), dtype=bool)
        calibrators, metadata = train_hybrid_calibrator(
            df, train_mask, min_samples_beta=args.min_samples_beta
        )

        save_calibrator(calibrators, metadata)
        print("\n✓ Training complete (no validation)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
