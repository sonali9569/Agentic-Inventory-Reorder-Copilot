"""
Offline step: fits and caches a Prophet model for every SKU, so the dashboard
never fits one inline. Run this once after `python3 data/generate_dataset.py`,
or after changing forecast.py's model. Needs CmdStan installed (see README.md
"Setup") -- that is the one dependency this script has that the rest of the
test suite does not.

    python3 src/prefit_forecasts.py

Takes roughly as long as running the backtest once (Prophet's fit dominates),
since it is fitting the same models the backtest already fits -- just saving
them to data/forecast_cache/ instead of throwing them away after one run.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_cache import prefit_all

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

if __name__ == "__main__":
    sales = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
    items = pd.read_csv(DATA / "items.csv")
    festival_calendar = pd.read_csv(DATA / "festival_calendar.csv", parse_dates=["date"])
    promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["date"])

    # fit on everything available -- this is the live cache, not a held-out
    # backtest, so there is no window to protect from leakage here
    train_end_date = sales["date"].max() + pd.Timedelta(days=1)

    result = prefit_all(items, sales, festival_calendar, promotions, train_end_date)

    print(f"{result['fit']} fitted, {result['cached']} already cached, "
          f"{len(result['failed'])} failed (out of {len(items)} SKUs)")
    for item_id, error in result["failed"]:
        print(f"  {item_id}: {error}")

    if result["failed"]:
        sys.exit(1)
