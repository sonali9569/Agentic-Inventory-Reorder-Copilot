"""
Disk cache for fitted per-SKU Prophet models, so the live dashboard can use a
real forecast instead of engine.py's trend proxy without refitting on every
page load. engine.py and forecast.py remain pure functions with no file I/O;
this is the one module that reads and writes the cache.

Setup: `python3 -c "import cmdstanpy; cmdstanpy.install_cmdstan()"` once,
then `python3 src/prefit_forecasts.py` to fit and cache every SKU.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
from prophet.serialize import model_from_json, model_to_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import fit_item_forecast, precompute_forecast_table

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "forecast_cache"


def _cache_key(item_id: str, sales: pd.DataFrame) -> str:
    """Keyed on the item's own sales history, not the whole file's mtime --
    regenerating the dataset invalidates every SKU's cache exactly once
    (the hash changes), but re-running this on an unchanged file is a cheap
    hit even if unrelated rows elsewhere were touched by something else."""
    hist = sales[sales.item_id == item_id][["date", "units_sold"]].sort_values("date")
    digest = hashlib.sha256(pd.util.hash_pandas_object(hist).values.tobytes()).hexdigest()[:16]
    return f"{item_id}_{digest}"


def load_or_fit(item_id: str, sales: pd.DataFrame, festival_calendar: pd.DataFrame,
                 promotions: pd.DataFrame, train_end_date: pd.Timestamp,
                 cache_dir: Path = DEFAULT_CACHE_DIR):
    """
    A fitted Prophet model for one item: from disk if this exact (item,
    sales-history) pair was already fit, otherwise fits it now and saves it.
    Prefer prefit_all() for the normal case -- fitting here is a cold-start
    fallback, and it is exactly as slow as the backtest's own fit, which is
    why the dashboard should not depend on this path running on every load.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_cache_key(item_id, sales)}.json"
    if path.exists():
        return model_from_json(path.read_text())

    model = fit_item_forecast(item_id, sales, festival_calendar, promotions, train_end_date)
    path.write_text(model_to_json(model))
    return model


def prefit_all(items: pd.DataFrame, sales: pd.DataFrame, festival_calendar: pd.DataFrame,
               promotions: pd.DataFrame, train_end_date: pd.Timestamp,
               cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """
    Fits and caches every SKU once, offline -- the expensive step, meant to be
    run after regenerating the dataset or changing the forecast model, never
    from inside a dashboard request. Returns {"fit": n, "cached": n, "failed":
    [item_id, ...]} rather than raising on one bad SKU, so one item with too
    little history (a genuinely new product) doesn't block every other item
    from getting a real forecast.
    """
    result = {"fit": 0, "cached": 0, "failed": []}
    for item_id in items["item_id"]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{_cache_key(item_id, sales)}.json"
        if path.exists():
            result["cached"] += 1
            continue
        try:
            model = fit_item_forecast(item_id, sales, festival_calendar, promotions, train_end_date)
            path.write_text(model_to_json(model))
            result["fit"] += 1
        except Exception as e:
            result["failed"].append((item_id, repr(e)))
    return result


def forecast_table_or_none(item_id: str, sales: pd.DataFrame, festival_calendar: pd.DataFrame,
                            promotions: pd.DataFrame, as_of_date: pd.Timestamp, horizon_days: int,
                            cache_dir: Path = DEFAULT_CACHE_DIR) -> "pd.Series | None":
    """
    The one function callers outside this module should use. Returns a
    forecast covering [as_of_date, as_of_date + horizon_days], or None on any
    failure -- no cached model and fitting fails (too little history, a bad
    date), a corrupt cache file, anything. The caller (gather_signals, via
    assess_item's forecast_table=None default) is expected to fall back to
    the trend proxy on None, never to block on a live fit or raise into the
    dashboard.
    """
    try:
        model = load_or_fit(item_id, sales, festival_calendar, promotions, as_of_date, cache_dir)
        dates = pd.date_range(as_of_date, as_of_date + pd.Timedelta(days=horizon_days))
        return precompute_forecast_table(model, item_id, dates, promotions)
    except Exception:
        return None


def forecast_tables_for(item_ids, sales: pd.DataFrame, festival_calendar: pd.DataFrame,
                         promotions: pd.DataFrame, as_of_date: pd.Timestamp, horizon_days: int,
                         cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Convenience for the dashboard/graph: builds the {item_id: pd.Series}
    dict make_gather_signals_node's forecast_tables expects, silently omitting
    any item that failed (see forecast_table_or_none) rather than raising."""
    tables = {}
    for item_id in item_ids:
        table = forecast_table_or_none(item_id, sales, festival_calendar, promotions,
                                        as_of_date, horizon_days, cache_dir)
        if table is not None:
            tables[item_id] = table
    return tables
