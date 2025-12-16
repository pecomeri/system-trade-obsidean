from backtest_core import run_backtest

def run_hyp(config):
    """
    config 例：
    {
      "family": "A",
      "hyp": "A002",
      "only_session": "W2",
      "strong_trend_only": False,
      "disable_daily_stop": False,
      "from_month": "2024-01",
      "to_month": "2024-12",
    }
    """
    return run_backtest(config)

