import config
from datetime import datetime, timedelta, timezone


def get_run_hours() -> int:
    week_day = datetime.now(tz=timezone(timedelta(hours=8))).weekday()
    return config.WEEKEND_RUN_HOURS if 4 <= week_day < 6 else config.DETECT_RUN_HOURS
