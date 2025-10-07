from datetime import datetime, time
import pytz

def trading_date_to_utc(from_date: str, to_date: str,
                        exchange_tz: str = "America/New_York"):
    """
    Convert local trading date range to UTC timestamps (seconds).
    Args:
        from_date (str): Start date in "YYYY-MM-DD" format (inclusive).
        to_date (str): End date in "YYYY-MM-DD" format (inclusive).
        exchange_tz (str): Exchange time zone (default: "America/New_York").
    Returns:
        tuple[int, int]:
            (utc_start_timestamp, utc_end_timestamp)
    Example:
        trading_date_to_utc("2025-09-01", "2025-09-30")
    """
    local_tz = pytz.timezone(exchange_tz)

    # start/end time
    local_start = local_tz.localize(datetime.combine(
        datetime.strptime(from_date, "%Y-%m-%d"), time(0, 0, 0)
    ))
    local_end = local_tz.localize(datetime.combine(
        datetime.strptime(to_date, "%Y-%m-%d"), time(23, 59, 59)
    ))

    # convert to UTC
    utc_start = local_start.astimezone(pytz.utc)
    utc_end = local_end.astimezone(pytz.utc)

    return int(utc_start.timestamp()), int(utc_end.timestamp())