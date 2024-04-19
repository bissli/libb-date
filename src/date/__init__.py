import datetime as _datetime
import pandas as pd
import numpy as np

from typing_extensions import Optional
from typing_extensions import Union
from typing_extensions import overload

from date.date import Date
from date.date import DateTime
from date.date import Entity
from date.date import Interval
from date.date import IntervalError
from date.date import LCL
from date.date import NYSE
from date.date import Time
from date.date import WeekDay
from date.date import expect_date
from date.date import expect_datetime
from date.date import expect_native_timezone
from date.date import expect_utc_timezone
from date.date import now
from date.date import parse
from date.date import prefer_native_timezone
from date.date import prefer_utc_timezone
from date.date import Timezone
from date.date import today


timezone = Timezone


def date(*args, **kwargs):
    return Date(*args, **kwargs)


def datetime(*args, **kwargs):
    return DateTime(*args, **kwargs)


def time(*args, **kwargs):
    return Time(*args, **kwargs)


def to_date(s: Union[str, _datetime.date, _datetime.datetime, pd.Timestamp,
             np.datetime64, Date, DateTime],
            fmt: str = None,
            raise_err: bool = False,
            shortcodes: bool = True
) -> Optional[Date]:
    """Return a new Date instance from parsed argument
    """
    return Date.parse(s, fmt, raise_err, shortcodes)


def to_datetime(s: Union[str, _datetime.date, _datetime.datetime, pd.Timestamp,
             np.datetime64, Date, DateTime],
                raise_err=False,
) -> Optional[DateTime]:
    """Return a new DateTime instance from parsed argument
    """
    return DateTime.parse(s, raise_err)


def to_time(s: Union[str, _datetime.time, Time],
            fmt: str = None,
            raise_err: bool = False):
    """Return a new Time instance from parsed argument
    """
    return Time.parse(s, fmt, raise_err)


__all__ = [
    'Date',
    'DateTime',
    'Interval',
    'IntervalError',
    'Time',
    'WeekDay',
    'now',
    'today',
    'parse',
    'LCL',
    'timezone',
    'expect_native_timezone',
    'expect_utc_timezone',
    'prefer_native_timezone',
    'prefer_utc_timezone',
    'expect_date',
    'expect_datetime',
    'Entity',
    'NYSE',
    'to_date',
    'to_datetime',
    'to_time',
    'date',
    'datetime',
    'time'
    ]
