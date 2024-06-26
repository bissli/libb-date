import calendar
import contextlib
import datetime
import logging
import os
import re
import time
import warnings
import zoneinfo
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Callable
from enum import IntEnum
from functools import lru_cache, partial, wraps
from typing import Self

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pendulum
from dateutil import parser

from libb import is_null, issequence

warnings.simplefilter(action='ignore', category=DeprecationWarning)

logger = logging.getLogger(__name__)

__all__ = [
    'Date',
    'DateTime',
    'Interval',
    'IntervalError',
    'Time',
    'Timezone',
    'WeekDay',
    'now',
    'today',
    'parse',
    'EST',
    'UTC',
    'GMT',
    'LCL',
    'expect_native_timezone',
    'expect_utc_timezone',
    'prefer_native_timezone',
    'prefer_utc_timezone',
    'expect_date',
    'expect_datetime',
    'Entity',
    'NYSE'
    ]


def Timezone(name:str = 'US/Eastern') -> zoneinfo.ZoneInfo:
    """Simple wrapper to convert name to timezone"""
    return pendulum.tz.Timezone(name)


UTC = Timezone('UTC')
GMT = Timezone('GMT')
EST = Timezone('US/Eastern')
LCL = pendulum.tz.Timezone(pendulum.tz.get_local_timezone().name)


class WeekDay(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


day_obj = {
    'MO': WeekDay.MONDAY,
    'TU': WeekDay.TUESDAY,
    'WE': WeekDay.WEDNESDAY,
    'TH': WeekDay.THURSDAY,
    'FR': WeekDay.FRIDAY,
    'SA': WeekDay.SATURDAY,
    'SU': WeekDay.SUNDAY
}


MONTH_SHORTNAME = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
}

DATEMATCH = r'^(?P<d>N|T|Y|P|M)(?P<n>[-+]?\d+)?(?P<b>b?)?$'


# def caller_entity(func):
    # """Helper to get current entity from function"""
    # # general frame args inspect
    # import inspect
    # frame = inspect.currentframe()
    # outer_frames = inspect.getouterframes(frame)
    # caller_frame = outer_frames[1][0]
    # args = inspect.getargvalues(caller_frame)
    # # find our entity
    # param = inspect.signature(func).parameters.get('entity')
    # default = param.default if param else NYSE
    # entity = args.locals['kwargs'].get('entity', default)
    # return entity


def isdateish(x):
    return isinstance(x, datetime.date | datetime.datetime | pd.Timestamp | np.datetime64)


def parse_arg(typ, arg):
    if isdateish(arg):
        if typ == datetime.datetime:
            return DateTime.parse(arg)
        if typ == datetime.date:
            return Date.parse(arg)
    return arg


def parse_args(typ, *args):
    this = []
    for a in args:
        if issequence(a):
            this.append(parse_args(typ, *a))
        else:
            this.append(parse_arg(typ, a))
    return this


def expect(func, typ: type[datetime.date], exclkw: bool = False) -> Callable:
    """Decorator to force input type of date/datetime inputs
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = parse_args(typ, *args)
        if not exclkw:
            for k, v in kwargs.items():
                if isdateish(v):
                    if typ == datetime.datetime:
                        kwargs[k] = DateTime.parse(v)
                        continue
                    if typ == datetime.date:
                        kwargs[k] = Date.parse(v)
        return func(*args, **kwargs)
    return wrapper


expect_date = partial(expect, typ=datetime.date)
expect_datetime = partial(expect, typ=datetime.datetime)
expect_time = partial(expect, typ=datetime.time)


def type_class(typ, obj):
    if typ:
        return typ
    if obj.__class__ in {datetime.datetime, pendulum.DateTime, DateTime}:
        return DateTime
    if obj.__class__ in {datetime.date, pendulum.Date, Date}:
        return Date
    raise ValueError(f'Unknown type {typ}')


def parent_params(cls):
    if cls in {datetime.date, pendulum.Date, Date}:
        return ('year', 'month', 'day')
    if cls in {datetime.datetime, pendulum.DateTime, DateTime}:
        return ('year', 'month', 'day', 'hour', 'minute',
                'second', 'microsecond', 'tzinfo')
    raise ValueError(f'Unknown cls {cls}')


def store_entity(func=None, *, typ=None):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        _entity = self._entity
        d = type_class(typ, self)(func(self, *args, **kwargs))
        d._entity = _entity
        return d
    if func is None:
        return partial(store_entity, typ=typ)
    return wrapper


def store_both(func=None, *, typ=None):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        _entity = self._entity
        _business = self._business
        d = type_class(typ, self)(func(self, *args, **kwargs))
        d._entity = _entity
        d._business = _business
        return d
    if func is None:
        return partial(store_both, typ=typ)
    return wrapper


def prefer_utc_timezone(func, force:bool = False):
    """Return datetime as UTC.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        d = func(*args, **kwargs)
        if not d:
            return
        if not force and d.tzinfo:
            return d
        return d.replace(tzinfo=UTC)
    return wrapper


def prefer_native_timezone(func, force:bool = False):
    """Return datetime as native.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        d = func(*args, **kwargs)
        if not d:
            return
        if not force and d.tzinfo:
            return d
        return d.replace(tzinfo=LCL)
    return wrapper


expect_native_timezone = partial(prefer_native_timezone, force=True)
expect_utc_timezone = partial(prefer_utc_timezone, force=True)


class Entity(ABC):
    """ABC for named entity types"""

    tz = UTC

    @staticmethod
    @abstractmethod
    def business_days(begdate: datetime.date, enddate: datetime.date):
        """Returns all business days over a range"""

    @staticmethod
    @abstractmethod
    def business_hours(begdate: datetime.date, enddate: datetime.date):
        """Returns all business open and close times over a range"""

    @staticmethod
    @abstractmethod
    def business_holidays(begdate: datetime.date, enddate: datetime.date):
        """Returns only holidays over a range"""


class NYSE(Entity):
    """New York Stock Exchange"""

    BEGDATE = datetime.date(1900, 1, 1)
    ENDDATE = datetime.date(2200, 1, 1)
    calendar = mcal.get_calendar('NYSE')

    tz = EST

    @staticmethod
    @lru_cache
    def business_days(begdate=BEGDATE, enddate=ENDDATE) -> set:
        return {Date(d.date()) for d in NYSE.calendar.valid_days(begdate, enddate)}

    @staticmethod
    @lru_cache
    def business_hours(begdate=BEGDATE, enddate=ENDDATE) -> dict:
        df = NYSE.calendar.schedule(begdate, enddate, tz=EST)
        open_close = [(DateTime(o.to_pydatetime()), DateTime(c.to_pydatetime()))
                      for o, c in zip(df.market_open, df.market_close)]
        return dict(zip(df.index.date, open_close))

    @staticmethod
    @lru_cache
    def business_holidays(begdate=BEGDATE, enddate=ENDDATE) -> set:
        return {Date(d.date()) for d in map(pd.to_datetime, NYSE.calendar.holidays().holidays)}


def parse():
    """Generic parser that guesses type
    """
    raise NotImplementedError('Generic parser not implemented, use Date or DateTime parsers')


class PendulumBusinessDateMixin:

    _pendulum = pendulum.Date
    _entity: type[NYSE] = NYSE
    _business: bool = False

    def business(self) -> Self:
        self._business = True
        return self

    @property
    def b(self) -> Self:
        return self.business()

    def entity(self, e: type[NYSE] = NYSE) -> Self:
        self._entity = e
        return self

    @store_entity
    def add(self, years: int = 0, months: int = 0, weeks: int = 0, days: int = 0, **kwargs) -> Self:
        """Add wrapper
        If not business use Pendulum
        If business assume only days (for now) and use local logic
        """
        _business = self._business
        self._business = False
        if _business:
            if days == 0:
                return self._business_or_next()
            if days < 0:
                return self.business().subtract(days=abs(days))
            while days > 0:
                self = self._business_next(days=1)
                days -= 1
            return self
        return self._pendulum.add(self, years, months, weeks, days, **kwargs)

    @store_entity
    def subtract(self, years: int = 0, months: int = 0, weeks: int = 0, days: int = 0, **kwargs) -> Self:
        """Subtract wrapper
        If not business use Pendulum
        If business assume only days (for now) and use local logic
        """
        _business = self._business
        self._business = False
        if _business:
            if days == 0:
                return self._business_or_previous()
            if days < 0:
                return self.business().add(days=abs(days))
            while days > 0:
                self = self._business_previous(days=1)
                days -= 1
            return self
        kwargs = {k: -1*v for k,v in kwargs.items()}
        return self._pendulum.add(self, -years, -months, -weeks, -days, **kwargs)

    @store_entity
    def first_of(self, unit: str, day_of_week: WeekDay | None = None) -> Self:
        """Returns an instance set to the first occurrence
        of a given day of the week in the current unit.
        """
        _business = self._business
        self._business = False
        self = self._pendulum.first_of(self, unit, day_of_week)
        if _business:
            self = self._business_or_next()
        return self

    @store_entity
    def last_of(self, unit: str, day_of_week: WeekDay | None = None) -> Self:
        """Returns an instance set to the last occurrence
        of a given day of the week in the current unit.
        """
        _business = self._business
        self._business = False
        self = self._pendulum.last_of(self, unit, day_of_week)
        if _business:
            self = self._business_or_previous()
        return self

    @store_entity
    def start_of(self, unit: str) -> Self:
        """Returns a copy of the instance with the time reset
        """
        _business = self._business
        self._business = False
        self = self._pendulum.start_of(self, unit)
        if _business:
            self = self._business_or_next()
        return self

    @store_entity
    def end_of(self, unit: str) -> Self:
        """Returns a copy of the instance with the time reset
        """
        _business = self._business
        self._business = False
        self = self._pendulum.end_of(self, unit)
        if _business:
            self = self._business_or_previous()
        return self

    @store_entity
    def previous(self, day_of_week: WeekDay | None = None) -> Self:
        """Modify to the previous occurrence of a given day of the week.
        """
        _business = self._business
        self._business = False
        self = self._pendulum.previous(self, day_of_week)
        if _business:
            self = self._business_or_next()
        return self

    @store_entity
    def next(self, day_of_week: WeekDay | None = None) -> Self:
        """Modify to the next occurrence of a given day of the week.
        """
        _business = self._business
        self._business = False
        self = self._pendulum.next(self, day_of_week)
        if _business:
            self = self._business_or_previous()
        return self

    @expect_date
    def business_open(self) -> bool:
        """Business open

        >>> thedate = Date(2021, 4, 19) # Monday
        >>> thedate.business_open()
        True
        >>> thedate = Date(2021, 4, 17) # Saturday
        >>> thedate.business_open()
        False
        >>> thedate = Date(2021, 1, 18) # MLK Day
        >>> thedate.business_open()
        False
        """
        return self.is_business_day()

    @expect_date
    def is_business_day(self) -> bool:
        """Is business date.

        >>> thedate = Date(2021, 4, 19) # Monday
        >>> thedate.is_business_day()
        True
        >>> thedate = Date(2021, 4, 17) # Saturday
        >>> thedate.is_business_day()
        False
        >>> thedate = Date(2021, 1, 18) # MLK Day
        >>> thedate.is_business_day()
        False
        >>> thedate = Date(2021, 11, 25) # Thanksgiving
        >>> thedate.is_business_day()
        False
        >>> thedate = Date(2021, 11, 26) # Day after ^
        >>> thedate.is_business_day()
        True
        """
        return self in self._entity.business_days()

    @expect_date
    def business_hours(self) -> 'tuple[DateTime, DateTime]':
        """Business hours

        Returns (None, None) if not a business day

        >>> thedate = Date(2023, 1, 5)
        >>> thedate.business_hours()
        (... 9, 30, ... 16, 0, ...)

        >>> thedate = Date(2023, 7, 3)
        >>> thedate.business_hours()
        (... 9, 30, ... 13, 0, ...)

        >>> thedate = Date(2023, 11, 24)
        >>> thedate.business_hours()
        (... 9, 30, ... 13, 0, ...)

        >>> thedate = Date(2024, 5, 27) # memorial day
        >>> thedate.business_hours()
        (None, None)
        """
        return self._entity.business_hours(self, self)\
            .get(self, (None, None))

    @store_both
    def _business_next(self, days=0):
        """Helper for cycling through N business day"""
        days = abs(days)
        while days > 0:
            try:
                self = self._pendulum.add(self, days=1)
            except OverflowError:
                break
            if self.is_business_day():
                days -= 1
        return self

    @store_both
    def _business_previous(self, days=0):
        """Helper for cycling through N business day"""
        days = abs(days)
        while days > 0:
            try:
                self = self._pendulum.add(self, days=-1)
            except OverflowError:
                break
            if self.is_business_day():
                days -= 1
        return self

    @store_entity
    def _business_or_next(self):
        self._business = False
        self = self._pendulum.subtract(self, days=1)
        self = self._business_next(days=1)
        return self

    @store_entity
    def _business_or_previous(self):
        self._business = False
        self = self._pendulum.add(self, days=1)
        self = self._business_previous(days=1)
        return self


def date_to_tuple(obj: pendulum.Date):
    return obj.year, obj.month, obj.day


class Date(PendulumBusinessDateMixin, pendulum.Date):
    """Inherits and wraps pendulum.Date
    """

    _pendulum = pendulum.Date

    def __new__(cls, *args, **kwargs):
        """
        >>> Date(2022, 1, 1)
        Date(2022, 1, 1)
        >>> Date('2022/1/1')
        Date(2022, 1, 1)
        >>> d = Date(datetime.date(2022, 1, 1))
        >>> d
        Date(2022, 1, 1)
        >>> d._business
        False
        >>> d = Date(Date(2022, 1, 1))
        >>> d
        Date(2022, 1, 1)
        >>> d._business
        False
        >>> Date(datetime.date(2022, 1, 1))
        Date(2022, 1, 1)
        >>> Date(pendulum.Date(2022, 1, 1))
        Date(2022, 1, 1)
        >>> Date(Date(2022, 1, 1))
        Date(2022, 1, 1)

        """
        this = None
        if len(args) == 0:
            this = date_to_tuple(pendulum.today().date())
        if args and isinstance(args[0], str):
            return cls.parse(*args, **kwargs)
        if args and isinstance(args[0], bytes):
            this = date_to_tuple(datetime.date(args[0]))
        if len(args) == 1 and isinstance(args[0], datetime.date):
            this = args[0].year, args[0].month, args[0].day
        if len(args) == 1 and args[0] is None:
            this = date_to_tuple(pendulum.today().date())
        if len(args) == 3:
            this = args[0], args[1], args[2]
        if this is not None:
            return super(cls._pendulum, cls).__new__(cls, *this, **kwargs)
        raise ValueError(f'Incompatible arguments for Date: {args}')

    def to_string(self, fmt: str) -> str:
        """Format cleaner https://stackoverflow.com/a/2073189.

        >>> Date(2022, 1, 5).to_string('%-m/%-d/%Y')
        '1/5/2022'
        """
        return self.strftime(fmt.replace('%-', '%#') if os.name == 'nt' else fmt)

    @classmethod
    def parse(
        cls,
        s: str | datetime.date | datetime.datetime | pd.Timestamp | np.datetime64,
        fmt: str = None,
        raise_err: bool = False,
        shortcodes: bool = True
    ) -> Self | None:
        """Convert a string to a date handling many different formats.

        previous business day accessed with 'P'
        >>> Date.parse('P')==Date().b.subtract(days=1)
        True
        >>> Date.parse('T-3b')==Date().b.subtract(days=3)
        True
        >>> Date.parse('T-3b')==Date().b.add(days=-3)
        True
        >>> Date.parse('T+3b')==Date().b.subtract(days=-3)
        True
        >>> Date.parse('T+3b')==Date().b.add(days=3)
        True
        >>> Date.parse('M')==Date().start_of('month').subtract(days=1)
        True

        m[/-]d[/-]yyyy  6-23-2006
        >>> Date.parse('6-23-2006')
        Date(2006, 6, 23)

        m[/-]d[/-]yy    6/23/06
        >>> Date.parse('6/23/06')
        Date(2006, 6, 23)

        m[/-]d          6/23
        >>> Date.parse('6/23') == Date(today().year, 6, 23)
        True

        yyyy-mm-dd      2006-6-23
        >>> Date.parse('2006-6-23')
        Date(2006, 6, 23)

        yyyymmdd        20060623
        >>> Date.parse('20060623')
        Date(2006, 6, 23)

        dd-mon-yyyy     23-JUN-2006
        >>> Date.parse('23-JUN-2006')
        Date(2006, 6, 23)

        mon-dd-yyyy     JUN-23-2006
        >>> Date.parse('20 Jan 2009')
        Date(2009, 1, 20)

        month dd, yyyy  June 23, 2006
        >>> Date.parse('June 23, 2006')
        Date(2006, 6, 23)

        dd-mon-yy
        >>> Date.parse('23-May-12')
        Date(2012, 5, 23)

        ddmonyyyy
        >>> Date.parse('23May2012')
        Date(2012, 5, 23)

        >>> Date.parse('Oct. 24, 2007', fmt='%b. %d, %Y')
        Date(2007, 10, 24)

        >>> Date.parse('Yesterday') == now().subtract(days=1).date()
        True
        >>> Date.parse('TODAY') == today()
        True
        >>> Date.parse('Jan. 13, 2014')
        Date(2014, 1, 13)

        >>> Date.parse('March') == Date(today().year, 3, today().day)
        True

        >>> Date.parse(np.datetime64('2000-01', 'D'))
        Date(2000, 1, 1)

        >>> Date.parse(Date(2000, 1, 1))
        Date(2000, 1, 1)

        only raise error when we explicitly say so
        >>> Date.parse('bad date') is None
        True
        >>> Date.parse('bad date', raise_err=True)
        Traceback (most recent call last):
        ...
        ValueError: Failed to parse date: bad date
        """

        def date_for_symbol(s):
            if s == 'N':
                return today()
            if s == 'T':
                return today()
            if s == 'Y':
                return today().subtract(days=1)
            if s == 'P':
                return cls().business().subtract(days=1)
            if s == 'M':
                return cls().start_of('month').subtract(days=1)

        def year(m):
            try:
                yy = int(m.group('y'))
                if yy < 100:
                    yy += 2000
            except IndexError:
                logger.warning('Using default this year')
                yy = today().year
            return yy

        if is_null(s) or s == '':
            if raise_err:
                raise ValueError('Empty value')
            return

        if s.__class__ == Date:
            return s
        if isinstance(s, np.datetime64 | pd.Timestamp):
            s = DateTime.parse(s)
        if isinstance(s, datetime.datetime):
            if any([s.hour, s.minute, s.second, s.microsecond]):
                logger.debug('Forced datetime with non-zero time to date')
            return cls(s.year, s.month, s.day)
        if isinstance(s, datetime.date):
            return cls(s.year, s.month, s.day)
        if not isinstance(s, str):
            raise TypeError(f'Invalid type for date column: {s.__class__}')

        # always use the format if specified
        if fmt:
            try:
                return cls(*time.strptime(s, fmt)[:3])
            except ValueError:
                logger.debug('Format string passed to strptime failed')

        # handle special symbolic values: T, Y-2, P-1b
        if shortcodes:
            if m := re.match(DATEMATCH, s):
                d = date_for_symbol(m.groupdict().get('d'))
                n = m.groupdict().get('n')
                if not n:
                    return d
                n = int(n)
                b = m.groupdict().get('b')
                if b:
                    assert b == 'b'
                    d = d.business().add(days=n)
                else:
                    d = d.add(days=n)
                return d
            if 'today' in s.lower():
                return today()
            if 'yester' in s.lower():
                return today().subtract(days=1)

        try:
            return cls(parser.parse(s).date())
        except (TypeError, ValueError):
            logger.debug('Date parser failed .. trying our custom parsers')

        # Regex with Month Numbers
        exps = (
            r'^(?P<m>\d{1,2})[/-](?P<d>\d{1,2})[/-](?P<y>\d{4})$',
            r'^(?P<m>\d{1,2})[/-](?P<d>\d{1,2})[/-](?P<y>\d{1,2})$',
            r'^(?P<m>\d{1,2})[/-](?P<d>\d{1,2})$',
            r'^(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})$',
            r'^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})$',
        )
        for exp in exps:
            if m := re.match(exp, s):
                mm = int(m.group('m'))
                dd = int(m.group('d'))
                yy = year(m)
                return cls(yy, mm, dd)

        # Regex with Month Name
        exps = (
            r'^(?P<d>\d{1,2})[- ](?P<m>[A-Za-z]{3,})[- ](?P<y>\d{4})$',
            r'^(?P<m>[A-Za-z]{3,})[- ](?P<d>\d{1,2})[- ](?P<y>\d{4})$',
            r'^(?P<m>[A-Za-z]{3,}) (?P<d>\d{1,2}), (?P<y>\d{4})$',
            r'^(?P<d>\d{2})(?P<m>[A-Z][a-z]{2})(?P<y>\d{4})$',
            r'^(?P<d>\d{1,2})-(?P<m>[A-Z][a-z][a-z])-(?P<y>\d{2})$',
            r'^(?P<d>\d{1,2})-(?P<m>[A-Z]{3})-(?P<y>\d{2})$',
        )
        for exp in exps:
            if m := re.match(exp, s):
                logger.debug('Matched month name')
                try:
                    mm = MONTH_SHORTNAME[m.group('m').lower()[:3]]
                except KeyError:
                    logger.debug('Month name did not match MONTH_SHORTNAME')
                    continue
                dd = int(m.group('d'))
                yy = year(m)
                return cls(yy, mm, dd)

        if raise_err:
            raise ValueError('Failed to parse date: %s', s)

    @classmethod
    def today(cls):
        return cls(pendulum.today().date())

    def to_date(self):
        """Simple drop to datetime.date
        """
        return datetime.date(self.year, self.month, self.day)

    def isoweek(self):
        """Week number 1-52 following ISO week-numbering

        Standard weeks
        >>> Date(2023, 1, 2).isoweek()
        1
        >>> Date(2023, 4, 27).isoweek()
        17
        >>> Date(2023, 12, 31).isoweek()
        52

        Belongs to week of previous year
        >>> Date(2023, 1, 1).isoweek()
        52
        """
        with contextlib.suppress(Exception):
            return self.isocalendar()[1]

    """
    See how pendulum does end_of and next_ with getattr

    Create a nearest [start_of, end_of] [week, day, month, quarter, year]

    combo that accounts for whatever prefix and unit is passed in
    """

    def nearest_start_of_month(self):
        """Get `nearest` start of month

        1/1/2015 -> Thursday (New Year's Day)
        2/1/2015 -> Sunday

        >>> Date(2015, 1, 1).nearest_start_of_month()
        Date(2015, 1, 1)
        >>> Date(2015, 1, 15).nearest_start_of_month()
        Date(2015, 1, 1)
        >>> Date(2015, 1, 15).b.nearest_start_of_month()
        Date(2015, 1, 2)
        >>> Date(2015, 1, 16).nearest_start_of_month()
        Date(2015, 2, 1)
        >>> Date(2015, 1, 31).nearest_start_of_month()
        Date(2015, 2, 1)
        >>> Date(2015, 1, 31).b.nearest_start_of_month()
        Date(2015, 2, 2)
        """
        _business = self._business
        self._business = False
        if self.day > 15:
            d = self.end_of('month')
            if _business:
                return d.business().add(days=1)
            return d.add(days=1)
        d = self.start_of('month')
        if _business:
            return d.business().add(days=1)
        return d

    def nearest_end_of_month(self):
        """Get `nearest` end of month

        12/31/2014 -> Wednesday
        1/31/2015 -> Saturday

        >>> Date(2015, 1, 1).nearest_end_of_month()
        Date(2014, 12, 31)
        >>> Date(2015, 1, 15).nearest_end_of_month()
        Date(2014, 12, 31)
        >>> Date(2015, 1, 15).b.nearest_end_of_month()
        Date(2014, 12, 31)
        >>> Date(2015, 1, 16).nearest_end_of_month()
        Date(2015, 1, 31)
        >>> Date(2015, 1, 31).nearest_end_of_month()
        Date(2015, 1, 31)
        >>> Date(2015, 1, 31).b.nearest_end_of_month()
        Date(2015, 1, 30)
        """
        _business = self._business
        self._business = False
        if self.day <= 15:
            d = self.start_of('month')
            if _business:
                return d.business().subtract(days=1)
            return d.subtract(days=1)
        d = self.end_of('month')
        if _business:
            return d.business().subtract(days=1)
        return d

    def next_relative_date_of_week_by_day(self, day='MO'):
        """Get next relative day of week by relativedelta code

        >>> Date(2020, 5, 18).next_relative_date_of_week_by_day('SU')
        Date(2020, 5, 24)
        >>> Date(2020, 5, 24).next_relative_date_of_week_by_day('SU')
        Date(2020, 5, 24)
        """
        if self.weekday() == day_obj.get(day):
            return self
        return self.next(day_obj.get(day))

    def weekday_or_previous_friday(self):
        """Return the date if it is a weekday, else previous Friday

        >>> Date(2019, 10, 6).weekday_or_previous_friday() # Sunday
        Date(2019, 10, 4)
        >>> Date(2019, 10, 5).weekday_or_previous_friday() # Saturday
        Date(2019, 10, 4)
        >>> Date(2019, 10, 4).weekday_or_previous_friday() # Friday
        Date(2019, 10, 4)
        >>> Date(2019, 10, 3).weekday_or_previous_friday() # Thursday
        Date(2019, 10, 3)
        """
        dnum = self.weekday()
        if dnum in {WeekDay.SATURDAY, WeekDay.SUNDAY}:
            return self.subtract(days=dnum - 4)
        return self

    def lookback(self, unit='last') -> Self:
        """Date back based on lookback string, ie last, week, month.

        >>> Date(2018, 12, 7).b.lookback('last')
        Date(2018, 12, 6)
        >>> Date(2018, 12, 7).b.lookback('day')
        Date(2018, 12, 6)
        >>> Date(2018, 12, 7).b.lookback('week')
        Date(2018, 11, 30)
        >>> Date(2018, 12, 7).b.lookback('month')
        Date(2018, 11, 7)
        """
        def _lookback(years=0, months=0, weeks=0, days=0):
            _business = self._business
            self._business = False
            d = self\
                .subtract(years=years, months=months, weeks=weeks, days=days)
            if _business:
                return d._business_or_previous()
            return d

        return {
            'day': _lookback(days=1),
            'last': _lookback(days=1),
            'week': _lookback(weeks=1),
            'month': _lookback(months=1),
            'quarter': _lookback(months=3),
            'year': _lookback(years=1),
            }.get(unit)

    """
    create a simple nth weekday function that accounts for
    [1,2,3,4] and weekday as options
    or weekday, [1,2,3,4]

    """

    @staticmethod
    def third_wednesday(year, month):
        """Third Wednesday date of a given month/year

        >>> Date.third_wednesday(2022, 6)
        Date(2022, 6, 15)
        >>> Date.third_wednesday(2023, 3)
        Date(2023, 3, 15)
        >>> Date.third_wednesday(2022, 12)
        Date(2022, 12, 21)
        >>> Date.third_wednesday(2023, 6)
        Date(2023, 6, 21)
        """
        third = Date(year, month, 15)  # lowest 3rd day
        w = third.weekday()
        if w != WeekDay.WEDNESDAY:
            third = third.replace(day=(15 + (WeekDay.WEDNESDAY - w) % 7))
        return third


def today():
    """Get current date
    """
    return Date(pendulum.today().date())


class Time(pendulum.Time):

    _pendulum = pendulum.Time

    def __new__(cls, *args, **kwargs):
        """
        >>> Time(12, 30, 1)
        Time(12, 30, 1, tzinfo=Timezone('UTC'))
        >>> Time('12:30:01 AM')
        Time(12, 30, 1, tzinfo=Timezone('UTC'))
        >>> Time(datetime.time(12, 30, 1))
        Time(12, 30, 1, tzinfo=Timezone('UTC'))
        >>> Time(pendulum.Time(12, 30, 1))
        Time(12, 30, 1, tzinfo=Timezone('UTC'))
        >>> Time(Time(12, 30, 1))
        Time(12, 30, 1, tzinfo=Timezone('UTC'))

        """
        this = None
        if args and isinstance(args[0], str):
            return cls.parse(*args, **kwargs)
        if len(args) == 0:
            return DateTime.now().time().replace(tzinfo=LCL)
        if len(args) == 1 and args[0] is None:
            return DateTime.now().time().replace(tzinfo=LCL)
        if len(args) == 1 and isinstance(args[0], datetime.time | datetime.datetime):
            this = time_to_dict(args[0])
        if len(args) >= 2:
            this = time_to_dict(*args)
        if this is not None:
            tzinfo = _get_tzinfo(*args, **kwargs)
            kwargs = {**kwargs, **this, 'tzinfo': tzinfo or UTC}
            this = []
            return super(cls._pendulum, cls).__new__(cls, *this, **kwargs)
        raise ValueError(f'Incompatible arguments for DateTime: {args}')

    @staticmethod
    @prefer_utc_timezone
    def parse(s, fmt=None, raise_err=False):
        """Convert a string to a time handling many formats::

            handle many time formats:
            hh[:.]mm
            hh[:.]mm am/pm
            hh[:.]mm[:.]ss
            hh[:.]mm[:.]ss[.,]uuu am/pm
            hhmmss[.,]uuu
            hhmmss[.,]uuu am/pm

        >>> Time.parse('9:30')
        Time(9, 30, 0, tzinfo=Timezone('UTC'))
        >>> Time.parse('9:30:15')
        Time(9, 30, 15, tzinfo=Timezone('UTC'))
        >>> Time.parse('9:30:15.751')
        Time(9, 30, 15, 751000, tzinfo=Timezone('UTC'))
        >>> Time.parse('9:30 AM')
        Time(9, 30, 0, tzinfo=Timezone('UTC'))
        >>> Time.parse('9:30 pm')
        Time(21, 30, 0, tzinfo=Timezone('UTC'))
        >>> Time.parse('9:30:15.751 PM')
        Time(21, 30, 15, 751000, tzinfo=Timezone('UTC'))
        >>> Time.parse('0930')  # Date treats this as a date, careful!!
        Time(9, 30, 0, tzinfo=Timezone('UTC'))
        >>> Time.parse('093015')
        Time(9, 30, 15, tzinfo=Timezone('UTC'))
        >>> Time.parse('093015,751')
        Time(9, 30, 15, 751000, tzinfo=Timezone('UTC'))
        >>> Time.parse('0930 pm')
        Time(21, 30, 0, tzinfo=Timezone('UTC'))
        >>> Time.parse('093015,751 PM')
        Time(21, 30, 15, 751000, tzinfo=Timezone('UTC'))
        """

        def seconds(m):
            try:
                return int(m.group('s'))
            except Exception:
                return 0

        def micros(m):
            try:
                return int(m.group('u'))
            except Exception:
                return 0

        def is_pm(m):
            try:
                return m.group('ap').lower() == 'pm'
            except Exception:
                return False

        if is_null(s) or s == '':
            if raise_err:
                raise ValueError('Empty value')
            return

        if s.__class__ == Time:
            return s
        if isinstance(s, datetime.datetime):
            return Time(pendulum.instance(s).time())
        if isinstance(s, datetime.time):
            return Time(pendulum.instance(s))

        if fmt:
            return Time(*time.strptime(s, fmt)[3:6])

        exps = (
            r'^(?P<h>\d{1,2})[:.](?P<m>\d{2})([:.](?P<s>\d{2})([.,](?P<u>\d+))?)?( +(?P<ap>[aApP][mM]))?$',
            r'^(?P<h>\d{2})(?P<m>\d{2})((?P<s>\d{2})([.,](?P<u>\d+))?)?( +(?P<ap>[aApP][mM]))?$',
        )

        for exp in exps:
            if m := re.match(exp, s):
                hh = int(m.group('h'))
                mm = int(m.group('m'))
                ss = seconds(m)
                uu = micros(m)
                if is_pm(m) and hh < 12:
                    hh += 12
                return Time(hh, mm, ss, uu * 1000)
        logger.debug('Custom parsers failed, trying pendulum parser')

        try:
            return Time(parser.parse(s).time())
        except (TypeError, ValueError):
            pass

        if raise_err:
            raise ValueError('Failed to parse time: %s', s)

    def to_time(self):
        """Simple drop to datetime.time
        """
        return time(self.hour, self.minute, self.second, self.microsecond, self.tzinfo)


def time_to_dict(*args):
    keys = ('hour', 'minute', 'second', 'microsecond')
    d = {k: 0 for k in keys}
    if isinstance(args[0], datetime.time | datetime.date):
        for k in keys:
            d[k] = getattr(args[0], k, 0)
    else:
        _args = list(reversed(args))
        for k in keys:
            with contextlib.suppress(IndexError):
                d[k] = _args.pop()
    return d


def datetime_to_dict(*args):
    keys = ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond')
    d = {k: 0 for k in keys}
    if isinstance(args[0], datetime.date):
        for k in keys:
            d[k] = getattr(args[0], k, 0)
    else:
        _args = list(reversed(args))
        for k in keys:
            with contextlib.suppress(IndexError):
                d[k] = _args.pop()
    return d


def _get_tzinfo(*args, **kw):
    for arg in args:
        if isinstance(arg, datetime.datetime | datetime.time):
            return arg.tzinfo
        if isinstance(arg, zoneinfo.ZoneInfo):
            return arg
    return kw.get('tzinfo', kw.get('tz'))


class DateTime(PendulumBusinessDateMixin, pendulum.DateTime):
    """Inherits and wraps pendulum.DateTime
    """

    _pendulum = pendulum.DateTime

    def __new__(cls, *args, **kwargs):
        """
        >>> DateTime(2022, 1, 1)
        DateTime(2022, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))
        >>> DateTime('2022/1/1')
        DateTime(2022, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))
        >>> DateTime(2022, 1, 1, 0, 0, 0)
        DateTime(2022, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))
        >>> DateTime(datetime.date(2022, 1, 1))
        DateTime(2022, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))
        >>> DateTime(Date(2022, 1, 1))
        DateTime(2022, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))
        >>> DateTime(Time(4, 4, 21))
        DateTime(..., 4, 4, 21, tzinfo=Timezone('UTC'))
        >>> DateTime(datetime.datetime(2022, 1, 1, 0, 0, 0))
        DateTime(2022, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))
        >>> DateTime(pendulum.DateTime(2022, 1, 1, 0, 0, 0))
        DateTime(2022, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))
        >>> DateTime(DateTime(2022, 1, 1, 0, 0, 0))
        DateTime(2022, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))

        """
        this = None
        if len(args) == 0:
            return cls.now()
        if len(args) == 1 and args[0] is None:
            return cls.now()
        if len(args) and isinstance(args[0], str):
            return DateTime.parse(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], datetime.date):
            this = datetime_to_dict(args[0])
        if len(args) == 1 and isinstance(args[0], datetime.time):
            return cls.combine(Date.today(), args[0], kwargs.get('tzinfo'))
        if len(args) >= 3:
            this = datetime_to_dict(*args)
        if this is not None:
            tzinfo = _get_tzinfo(*args, **kwargs)
            kwargs = {**kwargs, **this, 'tzinfo': tzinfo or LCL}
            this = []
            return super(cls._pendulum, cls).__new__(cls, *this, **kwargs)
        raise ValueError(f'Incompatible arguments for DateTime: {args}')

    def epoch(self):
        """Translate a datetime object into unix seconds since epoch
        """
        return self.timestamp()

    @classmethod
    def now(cls, tz=LCL):
        return cls(pendulum.now(tz))

    @classmethod
    def today(cls):
        """Unlike Pendulum, returns DateTime object at start of day
        """
        return DateTime.now().start_of('day')

    def date(self):
        return Date(self.year, self.month, self.day)

    @classmethod
    def combine(
        cls,
        date: datetime.date,
        time: datetime.time,
        tzinfo: zoneinfo.ZoneInfo | None = None,
    ) -> Self:
        """Combine date and time (*behaves differently from Pendulum `combine`*).
        """
        _tzinfo = tzinfo or time.tzinfo
        return DateTime(datetime.datetime.combine(date, time, tzinfo=_tzinfo))

    def to_datetime(self):
        """Simple drop to datetime.datetime
        """
        return datetime.datetime(self.year, self.month, self.day, self.hour,
                                 self.minute, self.second, self.microsecond, self.tzinfo)

    def rfc3339(self):
        """
        >>> DateTime.parse('Fri, 31 Oct 2014 10:55:00').rfc3339()
        '2014-10-31T10:55:00+00:00'
        """
        return self.isoformat()

    def time(self):
        """Extract time from self (preserve timezone)

        >>> d = DateTime(2022, 1, 1, 12, 30, 15, tzinfo=EST)
        >>> d.time()
        Time(12, 30, 15, tzinfo=Timezone('US/Eastern'))

        >>> d = DateTime(2022, 1, 1, 12, 30, 15, tzinfo=UTC)
        >>> d.time()
        Time(12, 30, 15, tzinfo=Timezone('UTC'))
        """
        return Time(self, tzinfo=self.tzinfo)

    @classmethod
    def parse(
        cls,
        s: str | datetime.date | datetime.datetime | pd.Timestamp | np.datetime64,
        raise_err=False,
    ) -> Self | None:
        """Thin layer on Date parser and our custom `Date.parse` and `to_time`

        Assume UTC, convert to EST
        >>> this_est1 = DateTime.parse('Fri, 31 Oct 2014 18:55:00').in_timezone(EST)
        >>> this_est1
        DateTime(2014, 10, 31, 14, 55, 0, tzinfo=Timezone('US/Eastern'))

        This is actually 18:55 UTC with -4 hours applied = EST
        >>> this_est2 = DateTime.parse('Fri, 31 Oct 2014 14:55:00 -0400')
        >>> this_est2
        DateTime(2014, 10, 31, 14, 55, 0, tzinfo=...)

        UTC time technically equals GMT
        >>> this_utc = DateTime.parse('Fri, 31 Oct 2014 18:55:00 GMT')
        >>> this_utc
        DateTime(2014, 10, 31, 18, 55, 0, tzinfo=Timezone('UTC'))

        We can freely compare time zones
        >>> this_est1==this_est2==this_utc
        True

        Convert date to datetime (will use native time zone)
        >>> DateTime.parse(datetime.date(2000, 1, 1))
        DateTime(2000, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))

        >>> DateTime.parse(DateTime(2000, 1, 1))
        DateTime(2000, 1, 1, 0, 0, 0, tzinfo=Timezone('...'))

        Format tests
        >>> DateTime.parse(1707856982).replace(tzinfo=UTC).epoch()
        1707856982.0
        >>> DateTime.parse('Jan 29  2010')
        DateTime(2010, 1, 29, 0, 0, 0, tzinfo=Timezone('UTC'))
        >>> DateTime.parse(np.datetime64('2000-01', 'D'))
        DateTime(2000, 1, 1, 0, 0, 0, tzinfo=Timezone('UTC'))
        >>> _ = DateTime.parse('Sep 27 17:11')
        >>> _.month, _.day, _.hour, _.minute
        (9, 27, 17, 11)
        """
        if is_null(s) or s == '':
            if raise_err:
                raise ValueError('Empty value')
            return

        if s.__class__ == DateTime:
            return s
        if isinstance(s, pd.Timestamp):
            return cls(pendulum.instance(s.to_pydatetime()))
        if isinstance(s, np.datetime64):
            dtm = np.datetime64(s, 'us').astype(datetime.datetime)
            return cls(pendulum.instance(dtm))
        if isinstance(s, int | float):
            iso = datetime.datetime.fromtimestamp(s).isoformat()
            return cls.parse(iso).replace(tzinfo=LCL)
        if isinstance(s, datetime.datetime):
            return cls(s.year, s.month, s.day, s.hour, s.minute, s.second,
                       s.microsecond, s.tzinfo)
        if isinstance(s, datetime.date):
            logger.debug('Forced date without time to datetime')
            return cls(s.year, s.month, s.day, tzinfo=LCL)
        if not isinstance(s, str):
            raise TypeError(f'Invalid type for date column: {s.__class__}')

        try:
            return cls(pendulum.instance(parser.parse(s)))
        except (TypeError, ValueError) as err:
            logger.debug('Date parser failed .. trying our custom parsers')

        for delim in (' ', ':'):
            bits = s.split(delim, 1)
            if len(bits) == 2:
                d = Date.parse(bits[0])
                t = Time.parse(bits[1])
                if d is not None and t is not None:
                    return DateTime.combine(d, t, LCL)

        d = Date.parse(s)
        if d is not None:
            return cls(d.year, d.month, d.day, 0, 0, 0)

        current = today()
        t = Time.parse(s)
        if t is not None:
            return cls.combine(current, t, LCL)

        if raise_err:
            raise ValueError('Invalid date-time format: ' + s)


def now(tz=LCL):
    """Get current datetime
    """
    return DateTime(pendulum.now(tz))


class IntervalError(AttributeError):
    pass


class Interval:

    _business: bool = False
    _entity: type[NYSE] = NYSE

    def __init__(self, begdate: Date = None, enddate: Date = None):
        self.begdate = Date.parse(begdate)
        self.enddate = Date.parse(enddate)

    def business(self) -> Self:
        self._business = True
        if self.begdate:
            self.begdate.business()
        if self.enddate:
            self.enddate.business()
        return self

    @property
    def b(self) -> Self:
        return self.business()

    def entity(self, e: type[NYSE] = NYSE) -> Self:
        self._entity = e
        if self.begdate:
            self.enddate._entity = e
        if self.enddate:
            self.enddate._entity = e
        return self

    def range(self, window=0) -> tuple[datetime.date, datetime.date]:
        """Set date ranges based on begdate, enddate and window.

        The combinations are as follows:

          beg end num    action
          --- --- ---    ---------------------
           -   -   -     Error, underspecified
          set set set    Error, overspecified
          set set  -
          set  -   -     end=max date
           -  set  -     beg=min date
           -   -  set    end=max date, beg=end - num
          set  -  set    end=beg + num
           -  set set    beg=end - num

        >>> Interval('4/3/2014', None).b.range(3)
        (Date(2014, 4, 3), Date(2014, 4, 8))
        >>> Interval(None, Date(2014, 7, 27)).range(20)
        (Date(2014, 7, 7), Date(2014, 7, 27))
        >>> Interval(None, '2014/7/27').b.range(20)
        (Date(2014, 6, 27), Date(2014, 7, 27))
        """
        begdate, enddate = self.begdate, self.enddate

        window = abs(int(window or 0))

        if begdate and enddate and window:
            raise IntervalError('Window requested and begdate and enddate provided')
        if not begdate and not enddate and not window:
            raise IntervalError('Missing begdate, enddate, and window')
        if not begdate and not enddate and window:
            raise IntervalError('Missing begdate and enddate, window specified')

        if begdate and enddate:
            return (begdate.business() if begdate._business else
                    begdate).add(days=window), \
                   (enddate.business() if enddate._business else
                    enddate).subtract(days=0)

        if (not begdate and not enddate) or enddate:
            begdate = (enddate.business() if enddate._business else
                       enddate).subtract(days=window)
        else:
            enddate = (begdate.business() if begdate._business else
                       begdate).add(days=window)

        return begdate, enddate

    def is_business_day_series(self) -> list[bool]:
        """Is business date range.

        >>> list(Interval(Date(2018, 11, 19), Date(2018, 11, 25)).is_business_day_series())
        [True, True, True, False, True, False, False]
        >>> list(Interval(Date(2021, 11, 22),Date(2021, 11, 28)).is_business_day_series())
        [True, True, True, False, True, False, False]
        """
        for thedate in self.series():
            yield thedate.is_business_day()

    def series(self, window=0):
        """Get a series of datetime.date objects.

        give the function since and until wherever possible (more explicit)
        else pass in a window to back out since or until
        - Window gives window=N additional days. So `until`-`window`=1
        defaults to include ALL days (not just business days)

        >>> next(Interval(Date(2014,7,16), Date(2014,7,16)).series())
        Date(2014, 7, 16)
        >>> next(Interval(Date(2014,7,12), Date(2014,7,16)).series())
        Date(2014, 7, 12)
        >>> len(list(Interval(Date(2014,7,12), Date(2014,7,16)).series()))
        5
        >>> len(list(Interval(Date(2014,7,12), None).series(window=4)))
        5
        >>> len(list(Interval(Date(2014,7,16)).series(window=4)))
        5

        Weekend and a holiday
        >>> len(list(Interval(Date(2014,7,3), Date(2014,7,5)).b.series()))
        1
        >>> len(list(Interval(Date(2014,7,17), Date(2014,7,16)).series()))
        Traceback (most recent call last):
        ...
        AssertionError: Begdate must be earlier or equal to Enddate

        since != business day and want business days
        1/[3,10]/2015 is a Saturday, 1/7/2015 is a Wednesday
        >>> len(list(Interval(Date(2015,1,3), Date(2015,1,7)).b.series()))
        3
        >>> len(list(Interval(Date(2015,1,3), None).b.series(window=3)))
        3
        >>> len(list(Interval(Date(2015,1,3), Date(2015,1,10)).b.series()))
        5
        >>> len(list(Interval(Date(2015,1,3), None).b.series(window=5)))
        5
        """
        window = abs(int(window))
        since, until = self.begdate, self.enddate
        _business = self._business
        assert until or since, 'Since or until is required'
        if not since and until:
            since = (until.business() if _business else
                     until).subtract(days=window)
        elif since and not until:
            until = (since.business() if _business else
                     since).add(days=window)
        assert since <= until, 'Since date must be earlier or equal to Until date'
        thedate = since
        while thedate <= until:
            if _business:
                if thedate.is_business_day():
                    yield thedate
            else:
                yield thedate
            thedate = thedate.add(days=1)

    def end_of_series(self, unit='month') -> list[Date]:
        """Return a series between and inclusive of begdate and enddate.

        >>> Interval(Date(2018, 1, 5), Date(2018, 4, 5)).end_of_series('month')
        [Date(2018, 1, 31), Date(2018, 2, 28), Date(2018, 3, 31), Date(2018, 4, 30)]
        >>> Interval(Date(2018, 1, 5), Date(2018, 4, 5)).end_of_series('week')
        [Date(2018, 1, 7), Date(2018, 1, 14), ..., Date(2018, 4, 8)]
        """
        begdate = self.begdate.end_of(unit)
        enddate = self.enddate.end_of(unit)
        interval = pendulum.interval(begdate, enddate)
        return [Date(d) for d in interval.range(f'{unit}s')]

    def days(self) -> int:
        """Return days between (begdate, enddate] or negative (enddate, begdate].

        >>> Interval(Date.parse('2018/9/6'), Date.parse('2018/9/10')).days()
        4
        >>> Interval(Date.parse('2018/9/10'), Date.parse('2018/9/6')).days()
        -4
        >>> Interval(Date.parse('2018/9/6'), Date.parse('2018/9/10')).b.days()
        2
        >>> Interval(Date.parse('2018/9/10'), Date.parse('2018/9/6')).b.days()
        -2
        """
        assert self.begdate
        assert self.enddate
        if self.begdate == self.enddate:
            return 0
        if self.begdate < self.enddate:
            return len(list(self.series())) - 1
        _reverse = Interval(self.enddate, self.begdate)
        _reverse._entity = self._entity
        _reverse._business = self._business
        return -len(list(_reverse.series())) + 1

    def quarters(self):
        """Return the number of quarters between two dates
        TODO: good enough implementation; refine rules to be heuristically precise

        >>> round(Interval(Date(2020, 1, 1), Date(2020, 2, 16)).quarters(), 2)
        0.5
        >>> round(Interval(Date(2020, 1, 1), Date(2020, 4, 1)).quarters(), 2)
        1.0
        >>> round(Interval(Date(2020, 1, 1), Date(2020, 7, 1)).quarters(), 2)
        1.99
        >>> round(Interval(Date(2020, 1, 1), Date(2020, 8, 1)).quarters(), 2)
        2.33
        """
        return 4 * self.days() / 365.0

    def years(self, basis: int = 0):
        """Years with Fractions (matches Excel YEARFRAC)

        Adapted from https://web.archive.org/web/20200915094905/https://dwheeler.com/yearfrac/calc_yearfrac.py

        Basis:
        0 = US (NASD) 30/360
        1 = Actual/actual
        2 = Actual/360
        3 = Actual/365
        4 = European 30/360

        >>> begdate = Date(1978, 2, 28)
        >>> enddate = Date(2020, 5, 17)

        Tested Against Excel
        >>> "{:.4f}".format(Interval(begdate, enddate).years(0))
        '42.2139'
        >>> '{:.4f}'.format(Interval(begdate, enddate).years(1))
        '42.2142'
        >>> '{:.4f}'.format(Interval(begdate, enddate).years(2))
        '42.8306'
        >>> '{:.4f}'.format(Interval(begdate, enddate).years(3))
        '42.2438'
        >>> '{:.4f}'.format(Interval(begdate, enddate).years(4))
        '42.2194'
        >>> '{:.4f}'.format(Interval(enddate, begdate).years(4))
        '-42.2194'

        Excel has a known leap year bug when year == 1900 (=YEARFRAC("1900-1-1", "1900-12-1", 1) -> 0.9178)
        The bug originated from Lotus 1-2-3, and was purposely implemented in Excel for the purpose of backward compatibility.
        >>> begdate = Date(1900, 1, 1)
        >>> enddate = Date(1900, 12, 1)
        >>> '{:.4f}'.format(Interval(begdate, enddate).years(4))
        '0.9167'
        """

        def average_year_length(date1, date2):
            """Algorithm for average year length"""
            days = (Date(date2.year + 1, 1, 1) - Date(date1.year, 1, 1)).days
            years = (date2.year - date1.year) + 1
            return days / years

        def feb29_between(date1, date2):
            """Requires date2.year = (date1.year + 1) or date2.year = date1.year.

            Returns True if "Feb 29" is between the two dates (date1 may be Feb29).
            Two possibilities: date1.year is a leap year, and date1 <= Feb 29 y1,
            or date2.year is a leap year, and date2 > Feb 29 y2.
            """
            mar1_date1_year = Date(date1.year, 3, 1)
            if calendar.isleap(date1.year) and (date1 < mar1_date1_year) and (date2 >= mar1_date1_year):
                return True
            mar1_date2_year = Date(date2.year, 3, 1)
            return bool(calendar.isleap(date2.year) and date2 >= mar1_date2_year and date1 < mar1_date2_year)

        def appears_lte_one_year(date1, date2):
            """Returns True if date1 and date2 "appear" to be 1 year or less apart.

            This compares the values of year, month, and day directly to each other.
            Requires date1 <= date2; returns boolean. Used by basis 1.
            """
            if date1.year == date2.year:
                return True
            return bool(date1.year + 1 == date2.year and (date1.month > date2.month or date1.month == date2.month and date1.day >= date2.day))

        def basis0(date1, date2):
            # change day-of-month for purposes of calculation.
            date1day, date1month, date1year = date1.day, date1.month, date1.year
            date2day, date2month, date2year = date2.day, date2.month, date2.year
            if date1day == 31 and date2day == 31:
                date1day = 30
                date2day = 30
            elif date1day == 31:
                date1day = 30
            elif date1day == 30 and date2day == 31:
                date2day = 30
            # Note: If date2day==31, it STAYS 31 if date1day < 30.
            # Special fixes for February:
            elif date1month == 2 and date2month == 2 and date1 == date1.end_of('month') \
                and date2 == date2.end_of('month'):
                date1day = 30  # Set the day values to be equal
                date2day = 30
            elif date1month == 2 and date1 == date1.end_of('month'):
                date1day = 30  # "Illegal" Feb 30 date.
            daydiff360 = (date2day + date2month * 30 + date2year * 360) \
                - (date1day + date1month * 30 + date1year * 360)
            return daydiff360 / 360

        def basis1(date1, date2):
            if appears_lte_one_year(date1, date2):
                if date1.year == date2.year and calendar.isleap(date1.year):
                    year_length = 366.0
                elif feb29_between(date1, date2) or (date2.month == 2 and date2.day == 29):
                    year_length = 366.0
                else:
                    year_length = 365.0
                return (date2 - date1).days / year_length
            return (date2 - date1).days / average_year_length(date1, date2)

        def basis2(date1, date2):
            return (date2 - date1).days / 360.0

        def basis3(date1, date2):
            return (date2 - date1).days / 365.0

        def basis4(date1, date2):
            # change day-of-month for purposes of calculation.
            date1day, date1month, date1year = date1.day, date1.month, date1.year
            date2day, date2month, date2year = date2.day, date2.month, date2.year
            if date1day == 31:
                date1day = 30
            if date2day == 31:
                date2day = 30
            # Remarkably, do NOT change Feb. 28 or 29 at ALL.
            daydiff360 = (date2day + date2month * 30 + date2year * 360) - \
                (date1day + date1month * 30 + date1year * 360)
            return daydiff360 / 360

        begdate, enddate = self.begdate, self.enddate
        if enddate is None:
            return

        sign = 1
        if begdate > enddate:
            begdate, enddate = enddate, begdate
            sign = -1
        if begdate == enddate:
            return 0.0

        if basis == 0:
            return basis0(begdate, enddate) * sign
        if basis == 1:
            return basis1(begdate, enddate) * sign
        if basis == 2:
            return basis2(begdate, enddate) * sign
        if basis == 3:
            return basis3(begdate, enddate) * sign
        if basis == 4:
            return basis4(begdate, enddate) * sign

        raise ValueError('Basis range [0, 4]. Unknown basis {basis}.')


Range = namedtuple('Range', ['start', 'end'])


def overlap_days(range_one, range_two, days=False):
    """Test by how much two date ranges overlap
    if `days=True`, we return an actual day count,
    otherwise we just return if it overlaps True/False
    poached from Raymond Hettinger http://stackoverflow.com/a/9044111

    >>> date1 = Date(2016, 3, 1)
    >>> date2 = Date(2016, 3, 2)
    >>> date3 = Date(2016, 3, 29)
    >>> date4 = Date(2016, 3, 30)

    >>> assert overlap_days((date1, date3), (date2, date4))
    >>> assert overlap_days((date2, date4), (date1, date3))
    >>> assert not overlap_days((date1, date2), (date3, date4))

    >>> assert overlap_days((date1, date4), (date1, date4))
    >>> assert overlap_days((date1, date4), (date2, date3))
    >>> overlap_days((date1, date4), (date1, date4), True)
    30

    >>> assert overlap_days((date2, date3), (date1, date4))
    >>> overlap_days((date2, date3), (date1, date4), True)
    28

    >>> assert not overlap_days((date3, date4), (date1, date2))
    >>> overlap_days((date3, date4), (date1, date2), True)
    -26
    """
    r1 = Range(*range_one)
    r2 = Range(*range_two)
    latest_start = max(r1.start, r2.start)
    earliest_end = min(r1.end, r2.end)
    overlap = (earliest_end - latest_start).days + 1
    if days:
        return overlap
    return overlap >= 0


def create_ics(begdate, enddate, summary, location):
    """Create a simple .ics file per RFC 5545 guidelines."""

    return f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//hacksw/handcal//NONSGML v1.0//EN
BEGIN:VEVENT
DTSTART;TZID=America/New_York:{begdate:%Y%m%dT%H%M%S}
DTEND;TZID=America/New_York:{enddate:%Y%m%dT%H%M%S}
SUMMARY:{summary}
LOCATION:{location}
END:VEVENT
END:VCALENDAR
    """


# apply any missing Date functions
for func in ('closest', 'farthest', 'nth_of', 'average', 'fromtimestamp',
             'fromordinal', 'replace'):
    setattr(Date, func, store_entity(getattr(pendulum.Date, func), typ=Date))

# apply any missing DateTime functions
for func in ('astimezone', 'fromordinal', 'in_timezone', 'in_tz', 'instance',
             'replace', 'strptime', 'utcfromtimestamp', 'fromtimestamp', 'utcnow'):
    setattr(DateTime, func, store_entity(getattr(pendulum.DateTime, func), typ=DateTime))


if __name__ == '__main__':
    __import__('doctest').testmod(optionflags=4 | 8 | 32)
