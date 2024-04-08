import pytest
from asserts import assert_raises

from date import Interval, IntervalError


def test_daterange_begdate_enddate_window():
    with assert_raises(IntervalError):
        Interval('4/3/2014', '4/3/2014').range(5)


def test_daterange_all_none():
    with assert_raises(IntervalError):
        Interval(None, None).range(0)


def test_daterange_only_window():
    with assert_raises(IntervalError):
        Interval(None, None).range(5)


if __name__ == '__main__':
    pytest.main([__file__])
