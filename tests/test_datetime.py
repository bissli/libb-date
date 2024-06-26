import copy
import pickle
from unittest import mock

import pendulum
import pytest
from asserts import assert_equal, assert_not_equal
from pendulum.tz import Timezone

from date import NYSE, Date, DateTime, Time, expect_datetime, now


def test_datetime_new_none_or_empty():
    T = DateTime().replace(second=0, microsecond=0)
    assert_equal(T, DateTime(None).replace(second=0, microsecond=0))
    assert_equal(type(T), DateTime)


def test_datetime_constructor():
    """None or empty constructor returns current Time
    """

    T = DateTime(None).replace(second=0, microsecond=0)
    assert_equal(T, pendulum.now().replace(second=0, microsecond=0))
    assert_equal(type(T), DateTime)

    T = DateTime().replace(second=0, microsecond=0)
    assert_equal(T, pendulum.now().replace(second=0, microsecond=0))
    assert_equal(type(T), DateTime)


def test_add():
    """Testing that add function preserves DateTime object
    """
    d = DateTime(2000, 1, 1, 12, 30)
    assert_equal(d.add(days=1), DateTime(2000, 1, 2, 12, 30))
    assert_not_equal(d.add(days=1), DateTime(2000, 1, 2, 12, 31))

    d = DateTime(2000, 1, 1, 12, 30)
    assert_equal(d.b.add(days=1), DateTime(2000, 1, 3, 12, 30))
    assert_not_equal(d.b.add(days=1), DateTime(2000, 1, 3, 12, 31))

    d = DateTime(2000, 1, 1, 12, 30)
    assert_equal(d.add(days=1, hours=1, minutes=1), DateTime(2000, 1, 2, 13, 31))


def test_subtract():
    """Testing that subtract function preserves DateTime object
    """
    d = DateTime(2000, 1, 4, 12, 30)
    assert_equal(d.subtract(days=1), DateTime(2000, 1, 3, 12, 30))
    assert_not_equal(d.subtract(days=1), DateTime(2000, 1, 3, 12, 31))

    d = DateTime(2000, 1, 4, 12, 30)
    assert_equal(d.b.subtract(days=1), DateTime(2000, 1, 3, 12, 30))
    assert_not_equal(d.b.subtract(days=1), DateTime(2000, 1, 3, 12, 31))

    d = DateTime(2000, 1, 4, 12, 30)
    assert_equal(d.subtract(days=1, hours=1, minutes=1), DateTime(2000, 1, 3, 11, 29))


def test_combine():
    """When combining, ignore default Time parse to UTC"""

    date = Date(2000, 1, 1)
    time = Time.parse('9:30 AM')  # default UTC

    d = DateTime.combine(date, time)
    assert isinstance(d, DateTime)
    assert d._business is False
    assert_equal(d, DateTime(2000, 1, 1, 9, 30, 0, tzinfo=Timezone('UTC')))


def test_copy():

    d = pendulum.DateTime(2022, 1, 1, 12, 30)
    assert_equal(copy.copy(d), d)

    d = DateTime(2022, 1, 1, 12, 30)
    assert_equal(copy.copy(d), d)


def test_deepcopy():

    d = pendulum.DateTime(2022, 1, 1, 12, 30)
    assert_equal(copy.deepcopy(d), d)

    d = DateTime(2022, 1, 1, 12, 30)
    assert_equal(copy.deepcopy(d), d)


def test_pickle():

    d = DateTime(2022, 1, 1, 12, 30)

    with open('datetime.pkl', 'wb') as f:
        pickle.dump(d, f)
    with open('datetime.pkl', 'rb') as f:
        d_ = pickle.load(f)

    assert_equal(d, d_)


def test_now():
    """Managed to create a terrible bug where now returned today()
    """
    assert_not_equal(now(), pendulum.today())
    DateTime.now()  # basic check


@mock.patch('date.DateTime.now')
def test_today(mock):
    mock.return_value = DateTime(2020, 1, 1, 12, 30)
    D = DateTime.today()
    assert_equal(D, DateTime(2020, 1, 1, 0, 0))


def test_type():
    """Checking that returned object is of type DateTime,
    not pendulum.DateTime
    """
    d = DateTime.now()
    assert_equal(type(d), DateTime)

    d = DateTime(entity=NYSE, tzinfo=NYSE.tz)
    assert_equal(type(d), DateTime)


def test_expects():

    @expect_datetime
    def func(args):
        return args

    p = pendulum.DateTime(2022, 1, 1)
    d = DateTime(2022, 1, 1)

    assert_equal(func(p), d)
    assert_equal(func((p, p)), [d, d])
    assert_equal(func(((p, p), p)), [[d, d], d])


if __name__ == '__main__':
    pytest.main([__file__])
