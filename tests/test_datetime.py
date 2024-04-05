import copy
import pickle

import pendulum
import pytest
from asserts import assert_equal, assert_false, assert_not_equal, assert_true
from date import Date, DateTime, Time
from pendulum.tz import Timezone


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


def test_is_business_day():
    """Testing that `business day` function (designed for Date)
    works for DateTime object
    """

    d = DateTime(2000, 1, 1, 12, 30)
    assert_false(d.is_business_day())
    assert_true(d.add(days=2).is_business_day())
    assert_true(d.subtract(days=2).b.add(days=1).is_business_day())


def test_combine():

    date = Date(2000, 1, 1)
    time = Time.parse('9:30 AM')
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


if __name__ == '__main__':
    pytest.main([__file__])
