import pytest
from asserts import assert_equal, assert_false, assert_not_equal, assert_true
from libb_date.date import DateTime


def test_add():
    """Testing that add function preserves DateTime object
    """
    d = DateTime(2000, 1, 1, 12, 30)
    assert_equal(d.add(days=1), DateTime(2000, 1, 2, 12, 30))
    assert_not_equal(d.add(days=1), DateTime(2000, 1, 2, 12, 31))

    d = DateTime(2000, 1, 1, 12, 30)
    assert_equal(d.business().add(days=1), DateTime(2000, 1, 3, 12, 30))
    assert_not_equal(d.business().add(days=1), DateTime(2000, 1, 3, 12, 31))


def test_subtract():
    """Testing that subtract function preserves DateTime object
    """
    d = DateTime(2000, 1, 4, 12, 30)
    assert_equal(d.subtract(days=1), DateTime(2000, 1, 3, 12, 30))
    assert_not_equal(d.subtract(days=1), DateTime(2000, 1, 3, 12, 31))

    d = DateTime(2000, 1, 4, 12, 30)
    assert_equal(d.business().subtract(days=1), DateTime(2000, 1, 3, 12, 30))
    assert_not_equal(d.business().subtract(days=1), DateTime(2000, 1, 3, 12, 31))


def test_is_business_day():
    """Testing that `business day` function (designed for Date)
    works for DateTime object
    """

    d = DateTime(2000, 1, 1, 12, 30)
    assert_false(d.is_business_day())
    assert_true(d.add(days=2).is_business_day())
    assert_true(d.subtract(days=2).business().add(days=1).is_business_day())


if __name__ == '__main__':
    pytest.main([__file__])
