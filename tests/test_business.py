import datetime

from asserts import assert_equal, assert_false, assert_true

from date import Date, DateTime


def test_date_business_date_or_next():
    # 9/1 is Saturday, 9/3 is Labor Day
    d = Date(datetime.date(2018, 9, 1))\
        .business()\
        .add(days=0)
    assert_equal(d, Date(2018, 9, 4))

    # regular Saturday
    d = Date(datetime.date(2024, 3, 30))\
        .business()\
        .add(days=0)
    assert_equal(d, Date(2024, 4, 1))

    d = Date(datetime.date(2024, 3, 30))\
        .subtract(days=1)\
        .business()\
        .add(days=1)
    assert_equal(d, Date(2024, 4, 1))

    # regular Sunday
    d = Date(datetime.date(2024, 3, 31))\
        .business()\
        .add(days=0)
    assert_equal(d, Date(2024, 4, 1))

    d = Date(datetime.date(2024, 3, 31))\
        .subtract(days=1)\
        .business()\
        .add(days=1)
    assert_equal(d, Date(2024, 4, 1))

    # regular Monday
    d = Date(datetime.date(2024, 4, 1))\
        .business()\
        .add(days=0)
    assert_equal(d, Date(2024, 4, 1))


def test_date_business_date_or_previous():

    # 3/29 is Good Friday

    # regular Saturday
    d = Date(datetime.date(2024, 3, 30))\
        .business()\
        .subtract(days=0)
    assert_equal(d, Date(2024, 3, 28))

    d = Date(datetime.date(2024, 3, 30))\
        .add(days=1)\
        .business()\
        .subtract(days=1)
    assert_equal(d, Date(2024, 3, 28))

    # regular Sunday
    d = Date(datetime.date(2024, 3, 31))\
        .business()\
        .subtract(days=0)
    assert_equal(d, Date(2024, 3, 28))

    d = Date(datetime.date(2024, 3, 31))\
        .add(days=1)\
        .business()\
        .subtract(days=1)
    assert_equal(d, Date(2024, 3, 28))

    # regular Monday
    d = Date(datetime.date(2024, 4, 1))\
        .business()\
        .subtract(days=0)
    assert_equal(d, Date(2024, 4, 1))


def test_datetime_is_business_day():
    """Testing that `business day` function (designed for Date)
    works for DateTime object
    """

    d = DateTime(2000, 1, 1, 12, 30)
    assert_false(d.is_business_day())
    assert_true(d.add(days=2).is_business_day())
    assert_true(d.subtract(days=2).b.add(days=1).is_business_day())


if __name__ == '__main__':
    __import__('pytest').main([__file__])
