import copy
import datetime
import pickle

import pendulum
import pytest
from asserts import assert_equal
from libb_date.date import NYSE, Date, WeekDay


def test_end_of_week():
    """Get the end date of the week."""

    # Regular Monday
    d = Date(datetime.date(2023, 4, 24)).end_of('week')
    assert_equal(d, Date(2023, 4, 30))

    # Regular Sunday
    d = Date(datetime.date(2023, 4, 30)).end_of('week')
    assert_equal(d, Date(2023, 4, 30))

    # Good Friday
    d = Date(datetime.date(2020, 4, 12)).end_of('week')
    assert_equal(d, Date(2020, 4, 12))
    d = Date(datetime.date(2020, 4, 10)).end_of('week')
    assert_equal(d, Date(2020, 4, 12))

    d = Date(datetime.date(2020, 4, 10))\
        .business()\
        .end_of('week')
    assert_equal(d, Date(2020, 4, 9))

    d = Date(datetime.date(2020, 4, 9))\
        .end_of('week')\
        .business()\
        .subtract(days=1)
    assert_equal(d, Date(2020, 4, 9))


def test_business_date_or_next():
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


def test_business_date_or_previous():

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


def test_next_friday():
    """Get next end of week (Friday)."""

    d = Date(datetime.datetime(2018, 10, 8, 0, 0, 0)).next(WeekDay.FRIDAY)
    assert_equal(d, Date(2018, 10, 12))

    d = Date(datetime.date(2018, 10, 12)).next(WeekDay.FRIDAY)
    assert_equal(d, Date(2018, 10, 19))


def test_start_of_week():
    """Start of week function (Monday unless not a holiday)."""

    # Regular Monday
    d = Date(datetime.date(2023, 4, 24)).start_of('week')
    assert_equal(d, Date(2023, 4, 24))

    # Regular Sunday
    d = Date(datetime.date(2023, 4, 30)).start_of('week')
    assert_equal(d, Date(2023, 4, 24))

    # Memorial day 5/25
    d = Date(datetime.date(2020, 5, 25)).start_of('week')
    assert_equal(d, Date(2020, 5, 25))
    d = Date(datetime.date(2020, 5, 27)).start_of('week')
    assert_equal(d, Date(2020, 5, 25))
    d = Date(datetime.date(2020, 5, 26))\
        .business()\
        .start_of('week')
    assert_equal(d, Date(2020, 5, 26))


def test_end_of_month():
    """End of month"""

    d = Date(datetime.date(2021, 6, 15)).end_of('month')
    assert_equal(d, Date(2021, 6, 30))
    d = Date(datetime.date(2021, 6, 30)).end_of('month')
    assert_equal(d, Date(2021, 6, 30))

    # Sunday -> Friday
    d =  Date(datetime.date(2023, 4, 30))\
        .business()\
        .end_of('month')
    assert_equal(d, Date(2023, 4, 28))


def test_previous_end_of_month():
    """Previous EOM"""

    d = Date(datetime.date(2021, 5, 30))\
        .start_of('month')\
        .subtract(days=1)
    assert_equal(d, Date(2021, 4, 30))


def test_previous_start_of_month():
    """Previous first of month"""

    d =  Date(datetime.date(2021, 6, 15))\
        .start_of('month')\
        .subtract(days=1)\
        .start_of('month')
    assert_equal(d, Date(2021, 5, 1))

    d = Date(datetime.date(2021, 6, 15))\
        .start_of('month')\
        .subtract(days=1)\
        .business()\
        .start_of('month')
    assert_equal(d, Date(2021, 5, 3))


def test_first_of_year():

    assert_equal(
        Date().first_of('year'),
        datetime.date(pendulum.today().year, 1, 1))

    assert_equal(
        Date(datetime.date(2012, 12, 31)).first_of('year'),
        datetime.date(2012, 1, 1))


def test_last_of_year():

    d = Date(2022, 4, 2).last_of('year')
    assert_equal(d, Date(2022, 12, 31))


def test_last_of_quarter():
    """Return the quarter start or quarter end of a given date."""

    d = Date(datetime.date(2013, 11, 5)).last_of('quarter')
    assert_equal(d, Date(2013, 12, 31))

    d = Date(datetime.date(2016, 3, 31)).last_of('quarter')
    assert_equal(d, Date(2016, 3, 31))


def test_first_of_quarter():
    """Return the quarter start or quarter end of a given date."""

    d =  Date(datetime.date(2013, 11, 5)).first_of('quarter')
    assert_equal(d, Date(2013, 10, 1))

    d = Date(datetime.date(1999, 1, 19)).first_of('quarter')
    assert_equal(d, Date(1999, 1, 1))


def test_last_quarter_date():
    """Return the previous quarter start or quarter end of a given date.
    """

    d = Date(datetime.date(2013, 11, 5))\
        .first_of('quarter')\
        .subtract(days=1)
    assert_equal(d, Date(2013, 9, 30))
    d = Date(datetime.date(2016, 3, 31))\
        .first_of('quarter')\
        .subtract(days=1)
    assert_equal(d, Date(2015, 12, 31))

    d = Date(datetime.date(2013, 11, 5))\
        .first_of('quarter')\
        .subtract(days=1)\
        .first_of('quarter')
    assert_equal(d, Date(2013, 7, 1))
    d = Date(datetime.date(1999, 1, 19))\
        .first_of('quarter')\
        .subtract(days=1)\
        .first_of('quarter')
    assert_equal(d, Date(1998, 10, 1))


def test_loaded_module():
    """We change the module signature. Check that Pendulum still works
    """

    pendulum_date = pendulum.Date(2000, 1, 1)
    d = pendulum_date.add(days=10)
    assert_equal(d, Date(2000, 1, 11))


def test_add():
    # Closed on 12/5/2018 due to George H.W. Bush's death
    i, thedate = 5, datetime.date(2018, 11, 29)
    while i > 0:
        thedate = Date(thedate).business().add(days=1)
        i -= 1
    assert_equal(thedate, Date(2018, 12, 7))

    i, thedate = 5, datetime.date(2021, 11, 17)
    while i > 0:
        thedate = Date(thedate).business().add(days=1)
        i -= 1
    assert_equal(thedate, Date(2021, 11, 24))

    # Infinite date
    d = Date(datetime.date(9999, 12, 31)).business().add(days=1)
    assert_equal(d, Date(9999, 12, 31))

    # In one week (from following doctests)
    d = Date(datetime.date(2018, 11, 29)).business().add(days=5)
    assert_equal(d, Date(2018, 12, 7))
    d = Date(datetime.date(2021, 11, 17)).business().add(days=5)
    assert_equal(d, Date(2021, 11, 24))

    # Test reverse direction
    d = Date(datetime.date(2018, 11, 29)).business().subtract(days=-5)
    assert_equal(d, Date(2018, 12, 7))
    d = Date(datetime.date(2021, 11, 17)).business().subtract(days=-5)
    assert_equal(d, Date(2021, 11, 24))


def test_subtract():

    # Closed on 12/5/2018 due to George H.W. Bush's death
    d = Date(datetime.date(2018, 12, 7)).business().subtract(days=5)
    assert_equal(d, Date(2018, 11, 29))
    d = Date(datetime.date(2021, 11, 24)).business().subtract(days=5)
    assert_equal(d, Date(2021, 11, 17))

    # One week ago (from following doctests)
    d = Date(datetime.date(2018, 12, 7)).business().subtract(days=5)
    assert_equal(d, Date(2018, 11, 29))
    d = Date(datetime.date(2021, 11, 24)).subtract(days=7)
    assert_equal(d, Date(2021, 11, 17))
    d = Date(datetime.date(2018, 12, 7)).business().subtract(days=5)
    assert_equal(d, Date(2018, 11, 29))
    d = Date(datetime.date(2021, 11, 24)).subtract(days=7)
    assert_equal(d, Date(2021, 11, 17))

    # Test reverse direction
    d = Date(datetime.date(2018, 12, 7)).business().add(days=-5)
    assert_equal(d, Date(2018, 11, 29))
    d = Date(datetime.date(2021, 11, 24)).add(days=-7)
    assert_equal(d, Date(2021, 11, 17))
    d = Date(datetime.date(2018, 12, 7)).business().add(days=-5)
    assert_equal(d, Date(2018, 11, 29))
    d = Date(datetime.date(2021, 11, 24)).add(days=-7)
    assert_equal(d, Date(2021, 11, 17))


def test_set_entity():

    d = Date(2000, 1, 1).entity(NYSE).business().add(days=10)
    assert_equal(d, Date(2000, 1, 14))


def start_of_month_weekday(self, weekday='MO'):
    """Get first X of the month

    >>> Date(datetime.date(2014, 8, 1)).start_of_month_weekday('WE')
    Date(2014, 8, 6)
    >>> Date(datetime.date(2014, 7, 31)).start_of_month_weekday('WE')
    Date(2014, 7, 2)
    >>> Date(datetime.date(2014, 8, 6)).start_of_month_weekday('WE')
    Date(2014, 8, 6)
    """
    return self.start_of('month').next(day_obj.get(weekday))


def end_of_month_weekday(self, weekday='SU'):
    """Like `start`, but for the end X weekday of month"""
    return self.end_of('month').previous(day_obj.get(weekday))


def test_parse():

    # cross test between parsing and adding/subtracting negative
    # business days
    assert_equal(Date.parse('T-3b'), Date().business().subtract(days=3))
    assert_equal(Date.parse('T-3b'), Date().business().add(days=-3))
    assert_equal(Date.parse('T+3b'), Date().business().subtract(days=-3))
    assert_equal(Date.parse('T+3b'), Date().business().subtract(days=-3))

    assert_equal(Date.parse('P'), Date().business().subtract(days=1))
    assert_equal(Date.parse('P'), Date().business().add(days=-1))
    assert_equal(Date.parse('P-3b'), Date().business().add(days=-1).business().subtract(days=3))
    assert_equal(Date.parse('P-3b'), Date().business().subtract(days=1).business().add(days=-3))
    assert_equal(Date.parse('P+3b'), Date().business().add(days=-1).business().subtract(days=-3))
    assert_equal(Date.parse('P+3b'), Date().business().subtract(days=1).business().subtract(days=-3))

    assert_equal(Date('01/11/19'), Date(2019, 1, 11))
    assert_equal(Date.parse('01/11/19'), Date(2019, 1, 11))
    assert_equal(Date('01/15/20'), Date(2020, 1, 15))
    assert_equal(Date('01/15/21'), Date(2021, 1, 15))
    assert_equal(Date('01/15/22'), Date(2022, 1, 15))


def test_copy():

    d = pendulum.Date(2022, 1, 1)
    assert_equal(copy.copy(d), d)

    d = Date(2022, 1, 1)
    assert_equal(copy.copy(d), d)


def test_deepcopy():

    d = pendulum.Date(2022, 1, 1)
    assert_equal(copy.deepcopy(d), d)

    d = Date(2022, 1, 1)
    assert_equal(copy.deepcopy(d), d)


def test_pickle():

    d = Date(2022, 1, 1)

    with open('date.pkl', 'wb') as f:
        pickle.dump(d, f)
    with open('date.pkl', 'rb') as f:
        d_ = pickle.load(f)

    assert_equal(d, d_)


if __name__ == '__main__':
    pytest.main([__file__])
