from date import NYSE, DateTime, Entity

__all__ = ['is_within_business_hours', 'is_business_day']


def is_within_business_hours(entity: Entity = NYSE) -> bool:
    """Return whether the current native datetime is between
    open and close of business hours.

    >>> from unittest.mock import patch
    >>> tz = NYSE.tz

    >>> with patch('date.DateTime.now') as mock:
    ...     mock.return_value = DateTime(2000, 5, 1, 12, 30, 0, 0, tzinfo=tz)
    ...     is_within_business_hours()
    True

    >>> with patch('date.DateTime.now') as mock:
    ...     mock.return_value = DateTime(2000, 7, 2, 12, 15, 0, 0, tzinfo=tz) # Sunday
    ...     is_within_business_hours()
    False

    >>> with patch('date.DateTime.now') as mock:
    ...     mock.return_value = DateTime(2000, 11, 1, 1, 15, 0, 0, tzinfo=tz)
    ...     is_within_business_hours()
    False

    """
    this = DateTime.now()
    this_entity = DateTime(entity=entity, tzinfo=entity.tz).now()
    bounds = this_entity.business_hours()
    return this_entity.business_open() and (bounds[0] <= this.astimezone(entity.tz) <= bounds[1])


def is_business_day(entity: Entity = NYSE) -> bool:
    """Return whether the current native datetime is a business day.
    """
    return DateTime.now().astimezone(entity.tz).is_business_day()


if __name__ == '__main__':
    __import__('doctest').testmod(optionflags=4 | 8 | 32)
