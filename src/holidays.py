"""Irish public holidays (Republic of Ireland), including weekend substitutions."""

from __future__ import annotations

import datetime as dt

import pandas as pd


def _easter(year: int) -> dt.date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return dt.date(year, month, day)


def _first_monday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    return d + dt.timedelta(days=(7 - d.weekday()) % 7)


def _last_monday(year: int, month: int) -> dt.date:
    d = (dt.date(year, month + 1, 1) if month < 12 else dt.date(year + 1, 1, 1)) - dt.timedelta(days=1)
    return d - dt.timedelta(days=d.weekday())


def irish_holidays(years) -> set[dt.date]:
    out: set[dt.date] = set()
    for y in years:
        fixed = [dt.date(y, 1, 1), dt.date(y, 3, 17), dt.date(y, 12, 25), dt.date(y, 12, 26)]
        moving = [_easter(y) + dt.timedelta(days=1), _first_monday(y, 5), _first_monday(y, 6),
                  _first_monday(y, 8), _last_monday(y, 10)]
        if y >= 2023:  # St Brigid's Day: first Monday of February, or 1 Feb if a Friday
            feb1 = dt.date(y, 2, 1)
            moving.append(feb1 if feb1.weekday() == 4 else _first_monday(y, 2))
        if y == 2022:
            moving.append(dt.date(2022, 3, 18))  # one-off COVID-19 recognition holiday
        out.update(moving)
        for d in fixed:  # weekend holidays move to the next free weekday
            out.add(d)
            if d.weekday() >= 5:
                sub = d + dt.timedelta(days=1)
                while sub.weekday() >= 5 or sub in out or sub in fixed:
                    sub += dt.timedelta(days=1)
                out.add(sub)
    return out


def holiday_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    days = index.normalize()
    hol = irish_holidays(range(index.min().year - 1, index.max().year + 2))
    is_hol = pd.Series(days.date, index=index).isin(hol)
    xmas = ((days.month == 12) & (days.day >= 24)) | ((days.month == 1) & (days.day == 1))
    return pd.DataFrame({"is_holiday": is_hol.astype(int).values,
                         "christmas_period": xmas.astype(int)}, index=index)
