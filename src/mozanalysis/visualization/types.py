from enum import Enum


class Statistic(Enum):
    mean = 1
    binomial = 2
    deciles = 3
    count = 4
    empirical_cdf = 5
    per_client_dau_impact = 6
    UNKNOWN = 7


class TimeRange(Enum):
    OneTime = 1
    TimeSeries = 2
