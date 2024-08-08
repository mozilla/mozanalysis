from enum import Enum
from typing import TypeAlias

import pandas as pd
import numpy as np


class ComparativeOption(str, Enum):
    INDIVIDUAL = "individual"
    COMPARATIVE = "comparative"


class Uplift(str, Enum):
    ABSOLUTE = "abs_uplift"
    RELATIVE = "rel_uplift"


Numeric: TypeAlias = bool | int | float
Estimates: TypeAlias = "pd.Series[float]"
BranchLabel: TypeAlias = str
EstimatesByBranch: TypeAlias = dict[BranchLabel, Estimates]

CompareBranchesOutput: TypeAlias = dict[ComparativeOption, EstimatesByBranch]


class AnalysisUnit(str, Enum):
    CLIENT = "client_id"
    PROFILE_GROUP = "profile_group_id"


class IncompatibleAnalysisUnit(ValueError):
    pass
