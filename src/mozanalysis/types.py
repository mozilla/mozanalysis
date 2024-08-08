from enum import Enum
from typing import NewType, Any

import pandas as pd


class ComparativeOption(str, Enum):
    INDIVIDUAL = "individual"
    COMPARATIVE = "comparative"


class Uplift(str, Enum):
    ABSOLUTE = "abs_uplift"
    RELATIVE = "rel_uplift"


Estimates = NewType("Estimates", pd.Series[Any])
BranchLabel = NewType("BranchLabel", str)
EstimatesByBranch = dict[BranchLabel, Estimates]

CompareBranchesOutput = dict[ComparativeOption, EstimatesByBranch]


class AnalysisUnit(str, Enum):
    CLIENT = "client_id"
    PROFILE_GROUP = "profile_group_id"


class IncompatibleAnalysisUnit(ValueError):
    pass
