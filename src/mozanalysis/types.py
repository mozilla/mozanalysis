from enum import Enum
from typing import TypeAlias

import pandas as pd


class ComparativeOption(str, Enum):
    INDIVIDUAL = "individual"
    COMPARATIVE = "comparative"


class Uplift(str, Enum):
    ABSOLUTE = "abs_uplift"
    RELATIVE = "rel_uplift"


Estimates: TypeAlias = pd.Series
BranchLabel = str
EstimatesByBranch = dict[BranchLabel, Estimates]

CompareBranchesOutput = dict[ComparativeOption, EstimatesByBranch]


class AnalysisUnit(str, Enum):
    CLIENT_ID = "client_id"
    GROUP_ID = "profile_group_id"
