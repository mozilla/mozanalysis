from enum import StrEnum
from typing import TypeAlias

import pandas as pd


class ComparativeOption(StrEnum):
    INDIVIDUAL = "individual"
    COMPARATIVE = "comparative"


class Uplift(StrEnum):
    ABSOLUTE = "abs_uplift"
    RELATIVE = "rel_uplift"


Estimates: TypeAlias = pd.Series
BranchLabel = str
EstimatesByBranch = dict[BranchLabel, Estimates]

CompareBranchesOutput = dict[ComparativeOption, EstimatesByBranch]


class IncompatibleAnalysisUnit(ValueError):
    pass
