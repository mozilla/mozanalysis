from enum import Enum
from typing import TypeAlias, Callable

import pandas as pd
import numpy as np
import numpy.typing as npt


class ComparativeOption(str, Enum):
    INDIVIDUAL = "individual"
    COMPARATIVE = "comparative"


class Uplift(str, Enum):
    ABSOLUTE = "abs_uplift"
    RELATIVE = "rel_uplift"


# generic
Numeric: TypeAlias = bool | int | float
NumericNDArray: TypeAlias = (
    npt.NDArray[np.bool_] | npt.NDArray[np.int_] | npt.NDArray[np.float_]
)
Estimates: TypeAlias = "pd.Series[float]"
BranchLabel: TypeAlias = str
EstimatesByBranch: TypeAlias = dict[BranchLabel, Estimates]

CompareBranchesOutput: TypeAlias = dict[ComparativeOption, EstimatesByBranch]

# bootstrapping specific

QuantilesType: TypeAlias = tuple[float, ...]
BootstrapSamples: TypeAlias = "pd.Series[Numeric]"

# each column is a parameter
# used when bootstrapping deciles or empirical cdf
ParameterizedBootstrapSamples: TypeAlias = pd.DataFrame
ParameterizedEstimates: TypeAlias = pd.DataFrame
ParameterizedEstimatesByBranch: TypeAlias = dict[BranchLabel, ParameterizedEstimates]
ParameterizedCompareBranchesOutput: TypeAlias = dict[
    ComparativeOption, ParameterizedEstimatesByBranch
]

StatFunctionReturnType: TypeAlias = Numeric | dict[str, Numeric]
StatFunctionType: TypeAlias = Callable[[BootstrapSamples], StatFunctionReturnType]

SamplesByBranch: TypeAlias = dict[BranchLabel, BootstrapSamples]
ParameterizedSamplesByBranch: TypeAlias = dict[
    BranchLabel, ParameterizedBootstrapSamples
]
AnySamplesByBranch = SamplesByBranch | ParameterizedSamplesByBranch


class AnalysisUnit(str, Enum):
    CLIENT = "client_id"
    PROFILE_GROUP = "profile_group_id"


class IncompatibleAnalysisUnit(ValueError):
    pass
