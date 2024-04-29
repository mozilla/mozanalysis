import pandas as pd
from pandas_gbq import read_gbq
from pandas_gbq.exceptions import GenericGBQException
from metric_config_parser.metric import AnalysisPeriod
from mozilla_nimbus_schemas.jetstream import AnalysisBasis
from typing import Dict, List
from dataclasses import dataclass
import re
from mozanalysis.visualization.types import Statistic


@dataclass
class ResultType:
    metric: str
    statistics: List[Statistic]


class ViewNotFound(Exception):
    pass


class StatisticsData:
    "Class for holding downloaded data"

    def __init__(self, slug):
        self.bq_slug = self.bq_normalize_name(slug)
        self._slug_raw = slug
        self._project_id = "moz-fx-data-experiments"

        self._view_paths: Dict[AnalysisPeriod, str] = dict()
        self._table_prefix: Dict[AnalysisPeriod, str] = dict()
        self._dfs: Dict[AnalysisPeriod, pd.DataFrame] = dict()

        for period in AnalysisPeriod:
            table_name = f"statistics_{self.bq_slug}_{period.table_suffix}"
            table_path = self._fully_qualify_statistics_table(table_name)
            self._view_paths[period] = table_path

            first_part = f"statistics_{self.bq_slug}_{period.value}"
            table_prefix = self._fully_qualify_statistics_table(first_part)
            self._table_prefix[period] = table_prefix

    def period_data(self, period: AnalysisPeriod) -> pd.DataFrame:
        if self._dfs.get(period) is None:
            self._download_statistics(period)
        return self._dfs[period]

    def period_path(self, period: AnalysisPeriod) -> str:
        return self._view_paths[period]

    def _download_statistics(self, period: AnalysisPeriod) -> None:
        try:
            df = read_gbq(
                f"SELECT * FROM {self._view_paths[period]}",
                project_id=self._project_id,
                progress_bar_type=None,
            )
        except GenericGBQException:
            raise ViewNotFound
        self._dfs[period] = df

    @staticmethod
    def _fully_qualify_statistics_table(table_name) -> str:
        return f"moz-fx-data-experiments.mozanalysis.{table_name}"

    @staticmethod
    def bq_normalize_name(name: str) -> str:
        # stolen from Jetstream
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    def analysis_bases_for_period(self, period: AnalysisPeriod) -> List[AnalysisBasis]:
        df = self.period_data(period)
        bases = [AnalysisBasis[ab] for ab in df.analysis_basis.unique()]
        return bases

    def result_types_for_period_and_basis(
        self, period: AnalysisPeriod, basis: AnalysisBasis
    ) -> List[ResultType]:
        df = self.period_data(period)
        basis_df = df.loc[df.analysis_basis == basis.value,]
        pairs = basis_df[["metric", "statistic"]].drop_duplicates().groupby("metric")
        out = []
        for metric, stats_df in pairs:
            stats = []
            for stat_str in stats_df.statistic:
                try:
                    stat = Statistic[stat_str]
                    stats.append(stat)
                except:
                    stat = Statistic.UNKNOWN
                    stats.append(stat)
            if len(stats) > 0:
                stats.sort(key=lambda s: s.value)
                out.append(ResultType(metric, stats))

        return sorted(out, key=lambda k: k.metric)

    def branches_for_period(self, period: AnalysisPeriod) -> List[str]:
        df = self.period_data(period)
        branches = df.branch.unique()
        return [branch for branch in branches]
