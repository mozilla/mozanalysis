# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Optional
import attr
from pytest import Config

from jetstream_config_parser.config import ConfigCollection

from mozanalysis.metrics import DataSource, Metric

class _ConfigLoader:
    config_collection: Optional[ConfigCollection] = attr.ib(None)

    @property
    def configs(self) -> ConfigCollection:
        if configs := getattr(self, "_configs", None):
            return configs

        if self.config_collection is None:
            self.config_collection = ConfigCollection.from_github_repo()
        self._configs = self.config_collection
        return self._configs

    def get_metric(self, metric_slug: str, app_name: str) -> Metric:
        metric_definition = self.configs.get_metric_definition(metric_slug, app_name)
        if metric_definition is None:
            raise Exception(f"Could not find definition for metric {metric_slug}")

        return Metric(
            name=metric_definition.name,
            select_expr=self.configs.get_env().from_string(metric_definition.select_expression).render(),
            friendly_name=metric_definition.friendly_name,
            description=metric_definition.friendly_name,
            data_source=self.configs.get_data_source(metric_definition.data_source.name, app_name),
            bigger_is_better=metric_definition.bigger_is_better
        )

    def get_data_source(self, data_source_slug: str, app_name: str) -> DataSource:
        data_source_definition = self.configs.get_data_source_definition(data_source_slug, app_name)
        if data_source_definition is None:
            raise Exception(f"Could not find definition for data source {data_source_slug}")

        return DataSource(
            name=data_source_definition.name,
            from_expr=data_source_definition.from_expression,
            client_id_column=data_source_definition.client_id_column,
            submission_date_column=data_source_definition.submission_date_column,
            experiments_column_type=data_source_definition.experiments_column_type,
            default_dataset=data_source_definition.default_dataset
        )

    def get_segment():
        pass

ConfigLoader = _ConfigLoader()
