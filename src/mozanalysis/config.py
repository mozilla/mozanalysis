# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Optional

from jinja2 import UndefinedError

from metric_config_parser.config import ConfigCollection
from metric_config_parser.outcome import OutcomeSpec
from metric_config_parser.data_source import DataSourceDefinition
from metric_config_parser.metric import MetricDefinition


class _ConfigLoader:
    """
    Loads config files from an external repository.

    Config objects are converted into mozanalysis native types.
    """

    config_collection: Optional[ConfigCollection] = None
    jetstream_config_collection: Optional[ConfigCollection] = None

    @property
    def configs(self) -> ConfigCollection:
        configs = getattr(self, "_configs", None)
        if configs:
            return configs

        if self.config_collection is None:
            self.config_collection = ConfigCollection.from_github_repo()
        self._configs = self.config_collection
        return self._configs

    @property
    def jetstream_configs(self) -> ConfigCollection:
        configs = getattr(self, "_jetstream_configs", None)
        if configs:
            return configs

        if self.jetstream_config_collection is None:
            self.jetstream_config_collection = ConfigCollection.from_github_repo(
                path="jetstream"
            )
        self._jetstream_configs = self.jetstream_config_collection
        return self._jetstream_configs

    def get_metric(self, metric_slug: str, app_name: str):
        """Load a metric definition for the given app.

        Returns a mozanalysis `Metric` instance.
        """
        from mozanalysis.metrics import Metric

        metric_definition = self.configs.get_metric_definition(metric_slug, app_name)
        if metric_definition is None:
            raise Exception(f"Could not find definition for metric {metric_slug}")

        return Metric(
            name=metric_definition.name,
            select_expr=self.configs.get_env()
            .from_string(metric_definition.select_expression)
            .render(),
            friendly_name=metric_definition.friendly_name,
            description=metric_definition.friendly_name,
            data_source=self.get_data_source(
                metric_definition.data_source.name, app_name
            ),
            bigger_is_better=metric_definition.bigger_is_better,
        )

    def get_data_source(self, data_source_slug: str, app_name: str):
        """Load a data source definition for the given app.

        Returns a mozanalysis `DataSource` instance.
        """
        from mozanalysis.metrics import DataSource

        data_source_definition = self.configs.get_data_source_definition(
            data_source_slug, app_name
        )
        if data_source_definition is None:
            raise Exception(
                f"Could not find definition for data source {data_source_slug}"
            )

        return DataSource(
            name=data_source_definition.name,
            from_expr=data_source_definition.from_expression,
            client_id_column=data_source_definition.client_id_column,
            submission_date_column=data_source_definition.submission_date_column,
            experiments_column_type=None
            if data_source_definition.experiments_column_type == "none"
            else data_source_definition.experiments_column_type,
            default_dataset=data_source_definition.default_dataset,
        )

    def get_segment(self, segment_slug: str, app_name: str):
        """Load a segment definition for the given app.

        Returns a mozanalysis `Segment` instance.
        """
        from mozanalysis.segments import Segment

        segment_definition = self.configs.get_segment_definition(segment_slug, app_name)
        if segment_definition is None:
            raise Exception(f"Could not find definition for segment {segment_slug}")

        return Segment(
            name=segment_definition.name,
            data_source=self.get_segment_data_source(
                segment_definition.data_source.name, app_name
            ),
            select_expr=self.configs.get_env()
            .from_string(segment_definition.select_expression)
            .render(),
            friendly_name=segment_definition.friendly_name,
            description=segment_definition.description,
        )

    def get_segment_data_source(self, data_source_slug: str, app_name: str):
        """Load a segment data source definition for the given app.

        Returns a mozanalysis `SegmentDataSource` instance.
        """
        from mozanalysis.segments import SegmentDataSource

        data_source_definition = self.configs.get_segment_data_source_definition(
            data_source_slug, app_name
        )
        if data_source_definition is None:
            raise Exception(
                f"Could not find definition for segment data source {data_source_slug}"
            )

        return SegmentDataSource(
            name=data_source_definition.name,
            from_expr=data_source_definition.from_expression,
            window_start=data_source_definition.window_start,
            window_end=data_source_definition.window_end,
            client_id_column=data_source_definition.client_id_column,
            submission_date_column=data_source_definition.submission_date_column,
            default_dataset=data_source_definition.default_dataset,
        )

    def get_outcome_metric(self, metric_slug: str, outcome_slug: str, app_name: str):
        """Load a metric definition from an outcome defined for the given app.

        Parametrized metrics are not supported, since they may not be defined outside of an experiment.

        Returns a mozanalysis `Metric` instance.
        """
        from mozanalysis.metrics import Metric

        def _get_metric_definition(
            spec: OutcomeSpec, slug: str
        ) -> Optional[MetricDefinition]:
            for m_slug, metric in spec.metrics.items():
                if m_slug == slug:
                    return metric
            return None

        outcome_spec = self.jetstream_configs.spec_for_outcome(
            slug=outcome_slug, platform=app_name
        )
        if not outcome_spec:
            raise Exception(f"Could not find definition for outcome {outcome_slug}")
        metric_definition = _get_metric_definition(outcome_spec, metric_slug)
        if metric_definition is None:
            raise Exception(
                f"Could not find definition for metric {metric_slug} in outcome {outcome_slug}"
            )

        # Data source associated with the metric can either be defined in the outcome or in the
        # general definition file for the app. Outcome definitions take precedence.
        try:
            data_source = self.get_outcome_data_source(
                metric_definition.data_source.name, outcome_slug, app_name
            )
        except Exception:
            data_source = self.get_data_source(
                metric_definition.data_source.name, app_name
            )

        # Functions used in templated metric definitions are defined both under `jetstream/` and at the top level.
        # Merge these with functions under `jetstream/` taking precedence.
        jinja_env = self.configs.get_env()
        jinja_env.globals.update(self.jetstream_configs.get_env().globals)

        try:
            select_expr = jinja_env.from_string(
                metric_definition.select_expression
            ).render()
        except UndefinedError as e:
            if "parameters" in str(e):
                raise NotImplementedError(
                    "Parametrized outcome metrics are not supported"
                )
            else:
                raise

        return Metric(
            name=metric_definition.name,
            select_expr=select_expr,
            friendly_name=metric_definition.friendly_name,
            description=metric_definition.friendly_name,
            data_source=data_source,
            bigger_is_better=metric_definition.bigger_is_better,
        )

    def get_outcome_data_source(
        self, data_source_slug: str, outcome_slug: str, app_name: str
    ):
        """Load a data source definition from an outcome defined for the given app.

        Returns a mozanalysis `DataSource` instance.
        """
        from mozanalysis.metrics import DataSource

        def _get_data_source_definition(
            spec: OutcomeSpec, slug: str
        ) -> Optional[DataSourceDefinition]:
            for ds_slug, data_source in spec.data_sources.definitions.items():
                if ds_slug == slug:
                    return data_source
            return None

        outcome_spec = self.jetstream_configs.spec_for_outcome(
            slug=outcome_slug, platform=app_name
        )
        if not outcome_spec:
            raise Exception(f"Could not find definition for outcome {outcome_slug}")
        data_source_definition = _get_data_source_definition(
            outcome_spec, data_source_slug
        )
        if data_source_definition is None:
            raise Exception(
                f"Could not find definition for data source {data_source_slug} in outcome {outcome_slug}"
            )

        return DataSource(
            name=data_source_definition.name,
            from_expr=data_source_definition.from_expression,
            client_id_column=data_source_definition.client_id_column,
            submission_date_column=data_source_definition.submission_date_column,
            experiments_column_type=None
            if data_source_definition.experiments_column_type == "none"
            else data_source_definition.experiments_column_type,
            default_dataset=data_source_definition.default_dataset,
        )


ConfigLoader = _ConfigLoader()
