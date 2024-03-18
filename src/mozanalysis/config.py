# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from dataclasses import dataclass

from metric_config_parser.config import ConfigCollection

METRIC_HUB_JETSTREAM_REPO = "https://github.com/mozilla/metric-hub/tree/main/jetstream"


class _ConfigLoader:
    """
    Loads config files from an external repository.

    Config objects are converted into mozanalysis native types.
    """

    config_collection: ConfigCollection | None = None

    @property
    def configs(self) -> ConfigCollection:
        configs = getattr(self, "_configs", None)
        if configs:
            return configs

        if self.config_collection is None:
            self.config_collection = ConfigCollection.from_github_repo()
        self._configs = self.config_collection
        return self._configs

    def with_configs_from(
        self, repo_urls: list[str] | None, is_private: bool = False
    ) -> "_ConfigLoader":
        """Load configs from another repository and merge with default configs."""
        if not repo_urls:
            return self

        config_collection = ConfigCollection.from_github_repos(
            repo_urls=repo_urls, is_private=is_private
        )
        self.configs.merge(config_collection)
        return self

    def get_metric(self, metric_slug: str, app_name: str):
        """Load a metric definition for the given app.

        Returns a :class:`mozanalysis.metrics.Metric` instance.
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
            description=metric_definition.description,
            data_source=self.get_data_source(
                metric_definition.data_source.name, app_name
            ),
            bigger_is_better=metric_definition.bigger_is_better,
            app_name=app_name,
        )

    def get_data_source(self, data_source_slug: str, app_name: str):
        """Load a data source definition for the given app.

        Returns a :class:`mozanalysis.metrics.DataSource` instance.
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
            experiments_column_type=(
                None
                if data_source_definition.experiments_column_type == "none"
                else data_source_definition.experiments_column_type
            ),
            default_dataset=data_source_definition.default_dataset,
            app_name=app_name,
        )

    def get_segment(self, segment_slug: str, app_name: str):
        """Load a segment definition for the given app.

        Returns a :class:`mozanalysis.segments.Segment` instance.
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
            app_name=app_name,
        )

    def get_segment_data_source(self, data_source_slug: str, app_name: str):
        """Load a segment data source definition for the given app.

        Returns a :class:`mozanalysis.segments.SegmentDataSource` instance.
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
            app_name=app_name,
        )

    def get_outcome_metric(self, metric_slug: str, outcome_slug: str, app_name: str):
        """Load a metric definition from an outcome defined for the given app.

        Parametrized metrics are not supported, since they may not be defined outside
        of an experiment.

        Returns a :class:`mozanalysis.metrics.Metric` instance.
        """
        from mozanalysis.metrics import Metric

        if not self.configs.outcomes:
            self.with_configs_from([METRIC_HUB_JETSTREAM_REPO])

        outcome_spec = self.configs.spec_for_outcome(
            slug=outcome_slug, platform=app_name
        )
        if not outcome_spec:
            raise Exception(f"Could not find definition for outcome {outcome_slug}")

        metric_definition = outcome_spec.metrics.get(metric_slug)
        if metric_definition is None:
            raise Exception(
                f"Could not find definition for metric {metric_slug}"
                + f" in outcome {outcome_slug}"
            )

        @dataclass
        class MinimalConfiguration:
            app_name: str

        conf = MinimalConfiguration(app_name)
        summaries = metric_definition.resolve(outcome_spec, conf, self.configs)
        metric = summaries[0].metric

        return Metric(
            name=metric.name,
            select_expr=metric.select_expression,
            friendly_name=metric.friendly_name,
            description=metric.description,
            data_source=metric.data_source,
            bigger_is_better=metric.bigger_is_better,
        )

    def get_outcome_data_source(
        self, data_source_slug: str, outcome_slug: str, app_name: str
    ):
        """Load a data source definition from an outcome defined for the given app.

        Returns a :class:`mozanalysis.metrics.DataSource` instance.
        """
        from mozanalysis.metrics import DataSource

        if not self.configs.outcomes:
            self.with_configs_from([METRIC_HUB_JETSTREAM_REPO])

        outcome_spec = self.configs.spec_for_outcome(
            slug=outcome_slug, platform=app_name
        )
        if not outcome_spec:
            raise Exception(f"Could not find definition for outcome {outcome_slug}")

        data_source_definition = outcome_spec.data_sources.definitions.get(
            data_source_slug
        )
        if data_source_definition is None:
            raise Exception(
                f"Could not find definition for data source {data_source_slug}"
                + f" in outcome {outcome_slug}"
            )

        return DataSource(
            name=data_source_definition.name,
            from_expr=data_source_definition.from_expression,
            client_id_column=data_source_definition.client_id_column,
            submission_date_column=data_source_definition.submission_date_column,
            experiments_column_type=(
                None
                if data_source_definition.experiments_column_type == "none"
                else data_source_definition.experiments_column_type
            ),
            default_dataset=data_source_definition.default_dataset,
        )


ConfigLoader = _ConfigLoader()
