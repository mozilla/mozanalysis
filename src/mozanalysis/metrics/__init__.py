# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import attr
from pyspark.sql import functions as F


@attr.s(frozen=True, slots=True)
class DataSource(object):
    """Wrapper for a Spark DataFrame.

    This class is lazy: no Spark activity takes place until
    ``get_dataframe()`` is called.

    Examples:
        There are three ways to instantiate a ``DataSource``::

            clients_daily = DataSource.from_table_name('clients_daily')

        or::

            @DataSource.from_func()
            def clients_daily(spark, experiment):
                return spark.table('clients_daily')

        or (for interactive use)::

            clients_daily_df = spark.table('clients_daily')
            clients_daily_ds = DataSource('clients_daily', clients_daily_df)

    Avoid instantiating duplicate ``DataSource`` s - if multiple metrics
    require the same source of data but use different ``DataSource``
    instances, then in some circumstances ``mozanalysis`` will not
    recognise that the metrics could share a DataFrame, and Spark will
    do extra work.
    """
    name = attr.ib(validator=attr.validators.instance_of(str))
    _dataframe_getter = attr.ib(validator=attr.validators.is_callable())
    description = attr.ib(type=str, default=None)

    @classmethod
    def from_table_name(cls, name):
        """Return a ``DataSource`` from a Spark table name.

        Don't repeatedly call this with the same ``name``, because
        we won't recognise that the different ``DataSource`` instances
        are equal.

        Args:
            name (str): Supplied to ``spark.table()``

        Example::

            clients_daily = DataSource.from_table_name('clients_daily')
        """
        return cls(
            name=name,
            dataframe_getter=lambda spark, experiment: spark.table(name)
        )

    @classmethod
    def from_func(cls):
        """Decorator method to turn a function into a ``DataSource``.

        The function must take two arguments: ``spark`` and
        ``experiment``. The function's name is used as the
        ``DataSource`` name.

        Example::

            from mozanalysis.experiment import Experiment
            from pyspark.sql.session import SparkSession

            DataSource.from_func()
            def clients_daily(spark: SparkSession, experiment: Experiment):
                return spark.table('clients_daily')
        """
        def f(dataframe_getter):
            return cls(
                name=dataframe_getter.__name__,
                dataframe_getter=dataframe_getter,
                description=dataframe_getter.__doc__
            )

        return f

    @classmethod
    def from_dataframe(cls, name, dataframe):
        """Return a ``DataSource`` for a DataFrame.

        Don't repeatedly call this with the same ``dataframe``, because
        we won't recognise that the different ``DataSource`` instances
        are equal.

        Args:
            name (str): Name for the ``DataSource``.
            dataframe (DataFrame): The DataFrame to wrap.
        """
        return cls(
            name=name,
            dataframe_getter=lambda spark, experiment: dataframe
        )

    def get_dataframe(self, spark, experiment):
        """Return the DataFrame for this ``DataSource``.

        We cache the results so that repeated calls to
        ``get_dataframe()`` return the same DataFrame, which makes it
        easier to compute multiple metrics from one DataFrame. The
        results are cached on the SparkSession, keyed by the
        ``DataSource`` instance and the ``Experiment``.

        Args:
            spark (SparkSession): The SparkSession - in DataBricks this
                is preloaded into every notebook as the variable
                ``spark``.
            experiment (Experiment): The Experiment being analysed. Some
                ``DataSource`` s require the experiment slug in order to
                identify the correct data.
        """
        key = (self, experiment)
        try:
            return spark._MOZANALYSIS_DATA_SOURCE_CACHE[key]
        except AttributeError:
            spark._MOZANALYSIS_DATA_SOURCE_CACHE = {}
        except KeyError:
            pass

        dataframe = self._dataframe_getter(spark, experiment)
        spark._MOZANALYSIS_DATA_SOURCE_CACHE[key] = dataframe

        return spark._MOZANALYSIS_DATA_SOURCE_CACHE[key]

    @staticmethod
    def _has_experiments_map(dataframe):
        return dict(dataframe.dtypes).get('experiments') == 'map<string,string>'

    @staticmethod
    def _has_glean_style_experiments_field(dataframe):
        if 'ping_info' not in dataframe.columns:
            return False

        ping_info_fields = dataframe.schema[
            'ping_info'
        ].jsonValue().get('type', {}).get('fields', [])

        for f in ping_info_fields:
            if f.get('name') == 'experiments':
                return True

        return False

    def get_sanity_metric_cols(self, experiment, enrollments):
        """Return a list of sanity metric Spark Columns.

        Args:
            experiment (Experiment): The experiment being analysed.
            enrollments (DataFrame): The DataFrame of enrollments,
                typically obtained from ``Experiment.get_enrollments()``

        The sanity metrics depend on the data source.

        Could return:
            * ``[data_source_name]_has_non_enrolled_data``: Check to see
              whether the client_id was sending data in the analysis
              window that wasn't tagged as being part of the experiment.
              Indicates either a client_id clash, or the client
              unenrolling. The fraction of such users should be small,
              and similar between branches.
            * ``[data_source_name]_has_contradictory_branch``: Check to
              see whether the client_id is also enrolled in other
              branches. Indicates problems, e.g. cloned profiles. The
              fraction of such users should be small, and similar
              between branches.


        """
        if 'branch' not in enrollments.columns:
            # Not running in an experiment context; the below are meaningless
            return []

        spark = enrollments.sql_ctx.sparkSession
        ds_df = self.get_dataframe(spark, experiment)

        if self._has_experiments_map(ds_df):
            # Regular desktop telemetry
            experiments_col = ds_df.experiments
            reported_branch = ds_df.experiments[experiment.experiment_slug]

        elif self._has_glean_style_experiments_field(ds_df):
            # Glean
            experiments_col = ds_df.ping_info.experiments
            reported_branch = experiments_col[experiment.experiment_slug].branch

        else:
            return []

        return [

            agg_any(
                experiments_col.isNotNull()
                & experiments_col[experiment.experiment_slug].isNull()
            ).alias(self.name + '_has_non_enrolled_data'),

            agg_any(reported_branch != enrollments.branch).alias(
                self.name + '_has_contradictory_branch'
            ),
        ]


@attr.s(frozen=True, slots=True)
class Metric(object):
    """Represents an experiment metric.

    Needs to be combined with an analysis window to be measurable!

    Essentially this wraps a Spark Column and a reference to a
    DataSource that can be used to obtain the DataFrame the column
    belongs to.

    These wrappers are needed in order for the metrics modules to be
    lazy.

    When defining metrics in the metric library, use the
    ``Metric.from_func()`` decorator.

    When defining metrics in interactive mode (i.e. in a notebook
    where you already have the Spark Column and DataFrame, and
    don't need to be lazy), use ``Metric.from_col()``.

    Examples:
        In the metrics library::

            clients_daily = DataSource.from_table_name('clients_daily')

            @Metric.from_func(clients_daily)
            def active_hours(cd):
                return agg_sum(cd.active_hours_sum)

        In interactive mode::

            cd = spark.table('clients_daily')

            active_hours = Metric.from_col(
                name='active_hours',
                col=agg_sum(cd.active_hours_sum),
                df_name='cd',
                df=cd,
            )
    """
    name = attr.ib(type=str)
    data_source = attr.ib(validator=attr.validators.instance_of(DataSource))
    _col_getter = attr.ib(validator=attr.validators.is_callable())
    description = attr.ib(type=str, default=None)

    @classmethod
    def from_func(cls, data_source):
        """Decorator method to instantiate a ``Metric``.

        Args:
            data_source (DataSource): A ``DataSource`` that may be used
                to obtain a DataFrame with the data required to compute
                this ``Metric``.

        The wrapped function needs to have one argument (the DataFrame
        output by ``data_source.get_dataframe()``), and return a Column.
        The function's name is taken as the ``Metric``'s name.

        Example::

            clients_daily = DataSource.from_table_name('clients_daily')

            @Metric.from_func(clients_daily)
            def active_hours(cd: DataFrame) -> Column:
                return agg_sum(cd.active_hours_sum)
        """
        def f(col_getter):
            return cls(
                name=col_getter.__name__,
                data_source=data_source,
                col_getter=col_getter,
                description=col_getter.__doc__
            )

        return f

    @classmethod
    def from_col(cls, metric_name, col, data_source):
        """Return a ``Metric`` defined by ``col``.

        Args:
            metric_name (str): Name for the resulting Column.
            col (Column): Spark Column representing the computed metric.
            data_source (DataSource): DataSource wrapper for the source
                DataFrame for the Column.

        Example::

            cd_ds = DataSource.from_table_name('clients_daily')

            active_hours = Metric.from_col(
                name='active_hours',
                col=agg_sum(cd.active_hours_sum),
                data_source=cd_ds
            )
        """
        return cls(
            name=metric_name,
            data_source=data_source,
            col_getter=lambda _: col
        )

    def get_col(self, spark, experiment):
        """Return a ``Column`` representing this metric.

        Args:
            spark (SparkSession): The Spark Session.
            experiment (Experiment): The experiment for which this
                metric is being computed.
        """
        # TODO: will this ever need access to ``enrollments``?
        df = self.data_source.get_dataframe(spark, experiment)

        try:
            col = self._col_getter(df)
        except TypeError:
            col = self._col_getter(df, experiment)
        return col.alias(self.name)


def agg_sum(col):
    """Return the sum over the data, with 0-filled nulls.

    Args:
        col (Column): A Spark Column of a numeric type
    """
    return F.coalesce(F.sum(col), F.lit(0))


def agg_any(col):
    """Return the bitwise OR, as an int, with 0-filled nulls.

    Args:
        col (Column): A Spark Column of bools
    """
    return F.coalesce(F.max(col).astype('int'), F.lit(0))
