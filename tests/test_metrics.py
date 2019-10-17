import mozanalysis.metrics as mm

from pyspark.sql import Column


def test_agg_sum(spark):
    df = register_fixture(spark)

    res = df.groupBy('client_id').agg(
        mm.agg_sum(df.numeric_col).alias('metric_value')
    ).toPandas().set_index('client_id').metric_value

    assert res['aaaa'] == 2
    assert res['bb'] == 0
    assert res['ccc'] == 5
    assert res['dd'] == 0


def test_agg_any(spark):
    df = register_fixture(spark)

    res = df.groupBy('client_id').agg(
        mm.agg_any(df.bool_col).alias('metric_value')
    ).toPandas().set_index('client_id').metric_value

    assert res['aaaa'] == 1
    assert res['bb'] == 0
    assert res['ccc'] == 1
    assert res['dd'] == 0


def register_fixture(spark, name='simple_fixture'):
    """Register a data source fixture as a table"""
    df = spark.createDataFrame(
        [
            ('aaaa', 1, True),
            ('aaaa', 1, True),
            ('aaaa', None, None),
            ('aaaa', 0, False),
            ('bb', None, None),
            ('ccc', 5, True),
            ('dd', 0, False),
        ],
        ["client_id", "numeric_col", "bool_col"]
    )
    df.createOrReplaceTempView(name)

    return df


def test_data_source_from_table(spark):
    name = 'name_to_care_about'
    orig_df = register_fixture(spark, name)

    ds = mm.DataSource.from_table_name(name)

    assert ds.name == name
    ds_df = ds.get_dataframe(spark, None)
    assert (ds_df.toPandas().fillna(0) == orig_df.toPandas().fillna(0)).all().all()


def test_data_source_from_func(spark):
    @mm.DataSource.from_func()
    def ds(spark, experiment):
        return register_fixture(spark)

    assert isinstance(ds, mm.DataSource)

    assert ds.name == 'ds'
    assert ds.get_dataframe(spark, None).count() > 0


def test_data_source_from_dataframe(spark):
    name = 'bob'

    orig_df = register_fixture(spark)

    ds = mm.DataSource.from_dataframe(name, orig_df)

    assert ds.name == 'bob'

    ds_df = ds.get_dataframe(spark, None)
    assert orig_df == ds_df  # Same spark object


def data_source_fixture(spark, name='a_data_source'):
    return mm.DataSource.from_dataframe(
        name,
        register_fixture(spark)
    )


def test_metric_from_func(spark):
    ds = data_source_fixture(spark)

    @mm.Metric.from_func(ds)
    def a_lovely_metric(df):
        """Hi there!"""
        return mm.agg_sum(df.numeric_col)

    assert isinstance(a_lovely_metric, mm.Metric)
    assert a_lovely_metric.name == 'a_lovely_metric'
    assert a_lovely_metric.data_source == ds
    assert isinstance(a_lovely_metric.get_col(spark, None), Column)
    assert a_lovely_metric.description == 'Hi there!'


def test_metric_from_col(spark):
    orig_df = register_fixture(spark)

    ds = mm.DataSource.from_dataframe('an_ordinary_data_source', orig_df)

    metric = mm.Metric.from_col(
        'a_special_metric',
        mm.agg_sum(orig_df.numeric_col),
        ds
    )

    assert metric.name == 'a_special_metric'
    assert metric.data_source.get_dataframe(spark, None) == orig_df

    res = orig_df.groupBy('client_id').agg(
        metric.get_col(spark, None)
    ).toPandas().set_index('client_id').a_special_metric

    assert res['aaaa'] == 2
    assert res['bb'] == 0
    assert res['ccc'] == 5
    assert res['dd'] == 0
