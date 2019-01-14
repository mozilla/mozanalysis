Changelog
---------

Next
^^^^

- ExperimentAnalysis.analyze no longer unpersists the dataset.


2018.12.0 (2018-12-05)
^^^^^^^^^^^^^^^^^^^^^^

- Updated to use basic bootstrap for confidence intervals (Fixes issue #5)
- Add PySpark UDFs for operating on histogram columns to facilitate per-user
  aggregation: histogram sum, count of events, quantiles, and fraction of
  events exceeding a threshold (Fixes issue #10)
- Revise metric definitions to avoid crashing in bootstrapping if a client had
  NULL values after daily aggregation
- Fix `EngagementIntensity` to correctly calculated active hours / total hours

2018.11.1 (2018-11-01)
^^^^^^^^^^^^^^^^^^^^^^

- Added `ExperimentAnalysis` class for automated experiment analysis
- Added `MetricDefinition` and defined 4 engagement metrics

2018.9.1 (2018-09-19)
^^^^^^^^^^^^^^^^^^^^^

- Moved `bootstrap` import path to `mozanalysis.stats.bootstrap`
- Added changelog

2018.9.0 (2018-09-18)
^^^^^^^^^^^^^^^^^^^^^

- Initial release
