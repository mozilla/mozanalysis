# Mozilla Experiments Analysis [![CircleCI](https://circleci.com/gh/mozilla/mozanalysis.svg?style=svg)](https://circleci.com/gh/mozilla/mozanalysis) [![codecov](https://codecov.io/gh/mozilla/mozanalysis/branch/master/graph/badge.svg)](https://codecov.io/gh/mozilla/mozanalysis)

The `mozanalysis` Python library is a library to standardize experiment analysis
at Mozilla for the purpose of producing decision reports templates that are
edited by data scientists.

## [TODO] Installing from pypi
- To install this package from pypi run:
```
pip install mozanalysis
```

## Testing locally

To test/debug this package locally, you can run exactly the job that
CircleCI runs for continuous integration by
[installing the CircleCI local CLI](https://circleci.com/docs/2.0/local-cli/#installing-the-circleci-local-cli-on-macos-and-linux-distros)
and invoking:

```bash
circleci build --job py27
```

See [.circleci/config.yml](https://github.com/mozilla/mozanalysis/blob/master/.circleci/config.yml)
for the other configured job names (for running tests on different python versions).

There is also a `bin/test` script that builds a docker image and
python environment (both of which are cached locally) and allows
you to run a subset of tests. Here are some sample invocations:

```bash
./bin/test tests/ -k <key>     # runs only tests with a given key
PYTHON_VERSION=2.7 ./bin/test  # specify a python version
```
