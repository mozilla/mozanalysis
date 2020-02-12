# Mozilla Experiments Analysis [![CircleCI](https://circleci.com/gh/mozilla/mozanalysis.svg?style=svg)](https://circleci.com/gh/mozilla/mozanalysis) [![codecov](https://codecov.io/gh/mozilla/mozanalysis/branch/master/graph/badge.svg)](https://codecov.io/gh/mozilla/mozanalysis) [![CalVer - Timely Software Versioning](https://img.shields.io/badge/calver-YYYY.M.MINOR-22bfda.svg)](https://calver.org/)

The `mozanalysis` Python library is a library to standardize experiment analysis
at Mozilla for the purpose of producing decision reports templates that are
edited by data scientists.

## Documentation

Online documentation is available at https://mozilla.github.io/mozanalysis/

## Installing from pypi
- To install this package from pypi run:
```
pip install mozanalysis
```

## Testing locally

### with Tox

Install tox into your global Python environment and run `tox`.

You can pass flags to tox to limit the different environments you test in
or the tests you run. Options after `--` or positional arguments are forwarded to pytest.

For example, you can run:

* `tox -e lint` to lint
* `tox -e py37 -- -k utils` to only run tests with "utils" somewhere in the name, on Python 3.7
* `tox tests/test_utils.py` to run tests in a specific file

### with the CircleCI utilities

To test/debug this package locally, you can run exactly the job that
CircleCI runs for continuous integration by
[installing the CircleCI local CLI](https://circleci.com/docs/2.0/local-cli/#installing-the-circleci-local-cli-on-macos-and-linux-distros)
and invoking:

```bash
circleci build --job py37
```

See [.circleci/config.yml](https://github.com/mozilla/mozanalysis/blob/master/.circleci/config.yml)
for the other configured job names (for running tests on different python versions).

## Deploying a new release

Releasing mozanalysis happens by tagging a CalVer based Git tag with the
following pattern:

    YYYY.M.MINOR

where YYYY is the four-digit year number, M is a single-digit month number and
MINOR is a single-digit zero-based counter which does NOT relate to the day of
the release. Valid versions numbers are:

    2017.10.0
    2018.1.0
    2018.12.12

Once the (signed) Git tag has been pushed to the main GitHub repository using
git push origin --tags, Circle CI will automatically build and push a release to
PyPI after the tests have passed.
