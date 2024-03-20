# Tooling for Better Dependency Management

* Status: Pending
* Deciders: jsnyder, mikewilli
* Date: 2024-03-08


## Context and Problem Statement
Currently dependencies are very loosly defined in the `setup.py`. This can lead to issues when installing the package, an example of which is detailes [in this issue](https://github.com/mozilla/mozanalysis/issues/202).  We need to find a way to more clearly define dependencies in a way that will not be disruptive for developers and users.

## Decision Drivers
- Minimize the potential for incompatible dependencies
- Have guardrails/automation so that changes to dependencies made by developers will easily be reflected in the PRs they make
- Ideally no disruption to user workflow
- Minimal distruption to developer workflow

## Options

### 1. pip-tools to generate requirements.txt
Jetstream currently uses this process to manage dependencies.  It involves uses pip-tools to generate a requirements.txt and requirements.in file from the setup.py. [dependabot](https://github.blog/2020-06-01-keep-all-your-packages-up-to-date-with-dependabot/) is a built-in github tool that is used to periodically check for dependency updates and create pull requests to implement them.  [Here](https://github.com/mozilla/jetstream/pull/2017) is an example of such a pull request.

* : Dependabot already used in Jetstream
* +: Jetstream uses it already so there is an existing base of experties in the data team
* +: Dependencies can be automatically updated with a script like the [update_deps](https://github.com/mozilla/jetstream/blob/main/script/update_deps) script in Jetstream
* -: Still using pip's dependency resolver, which is fast but can have issues with incompatible versions.  This may be mitigated or even solved by the way pip-tools pins dependencies.
* -: Jetstream doesn't use `pyproject.toml` for configuring dependencies, though it does seem to be supported by `pip-tools`

### 2. poetry
(poetry)[https://python-poetry.org/] is a popular tool for dependency managment.  When starting development with a poetry project, one uses the `poetry install` command to create a virtual environment for the project and install the package and its dependencies.  New dependencies are added with the `poetry add` command.  poetry (and other tools) are configured in the `pyproject.toml`.

* : Can add Dependabot, which is alrady built-in to github tool that has additional uses like checking for security vulnerabilities
* +: Very reliable dependency resolution
* +: Uses `pyproject.toml`, which is the new standard and is nice for consolidating other config files
* +: Also acts as a virtualenv manager
* +: No changes to user workflow
* +: When a new dependency is added with `poetry add` the `poetry.lock` and `pyproject.toml` files are automatically updated so there minimal risk of a new dependency not being accounted for
* -: Dependency resolution is slow, when you add a new package it can take several minutes to figure it out
* -: Developers would have to modify their workflow to accomodate poetry's virtualenv management
* -: Not wildely use in the data team
* -: would need to update tox to use poetry to build and publish package

### 3. mamba/conda
(mamba)[https://mamba.readthedocs.io/en/latest/] is a fast, robust, and cross-platform package manager. It is fully compatible with conda, which is a common package manager in Python, and can be a drop-in replacement for conda. Mamba creates a virtual environment that is isolated from each other.

* +: Creates isolated virtual environment that encapsulates all the packages in its own designated space
* +: Can assign a specific Python version to the given virtual environment
* +: Dependency resoluion is fast and optimized for parallelism
* +: Uses an `environment.yml` to specify dependencies
* +: Can consume `pyproject.toml` when specifying packages

* -: low established use of mamba/conda tooling in existing workflows
* -: potential increase in storage size of a mozanalysis environment since each isolated enivornment enscapsulates its packages
* -: the isolated virtual environment comes with potential for confusion over system libraries vs conda libraries as developers move between projects
* -: the need to reconfigure scripts and learn new tooling could make the cost of switching high


## Decision Outcome

The pip-tools method wins out because it is familiar and simple.  It would be nice to use the PR that adds this functionality to move as much configuation as possible in to a `pyproject.toml` file.
