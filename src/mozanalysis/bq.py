# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import re
from google.cloud import bigquery
from google.api_core.exceptions import NotFound


def sanitize_table_name_for_bq(table_name):
    of_good_character_but_possibly_verbose = re.sub(r'[^a-zA-Z_0-9]', '_', table_name)

    if len(of_good_character_but_possibly_verbose) <= 1024:
        return of_good_character_but_possibly_verbose

    return of_good_character_but_possibly_verbose[:500] + '___' \
        + of_good_character_but_possibly_verbose[-500:]


class BigQueryContext:
    """Holds a BigQuery client, and some configuration.

    Args:
        dataset_id (str): Your `BigQuery dataset id`_.
        project_id (str, optional): Your BigQuery project_id.
            Defaults to the DS team's project.

    .. _BigQuery dataset id: https://cloud.google.com/bigquery/docs/datasets
    """
    def __init__(self, dataset_id, project_id='moz-fx-data-bq-data-science'):
        self.dataset_id = dataset_id
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def run_script_or_fetch(self, sql, results_table):
        """Runs a BigQuery SQL script and returns a RowIterator for results_table.
        The script is assumed to create a table named results_table after completing
        succesfully. If results_table already exists, a RowIterator for the
        existing table will be returned without invoking the script.

        results_table is assumed to be an unqualified table name without
        a project or dataset reference.

        Learn more about BigQuery scripting at
        https://cloud.google.com/bigquery/docs/reference/standard-sql/scripting.
        """

        fqtn = self.fully_qualify_table_name(results_table)

        try:
            cached = self.client.list_rows(fqtn)
            print("Full results table already exists. Reusing", results_table)
            return cached
        except NotFound:
            pass

        self.client.query(sql).result()
        print('Saved into', results_table)
        return self.client.list_rows(fqtn)

    def fully_qualify_table_name(self, table_name):
        """Given a table name, return it fully qualified."""
        return "`{project_id}.{dataset_id}.{full_table_name}`".format(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            full_table_name=table_name,
        )
