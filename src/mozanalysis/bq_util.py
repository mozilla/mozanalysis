# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import re
from google.cloud import bigquery
from google.api_core.exceptions import Conflict


def sanitize_table_name_for_bq(table_name):
    of_good_character_but_possibly_verbose = re.sub(r'[^a-zA-Z_0-9]', '_', table_name)

    if len(of_good_character_but_possibly_verbose) <= 1024:
        return of_good_character_but_possibly_verbose

    return of_good_character_but_possibly_verbose[:500] + '___' \
        + of_good_character_but_possibly_verbose[-500:]


class BigqueryStuff(object):
    def __init__(self, dataset_id, project_id='moz-fx-data-bq-data-science'):
        self.dataset_id = dataset_id
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def run_query(self, sql, results_table=None):
        """Run a query and return the result.

        If ``results_table`` is provided, then save the results
        into there (or just query from there if it already exists).
        """
        if not results_table:
            return self.client.query(sql).result()

        try:
            full_res = self.client.query(
                sql,
                job_config=bigquery.QueryJobConfig(
                    destination=self.client.dataset(
                        self.dataset_id
                    ).table(results_table)
                )
            ).result()
            print('Saved into', results_table)
            return full_res

        except Conflict:
            print("Full results table already exists. Reusing", results_table)
            return self.client.query(
                "SELECT * FROM `{project_id}.{dataset_id}.{full_table_name}`".format(
                    project_id=self.project_id,
                    dataset_id=self.dataset_id,
                    full_table_name=results_table,
                )
            ).result()
