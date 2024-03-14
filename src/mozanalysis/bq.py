# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import logging
import re

from google.api_core.exceptions import Conflict
from google.cloud import bigquery

logger = logging.getLogger(__name__)


def sanitize_table_name_for_bq(table_name):
    of_good_character_but_possibly_verbose = re.sub(r"[^a-zA-Z_0-9]", "_", table_name)

    if len(of_good_character_but_possibly_verbose) <= 1024:
        return of_good_character_but_possibly_verbose

    return (
        of_good_character_but_possibly_verbose[:500]
        + "___"
        + of_good_character_but_possibly_verbose[-500:]
    )


class BigQueryContext:
    """Holds a BigQuery client, and some configuration.

    Args:
        dataset_id (str): Your `BigQuery dataset id`_.
        project_id (str, optional): Your BigQuery project_id.
            Defaults to the DS team's project.

    .. _BigQuery dataset id: https://cloud.google.com/bigquery/docs/datasets
    """

    def __init__(self, dataset_id, project_id="moz-fx-data-bq-data-science"):
        self.dataset_id = dataset_id
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def run_query(self, sql, results_table=None, replace_tables=False):
        """Run a query and return the result.

        If ``results_table`` is provided, then save the results
        into there (or just query from there if it already exists).
        Returns a ``google.cloud.bigquery.table.RowIterator``

        Args:
            sql (str): A SQL query.
            results_table (str, optional): A table name, not including
                a project_id or dataset_id. The table name is used as a
                cache key (if the table already exists, we ignore ``sql``
                and return the table's contents), so it is wise for
                ``results_table`` to include a hash of ``sql``.
            replace_tables (bool): Indicates if the results table should
                be replaced with new results, if that table is found.
        """
        if not results_table:
            return self.client.query(sql).result()

        if replace_tables:
            self.client.delete_table(
                self.fully_qualify_table_name(results_table),
                not_found_ok=True,
            )

        try:
            full_res = self.client.query(
                sql,
                job_config=bigquery.QueryJobConfig(
                    destination=self.client.dataset(self.dataset_id).table(
                        results_table
                    )
                ),
            ).result()
            logger.info("Saved into", results_table)
            return full_res

        except Conflict:
            logger.info("Table already exists. Reusing", results_table)
            return self.client.list_rows(self.fully_qualify_table_name(results_table))

    def fully_qualify_table_name(self, table_name):
        """Given a table name, return it fully qualified."""
        return f"{self.project_id}.{self.dataset_id}.{table_name}"
