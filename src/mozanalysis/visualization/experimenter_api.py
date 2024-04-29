import requests
from typing import List


class APIData:
    _API_URL = "https://experimenter.services.mozilla.com/api/v6/experiments"

    def __init__(self, experiment_slug: str) -> None:
        self.experiment_slug = experiment_slug
        self._api_response = None
        self._reference_branch = None
        self._branch_labels = None

        self._download_data()

    def reference_branch(self) -> str:
        if self._reference_branch is None:
            self._reference_branch = self._api_response["referenceBranch"]
        return self._reference_branch

    def branch_labels(self) -> List[str]:
        if self._branch_labels is None:
            branches = []
            for bdata in self._api_response["branches"]:
                branches.append(bdata["slug"])
            self._branch_labels = branches
        return self._branch_labels

    def _download_data(self) -> None:
        response = requests.get(f"{self._API_URL}/{self.experiment_slug}/")
        response.raise_for_status()
        self._api_response = response.json()
