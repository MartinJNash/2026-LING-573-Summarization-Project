"""
Wikipedia-backed lay-term lookup with disk caching.

Inspired by MedJEx (Kwon et al., EMNLP 2022): Wikipedia hyperlinks explain
concepts that need clarification for a general reader, making it a natural
source of lay-language descriptions for technical terms.
"""

import requests

class WikiDefinitionLookup:
    """
    Looks up lay-language descriptions for terms via the Wikipedia REST API.
    Results are cached in memory to avoid redundant API calls in same run.
    """

    _MAX_DESCRIPTION_LEN = 120
    _TIMEOUT = 5

    def __init__(self):
        self._cache: dict[str, str | None] = {}

    def _wikipedia_url(self, term: str) -> str:
        formatted = requests.utils.quote(term, safe="")
        return f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted}"

    def _fetch_wiki(self, term: str) -> str | None:
        url = self._wikipedia_url(term)
        try:
            resp = requests.get(
                url,
                timeout=self._TIMEOUT,
                headers={"User-Agent": "LING573-Summarization"},
            )
        except requests.RequestException:
            return None

        if resp.status_code != 200:
            return None

        data = resp.json()

        if data.get("type") == "disambiguation":
            return None

        description = data.get("description", "")
        if description and len(description) < self._MAX_DESCRIPTION_LEN:
            return description

        extract = data.get("extract", "")
        first_sentence = extract.split(". ")[0] if extract else ""
        return first_sentence or None

    def lookup(self, term: str) -> str | None:
        """
        Returns a short lay-language description of term, or None if not found.
        Tries exact match, then '{term} (medicine)' as fallback.
        """
        key = term.lower()
        if key in self._cache:
            return self._cache[key]

        result = self._fetch_wiki(term)
        if result is None:
            result = self._fetch_wiki(f"{term} (medicine)")

        self._cache[key] = result
        return result

