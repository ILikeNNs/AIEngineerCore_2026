from urllib.parse import urljoin, urlparse
import requests
from pydantic import BaseModel, PrivateAttr, computed_field
from bs4 import BeautifulSoup

class FocusedCrawler(BaseModel):

    base_url: str
    target_path: str
    target_suffix: str
    _visited_urls = PrivateAttr(default=set())
    _matched_urls = PrivateAttr(default=set())

    @computed_field()
    def _domain(self) -> str:
        return urlparse(self.base_url).netloc


    def is_valid_internal_url(self, url: str) -> bool:
        """Ensure the URL stays within the target website domain.
        Args:
            url: the provided url
        Returns:
            boolean: checking if the provided url stays within the target domain
        """

        parsed = urlparse(url)
        return parsed.netloc == self._domain


    def matches_criteria(self, url: str) -> tuple[bool, bool]:
        """Check if the URL contains the target path and ends with the suffix.
        Args:
            url: the provided url
        Returns:
            boolean: checking if the provided url contains the target path
            boolean: checking if the provided url ends with the provided suffix
        """

        parsed = urlparse(url)
        has_path = self.target_path in parsed.path
        has_suffix = parsed.path.endswith(self.target_suffix)
        return has_path and has_suffix


    def crawl(self, current_url: str):
        """Recursively crawl pages and extract matching links.
        Args:
            current_url: the current url being crawled
        """
        if current_url in self._visited_urls:
            return

        print(f"Scanning: {current_url}")
        self._visited_urls.add(current_url)

        try:
            response = requests.get(current_url, timeout=5)
            # Only parse HTML content
            if "text/html" not in response.headers.get("Content-Type", ""):
                return

            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a", href=True):
                # Convert relative links to absolute links
                absolute_url = urljoin(current_url, link["href"])

                # Clean fragments (e.g., ://website.com -> ://website.com)
                clean_url = absolute_url.split("#")[0]

                if self.is_valid_internal_url(clean_url):
                    # Save it if it matches your specific criteria
                    if self.matches_criteria(clean_url):
                        self._matched_urls.add(clean_url)

                    # Continue crawling if we haven't visited this internal page yet
                    if clean_url not in self._visited_urls:
                        self.crawl(clean_url)

        except requests.RequestException as e:
            print(f"Failed to fetch {current_url}: {e}")
