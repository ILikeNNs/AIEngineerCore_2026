from pathlib import Path
import nbformat
import os
import sys
import requests
from nbconvert import MarkdownExporter
from .crawler import FocusedCrawler
 

# --- CONFIGURATION & EXECUTION ---
# Example: Crawl a store to find ipynb user manuals inside a /en/stable/ folder
START_URL = "https://sagemaker.readthedocs.io/en/stable/"
PATH_REQUIRED = "/en/stable/_sources/v3-examples/"
SUFFIX_REQUIRED = ".ipynb"


def download_files(matched_urls: list[str], folder_name: str) -> None:
    """
    Downloads files from the scraped list of links.
    Args:
        matched_urls: list of URLs
        folder_name: the folder where the files should be downloaded
    """

    os.makedirs("../" + folder_name, exist_ok=True)
    for elem in matched_urls:
        save_path = os.path.join(folder_name, elem.split('/')[-1])
        print(f"Downloading notebook: {elem.split('/')[-1]}")
        file_response = requests.get(elem, timeout=10)
        with open("../" + save_path, 'wb') as f:
            f.write(file_response.content)


def ipynbtomd(path: str) -> None:
    """
    Transforms ipynb files into markdown files
    Args:
        path: local path to where the ipynb files are stored
    """
    
    dir_path = Path(path)
    ipynb_files = list(dir_path.glob('*.ipynb'))
    print(dir_path)
    md_exporter = MarkdownExporter()
    print('making a directory')
    os.makedirs("../knowledge-base", exist_ok=True)
    for elem in ipynb_files:
        print(elem)
        file_name = str(elem).split('\\')[-1].split('.')[0]

        with open(elem, "r", encoding="utf-8") as f:
            nb_content = nbformat.read(f, as_version=4)
        (body, resources) = md_exporter.from_notebook_node(nb_content)
        with open(f'../knowledge-base/{file_name}.md',"w", encoding="utf-8") as f:
            f.write(body)


def main():
    crawler = FocusedCrawler(base_url=START_URL, target_path=PATH_REQUIRED, target_suffix=SUFFIX_REQUIRED)
    crawler.crawl(START_URL)

    print("\n--- Extraction Results ---")
    for matched in sorted(crawler._matched_urls):
        print(matched)

    print("\n--- Downloading files ---")
    download_files(crawler._matched_urls, 'downloaded_files')
    print("\n--- Reformatting to Markdown format ---")
    ipynbtomd('../downloaded_files')


if __name__ == "__main__":
    main()
    
