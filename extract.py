import glob
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque
FEATURES = {
    "apigateway": 1,
    "cli": 2,
    "cloudformation": 2,
    "cloudwatch": 1,
    "dynamodb": 1,
    "elasticloadbalancing": 2,
    "ec2": 1,
    "ecs": 2,
    "eks": 1,
    "iam": 2,
    "lambda": 1,
    "rds": 1,
    "s3": 1,
    "sagemaker": 1,
    "vpc": 2,
    "xray": 1
}

feature_level_page_skip=False
depth_level_page_skip=True
BASE_DOMAIN = "docs.aws.amazon.com"
SCHEME = "https"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (DepthCrawler-RAG)"
}

def normalize_url(url):
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="", query="")
    return urlunparse(parsed)


def is_valid(url):
    return urlparse(url).netloc.endswith(BASE_DOMAIN)


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    if not main:
        return ""

    text = main.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.split("\n")]
    return "\n".join([l for l in lines if l])


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        url = normalize_url(urljoin(base_url, a["href"]))
        if is_valid(url):
            links.append(url)

    return links


def save(path, text, url):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Source\n{url}\n\n")
        f.write(text)

def crawl(start_url, feature_name, max_depth=2, max_pages=100, delay=0.5):
    visited = set()
    queue = deque([(normalize_url(start_url), 0)])
    os.makedirs("data/", exist_ok=True)
    print(f"Crawling {start_url} for {feature_name}:")
    page_id = 0
    downloaded = 0

    while queue and page_id < max_pages:
        url, depth = queue.popleft()

        if url in visited:
            continue

        if depth > max_depth:
            continue
        existing_files = glob.glob(f"data/{feature_name}*")
        if feature_level_page_skip and existing_files:
            print(f"Skipping {url} as files for {feature_name} already exist.")
            continue
        if depth_level_page_skip and max_depth <= 1 and existing_files:
            print(f"Skipping {feature_name} as its max depth is {max_depth}")
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"ERROR: {resp.status_code}: {url}")
                continue

            html = resp.text
            url_parts = urlparse(url)
            # print(f"Saving {url_parts}")
            url_path = url_parts.path.replace("/", "_")
            path = f"data/{feature_name}{url_path}.md"
            skip_download = False
            if os.path.exists(path):
                print(f"Skipping {url} as file already exists.")
                skip_download = True
                # continue
            else:
                print(f"[depth={depth}] {url}")
            text = extract_text(html)
            visited.add(url)

            if len(text) > 300:
                if not skip_download:
                    print(f"Saving {url} to {path}")
                    save(path, text, url)
                    downloaded += 1
                page_id += 1

            if depth <= max_depth:
                for link in extract_links(html, url):
                    if link not in visited:
                        queue.append((link, depth + 1))

            time.sleep(delay)

        except Exception as e:
           print(f"Error {url}: {e}")
    print(f"Downloading {downloaded} of {page_id} pages for {feature_name}")


def run():
    for f in FEATURES:
        crawl(f"{SCHEME}://{BASE_DOMAIN}/{f}", feature_name=f , max_depth=FEATURES.get(f), max_pages=80)

if __name__ == "__main__":
    run()
