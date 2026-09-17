"""Quick script to list MERL Shopping Dataset files and download one sample."""
import urllib.request
import re
import os

url = "https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset/"
print(f"Fetching: {url}")
resp = urllib.request.urlopen(url, timeout=15)
content = resp.read().decode("utf-8")

# Find all links
all_links = re.findall(r'href="([^"]+)"', content)
print(f"Total links found: {len(all_links)}")
for link in all_links:
    if link.startswith("?") or link == "/pub/tmarks/":
        continue
    print(f"  {link}")
