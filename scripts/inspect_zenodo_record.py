#!/usr/bin/env python
import json
import sys
import urllib.request


def main():
    record = sys.argv[1] if len(sys.argv) > 1 else "11060089"
    url = f"https://zenodo.org/api/records/{record}"
    with urllib.request.urlopen(url, timeout=60) as r:
        meta = json.load(r)
    print(meta.get("title"))
    for f in meta.get("files", []):
        print(f["key"], f.get("size"), f["links"].get("self"))


if __name__ == "__main__":
    main()
