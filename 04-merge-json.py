#!/usr/bin/env python3
import argparse
import gzip
import json
from itertools import groupby
import os.path


def _get_video_id(path):
    return os.path.basename(path).split("-")[0]


def _load_messages(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return {m["message_id"]: m for m in json.load(f)}


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    messages = {}
    for video_id, paths in groupby(sorted(args.files), _get_video_id):
        for path in paths:
            messages.update(_load_messages(path))

    messages = sorted(messages.values(), key=lambda m: m["timestamp"])
    with gzip.open("messages.json.gz", "wt") as f:
        json.dump(messages, f, indent=2)


if __name__ == "__main__":
    _main()
