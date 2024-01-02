#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import Counter
import os.path

import numpy as np

WINDOWSIZE = 10
THRESHOLD = 3
GROUPINGRANGE = 60


def _count_message(filename):
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rt") as f:
        return Counter(message["time_text"] for message in json.load(f))


def _parse_time(s):
    t = 0
    for part in s.split(":"):
        t = t * 60 + int(part)
    return t


def _time2str(t):
    s = f"{t // 60 % 60:02}:{t % 60:02}"
    if t >= 3600:
        s = f"{t // 3600}:{s}"
    else:
        s = s.replace("00:", "0:")
    return s


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    for path in args.files:
        name = os.path.basename(path).split(".")[0]
        counts = _count_message(path)
        last = sorted(counts.keys(), key=_parse_time)[-1]

        counts = np.array([counts[_time2str(t)] for t in range(0, _parse_time(last) + 1)])
        # moving average
        average = np.convolve(counts, np.ones(WINDOWSIZE) / WINDOWSIZE, mode="same")
        lostframe = (average >= THRESHOLD) * (counts == 0)

        # for i, (count, ave, is_lost) in enumerate(zip(counts, average, lostframe)):
        #     print(name, _time2str(i), count, f"{ave:.3}", is_lost, sep="\t")

        # last = None
        # for i in range(len(lostframe)):
        #     if not lostframe[i]:
        #         continue
        #     if last is not None:
        #         print(i - last)
        #     last = i
        # continue

        ranges = []
        begin = last = None
        for i in range(len(lostframe)):
            if not lostframe[i]:
                continue

            if last is None:
                begin = last = i
            elif i - last > GROUPINGRANGE:
                ranges.append((begin, last))
                begin = last = i
            else:
                last = i

        if last is not None:
            ranges.append((begin, last))

        for begin, last in ranges:
            print(name, _time2str(begin), _time2str(last), sep="\t")


if __name__ == "__main__":
    _main()
