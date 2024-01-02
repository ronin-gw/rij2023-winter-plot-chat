#!/bin/bash

python3 -m pip install -r requirements.txt

./01-download.sh
./02-detect-lostframe.py chat/*.gz > chat/lostframes.tab
./03-reload.sh
./04-merge-json.py chat/*.gz chat/retry/*

./05-plot.py messages.json.gz
mkdir normal
mv [1-4].png normal
./05-plot.py -n messages.json.gz
