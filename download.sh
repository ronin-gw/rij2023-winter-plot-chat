#!/bin/bash

python3 -m pip install -r requirements.txt

for i in 2013919301 2014302846 2015229349 2016984270; do
    json=${i}.json
    if [ ! -f "$json" ]; then
        chat_downloader -o $json https://www.twitch.tv/videos/${i} > /dev/null
    fi
done

./main.py -n *.json*
