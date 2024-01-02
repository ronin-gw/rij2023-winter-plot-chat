#!/bin/bash

OUTDIR=chat

mkdir -p $OUTDIR
for i in 2013919301 2014302846 2015229349 2016984270; do
    json=$OUTDIR/${i}.json
    if [ ! -f "$json.gz" ]; then
        chat_downloader -o $json https://www.twitch.tv/videos/${i} > /dev/null &
    fi
done
wait
pigz -p 4 $OUTDIR/*.json
