#!/bin/bash

OUTDIR=chat/retry
mkdir -p $OUTDIR
readingid=""

NTHREADS=4

cat chat/lostframes.tab \
    | while read line; do
        args=($line)
        id=${args[0]}
        begin=${args[1]}
        end=${args[2]}

        if [ "$readingid" != "$id" ]; then
            i=1
            readingid=$id
        fi

        output=$OUTDIR/${id}-$(printf "%03d" $i).json
        if [ -f "$output.gz" ]; then
            i=$((i+1))
            continue
        fi
        echo "chat_downloader -o $output -s $begin -e $end https://www.twitch.tv/videos/${id} > /dev/null"
        i=$((i+1))
    done \
    | xargs -P $NTHREADS -I {} bash -c "{}"

pigz -p $NTHREADS $OUTDIR/*.json
