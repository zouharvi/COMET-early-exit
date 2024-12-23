#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/comet-early-exit

# rsync -azP data/csv/ euler:/cluster/work/sachan/vilem/comet-early-exit/data/csv