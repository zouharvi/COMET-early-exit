#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/comet-early-exit

# rsync -azP data/csv/ euler:/cluster/work/sachan/vilem/comet-early-exit/data/csv

# rsync -azP euler:/cluster/work/sachan/vilem/comet-early-exit/lightning_logs/version_19245793/ lightning_logs/version_19245793/