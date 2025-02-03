#!/usr/bin/bash


rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-early-exit-experiments/
# rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-early-exit/

# rsync -azP data/csv/ euler:/cluster/work/sachan/vilem/comet-early-exit/data/csv
# rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/comet-early-exit
# rsync -azP euler:/cluster/work/sachan/vilem/comet-early-exit/lightning_logs/version_19245798/ lightning_logs/version_19245798/

# rsync -azP data/csv/*_partial.csv euler:/cluster/work/sachan/vilem/COMET-early-exit/data/csv/