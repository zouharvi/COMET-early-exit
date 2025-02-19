#!/usr/bin/bash


rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-early-exit-experiments/
# rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-early-exit/
# rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-zerva/

# rsync -azP data/jsonl/ euler:/cluster/work/sachan/vilem/COMET-early-exit/data/jsonl
# rsync -azP euler:/cluster/work/sachan/vilem/COMET-early-exit/lightning_logs/version_19245798/ lightning_logs/version_19245798/
# rsync -azP data/candidates/ euler:/cluster/work/sachan/vilem/COMET-early-exit-experiments/data/candidates/