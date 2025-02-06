#!/usr/bin/bash


rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-early-exit-experiments/
# rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/COMET-early-exit/

# rsync -azP data/csv/ euler:/cluster/work/sachan/vilem/COMET-early-exit/data/csv
# rsync -azP data/jsonl/ euler:/cluster/work/sachan/vilem/COMET-early-exit/data/jsonl
# rsync -azP euler:/cluster/work/sachan/vilem/COMET-early-exit/lightning_logs/version_19245798/ lightning_logs/version_19245798/

# scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_beryllium_conf.out computed/eval_beryllium_conf.json
# scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_beryllium.out computed/eval_beryllium.json
# scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_helium2hydrogen.out computed/eval_helium2hydrogen.json
# scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_lithium.out computed/eval_lithium.json
# scp euler:/cluster/work/sachan/vilem/COMET-early-exit/logs/eval_lithium_conf.out computed/eval_lithium_conf.json
