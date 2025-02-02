```
# Set this to wherever you generated the data
DATA_DIR=vilem/scripts/data
# Set this to whatever you like
WORK_DIR=output
MODEL_CLASS=skintle

# 1. Generate candidates
python efficient_reranking/scripts/generate_candidates.py $DATA_DIR dev $WORK_DIR

# 2A. Run COMET evals for differently sized models
python efficient_reranking/scripts/score_comet.py $DATA_DIR dev $WORK_DIR --comet_repo=Unbabel/wmt22-cometkiwi-da

for ckpt in $(ls comet_models/$MODEL_CLASS-*/model/*.ckpt); do
    python efficient_reranking/scripts/score_comet.py $DATA_DIR dev $WORK_DIR --comet_path=$ckpt
done

# 3. Fit multivariate gaussian metamodel
python efficient_reranking/scripts/run_mv_gaussian.py $DATA_DIR $WORK_DIR $MODEL_CLASS en-cs --zero_mean
```