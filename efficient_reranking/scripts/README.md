```
# Set this to wherever you generated the data
DATA_DIR=vilem/scripts/data
# Set this to whatever you like
WORK_DIR=output
MODEL_CLASS=riem

# 1. Generate candidates
python efficient_reranking/scripts/generate_candidates.py $DATA_DIR dev $WORK_DIR

# 2A. Run COMET evals for differently sized models
python efficient_reranking/scripts/score_comet.py $DATA_DIR dev $WORK_DIR --comet_repo=Unbabel/wmt22-cometkiwi-da

for MODEL_CLASS in {riem,skintle}; do
    for ckpt in $(ls comet_models/$MODEL_CLASS-*/model/*.ckpt); do
        python efficient_reranking/scripts/score_comet.py $DATA_DIR dev $WORK_DIR --comet_path=$ckpt
    done
done

for MODEL_CLASS in {riem,skintle}; do
    for ckpt in $(ls comet_models/$MODEL_CLASS-*/model/*.ckpt); do
        for seed in {0,1,2,3}; do
            python efficient_reranking/scripts/score_comet.py $DATA_DIR dev $WORK_DIR --comet_path=$ckpt --mc_dropout --seed=$seed
        done
    done
done

# 3. Fit multivariate gaussian metamodel
MODEL_CLASS=skintle
python efficient_reranking/scripts/run_mv_gaussian.py $DATA_DIR $WORK_DIR $MODEL_CLASS cs-en --zero_mean

# ?. Get embeddings
# COMET
python efficient_reranking/scripts/get_comet_embeddings.py $DATA_DIR dev $WORK_DIR --comet_path=comet_models/skintle-S/model/skintle-S-v1.ckpt
```