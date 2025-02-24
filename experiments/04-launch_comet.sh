function sbatch_gpu() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=1-0 \
        --wrap="$JOB_WRAP";
}

function sbatch_gpu_short() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:22g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=0-4 \
        --wrap="$JOB_WRAP";
}


function sbatch_gpu_big() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:40g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=1-0 \
        --wrap="$JOB_WRAP";
}


function sbatch_gpu_bigg() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:70g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=1-0 \
        --wrap="$JOB_WRAP";
}


function sbatch_gpu_big_short() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
        -J $JOB_NAME --output=logs/%x.out --error=logs/%x.err \
        --gpus=1 --gres=gpumem:40g \
        --ntasks-per-node=1 \
        --cpus-per-task=6 \
        --mem-per-cpu=8G --time=0-4 \
        --wrap="$JOB_WRAP";
}



sbatch_gpu_big  "train_helium"                    "comet-early-exit-train --cfg comet_early_exit/configs/experimental/baseline_model.yaml"
sbatch_gpu_big  "train_helium_fused"              "comet-early-exit-train --cfg comet_early_exit/configs/experimental/baseline_model.yaml"
sbatch_gpu_big  "train_helium_pinned"             "comet-early-exit-train --cfg comet_early_exit/configs/experimental/baseline_model.yaml"
sbatch_gpu_big  "train_hydrogen"                  "comet-early-exit-train --cfg comet_early_exit/configs/experimental/earlyexit_model.yaml"
sbatch_gpu_big  "train_lithium_confidence_human"  "comet-early-exit-train --cfg comet_early_exit/configs/experimental/earlyexitconf_model_human.yaml"
sbatch_gpu_big  "train_beryllium_confidence_last" "comet-early-exit-train --cfg comet_early_exit/configs/experimental/earlyexitconf_model_last.yaml"
sbatch_gpu_big  "train_boron"                     "comet-early-exit-train --cfg comet_early_exit/configs/experimental/baseline_model_boron.yaml"
sbatch_gpu_big  "train_carbon"                    "comet-early-exit-train --cfg comet_early_exit/configs/experimental/baseline_model_carbon.yaml"
sbatch_gpu_big  "train_nitrogen"                  "comet-early-exit-train --cfg comet_early_exit/configs/experimental/earlyexitconfmulti_model_last_old.yaml"
sbatch_gpu_bigg "train_oxygen"                    "comet-early-exit-train --cfg comet_early_exit/configs/experimental/earlyexitconfmulti_model_last.yaml"
sbatch_gpu_bigg "train_fluorine"                  "comet-early-exit-train --cfg comet_early_exit/configs/experimental/baseline_model_fluorine.yaml"
sbatch_gpu_bigg "train_neon"                      "comet-early-exit-train --cfg comet_early_exit/configs/experimental/baseline_model_neon.yaml"
sbatch_gpu_bigg "train_sodium"                    "comet-early-exit-train --cfg comet_early_exit/configs/experimental/baseline_model_sodium.yaml"
sbatch_gpu_bigg "train_magnesium"                 "comet-early-exit-train --cfg comet_early_exit/configs/experimental/instantconf_model.yaml"
sbatch_gpu_bigg "train_aluminium"                 "comet-early-exit-train --cfg comet_early_exit/configs/experimental/direct_uncertainty_prediction.yaml"
sbatch_gpu_big  "train_silicon"                   "comet-early-exit-train --cfg comet_early_exit/configs/experimental/earlyexitmulti_model.yaml"
sbatch_gpu_big  "train_sulfur"                    "comet-early-exit-train --cfg comet_early_exit/configs/experimental/instantconf_model_025.yaml"
sbatch_gpu_big  "train_chlorine"                  "comet-early-exit-train --cfg comet_early_exit/configs/experimental/instantconf_model_15.yaml"
sbatch_gpu_big  "train_argon"                     "comet-early-exit-train --cfg comet_early_exit/configs/experimental/instantconf_model_075.yaml"
sbatch_gpu_big  "train_phosphorus"                "comet-early-exit-train --cfg comet_early_exit/configs/experimental/instantconf_model_10.yaml"

# make sure to use ZervaCOMET
sbatch_gpu_big "train_potassium" "comet-train --cfg comet_early_exit/configs/models/baseline_hts.yaml"
sbatch_gpu_big "train_calcium"   "comet-train --cfg comet_early_exit/configs/models/baseline_hts2.yaml"


# remove all but last checkpoint
ls lightning_logs/version_*/checkpoints/epoch\={0,1,2,3}*.ckpt  
rm lightning_logs/version_*/checkpoints/epoch\={0,1,2,3}*.ckpt

tar -C lightning_logs/version_22504094/ -cf lightning_logs/model-helium.tar .
tar -C lightning_logs/version_22504382/ -cf lightning_logs/model-hydrogen.tar .
tar -C lightning_logs/version_22504384/ -cf lightning_logs/model-lithium.tar .
tar -C lightning_logs/version_22504386/ -cf lightning_logs/model-beryllium.tar .
tar -C lightning_logs/version_22525419/ -cf lightning_logs/model-boron.tar .
tar -C lightning_logs/helium2hydrogen/ -cf lightning_logs/model-helium2hydrogen.tar .
tar -C lightning_logs/version_22525421/ -cf lightning_logs/model-carbon.tar .
tar -C lightning_logs/version_22525435/ -cf lightning_logs/model-oxygen.tar .
tar -C lightning_logs/version_22525433/ -cf lightning_logs/model-nitrogen.tar .
tar -C lightning_logs/version_22525447/ -cf lightning_logs/model-magnesium.tar .

rclone copy lightning_logs/model-helium.tar polybox:t/
rclone copy lightning_logs/model-hydrogen.tar polybox:t/
rclone copy lightning_logs/model-lithium.tar polybox:t/
rclone copy lightning_logs/model-beryllium.tar polybox:t/
rclone copy lightning_logs/model-boron.tar polybox:t/
rclone copy lightning_logs/model-helium2hydrogen.tar polybox:t/
rclone copy lightning_logs/model-carbon.tar polybox:t/
rclone copy lightning_logs/model-oxygen.tar polybox:t/
rclone copy lightning_logs/model-nitrogen.tar polybox:t/
rclone copy lightning_logs/model-magnesium.tar polybox:t/