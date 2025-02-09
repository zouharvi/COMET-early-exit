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



sbatch_gpu_big "train_helium" "comet-train --cfg configs/experimental/baseline_model.yaml"
sbatch_gpu_big "train_hydrogen" "comet-train --cfg configs/experimental/earlyexit_model.yaml"
sbatch_gpu_big "train_lithium_confidence_human" "comet-train --cfg configs/experimental/earlyexitconf_model_human.yaml"
sbatch_gpu_big "train_beryllium_confidence_last" "comet-train --cfg configs/experimental/earlyexitconf_model_last.yaml"
sbatch_gpu_big "train_boron" "comet-train --cfg configs/experimental/baseline_model_boron.yaml"
sbatch_gpu_big "train_carbon" "comet-train --cfg configs/experimental/baseline_model_carbon.yaml"
sbatch_gpu_big "train_nitrogen" "comet-train --cfg configs/experimental/earlyexitconfmulti_model_last_old.yaml"
sbatch_gpu_bigg "train_oxygen" "comet-train --cfg configs/experimental/earlyexitconfmulti_model_last.yaml"
sbatch_gpu_bigg "train_fluorine" "comet-train --cfg configs/experimental/baseline_model_fluorine.yaml"
sbatch_gpu_bigg "train_neon" "comet-train --cfg configs/experimental/baseline_model_neon.yaml"
sbatch_gpu_bigg "train_sodium" "comet-train --cfg configs/experimental/baseline_model_sodium.yaml"
sbatch_gpu_bigg "train_magnesium" "comet-train --cfg configs/experimental/instantconf_model.yaml"
sbatch_gpu_short "train_aluminium" "comet-train --cfg configs/experimental/direct_uncertainty_prediction.yaml"


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

rclone copy lightning_logs/model-helium.tar polybox:t/
rclone copy lightning_logs/model-hydrogen.tar polybox:t/
rclone copy lightning_logs/model-lithium.tar polybox:t/
rclone copy lightning_logs/model-beryllium.tar polybox:t/
rclone copy lightning_logs/model-boron.tar polybox:t/
rclone copy lightning_logs/model-helium2hydrogen.tar polybox:t/
rclone copy lightning_logs/model-carbon.tar polybox:t/
rclone copy lightning_logs/model-oxygen.tar polybox:t/