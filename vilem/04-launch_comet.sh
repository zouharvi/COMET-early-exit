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



sbatch_gpu_big "train_helium" "comet-train --cfg configs/experimental/baseline_model.yaml"
sbatch_gpu_big "train_hydrogen" "comet-train --cfg configs/experimental/earlyexit_model.yaml"
sbatch_gpu_big "train_lithium_confidence_human" "comet-train --cfg configs/experimental/earlyexitconf_model_human.yaml"
sbatch_gpu_big "train_beryllium_confidence_last" "comet-train --cfg configs/experimental/earlyexitconf_model_last.yaml"
sbatch_gpu_big "train_boron" "comet-train --cfg configs/experimental/partial_model.yaml"


# remove all but last checkpoint
ls lightning_logs-tmp/version_*/checkpoints/epoch\={0,1,2,3}*.ckpt
rm lightning_logs-tmp/version_*/checkpoints/epoch\={0,1,2,3}*.ckpt

tar -C lightning_logs-tmp/version_22504094/ -cf lightning_logs-tmp/model-helium.tar .
tar -C lightning_logs-tmp/version_22504382/ -cf lightning_logs-tmp/model-hydrogen.tar .
tar -C lightning_logs-tmp/version_22504384/ -cf lightning_logs-tmp/model-lithium.tar .
tar -C lightning_logs-tmp/version_22504386/ -cf lightning_logs-tmp/model-beryllium.tar .
tar -C lightning_logs-tmp/version_22525419/ -cf lightning_logs-tmp/model-boron.tar .


rclone copy lightning_logs-tmp/model-helium.tar polybox:t/
rclone copy lightning_logs-tmp/model-hydrogen.tar polybox:t/
rclone copy lightning_logs-tmp/model-lithium.tar polybox:t/
rclone copy lightning_logs-tmp/model-beryllium.tar polybox:t/
rclone copy lightning_logs-tmp/model-boron.tar polybox:t/