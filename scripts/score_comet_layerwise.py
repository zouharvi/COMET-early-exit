import argparse
import json
import logging

from pathlib import Path
from tqdm import tqdm

import comet
import h5py
import numpy as np
import torch

from lib import utils


class MissingArgumentError(ValueError):
    pass

def main(args):
    torch.manual_seed(args.seed)

    work_dir = Path(args.work_dir) / args.split
    work_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.comet_repo:
        comet_base_name = args.comet_repo.split("/")[-1]
        model_path = comet.download_model(args.comet_repo)
        model = comet.load_from_checkpoint(model_path).eval()
    elif args.comet_path:
        comet_base_name = args.comet_path.split("/")[-3]
        model = comet.load_from_checkpoint(args.comet_path)
    else:
        raise MissingArgumentError("Must provide --comet_repo or --comet_path.")

    model = model.to(device)

    if args.mc_dropout:
        output_path_base = work_dir / (utils.COMET_SCORES_FILENAME_BASE + comet_base_name + f"_dropout_{args.seed}.h5")
    else:
        output_path_base = work_dir / (utils.COMET_SCORES_FILENAME_BASE + comet_base_name + ".h5")
    output_path = output_path_base.with_suffix(".h5")

    if output_path.exists():
        raise ValueError(f"Output file {output_path} already exists.")

    # TODO (julius): This logs to stdout twice for some reason
    utils.configure_logger("score_comet.py", output_path_base.with_suffix(".log"))

    logging.info(f"Evaluating candidates with COMET...")

    candidates_path = work_dir / (utils.CANDIDATES_FILENAME + ".h5")
    confidences_path = work_dir /  (utils.CONFIDENCES_FILENAME_BASE + comet_base_name + ".h5")
    data_path = Path(args.data_dir) / "jsonl" / f"{args.split}.jsonl"
    data_lines = open(data_path).readlines()

    all_score_h5ds = []
    all_confidences_h5ds = []
    with (h5py.File(output_path, "a") as output_file,
          h5py.File(candidates_path) as candidates_file,
          h5py.File(confidences_path, "a") as confidences_file):
        candidates_text_h5ds = candidates_file[utils.CANDIDATES_TEXT_H5DS_NAME]
        for layer_idx in range(len(model.encoder.model.encoder.layer)+1):
            comet_h5ds_name = utils.COMET_SCORES_H5DS_NAME + comet_base_name + f"_{layer_idx}"
            confidences_h5ds_name = utils.COMET_CONFIDENCES_H5DS_NAME + comet_base_name + f"_{layer_idx}"
            all_score_h5ds.append(output_file.create_dataset(
                comet_h5ds_name,
                candidates_text_h5ds.shape,
                float))
            all_confidences_h5ds.append(confidences_file.create_dataset(
                confidences_h5ds_name,
                candidates_text_h5ds.shape,
                float))


 

        data_idxs = []
        inputs = []
        logging.info("Preparing inputs...")


        # print("CAREFUL LESS EXAMPLES")
        # candidates_text_h5ds = candidates_text_h5ds[0:50]
        # data_lines = data_lines[0:50]
        assert candidates_text_h5ds.shape[0] == len(data_lines)

        for i, data_line in enumerate(tqdm(data_lines)):

            data = json.loads(data_line)
            src = data["src"]
            tgts = [candidates_text_h5ds[i, j].decode() for j in range(candidates_text_h5ds.shape[1])]
            # Skip missing candidates
            if all(not tgt for tgt in tgts):
                continue
            data_idxs.append(i)
            
            if comet_base_name in utils.VILEM_MODELS:
                for tgt in tgts:
                    inputs.append({"src": src, "mt": tgt})
            elif comet_base_name == "wmt22-cometkiwi-da":
                src_with_all_preds = [{"src": src, "mt": tgt} for tgt in tgts]
                inputs.append(src_with_all_preds)
            else:
                raise NotImplementedError(f"Comet model {comet_base_name} not implemented.")

        if not inputs:
            return
        logging.info("Scoring...")

        with torch.no_grad():
            if comet_base_name in utils.NO_CONFIDENCE_MODELS :
            
                result = model.predict(samples=inputs, batch_size=args.comet_batch_size, mc_dropout=args.mc_dropout)
                num_layers = len(result.scores[0])
                assert num_layers == len(model.encoder.model.encoder.layer)+1
                scores = np.array(result.scores).reshape( -1,  candidates_text_h5ds.shape[1], num_layers) # instances x candidates x layers
                scores = scores.transpose(2, 0, 1) # layers x instances x candidates 

                for layer in range(num_layers):
                    for result_idx, data_idx in enumerate(data_idxs):
                        all_score_h5ds[layer][data_idx] = scores[layer][result_idx]

            elif comet_base_name in utils.CONFIDENCE_MODELS:
                result = model.predict(samples=inputs, batch_size=args.comet_batch_size, mc_dropout=args.mc_dropout)
                num_layers = len(result.scores[0])
                assert num_layers == len(model.encoder.model.encoder.layer)+1 == len(result.confidences[0])

                scores = np.array(result.scores).reshape( -1,  candidates_text_h5ds.shape[1], num_layers) # instances x candidates x layers
                scores = scores.transpose(2, 0, 1) # layers x instances x candidates 
                confidences = np.array(result.confidences).reshape( -1,  candidates_text_h5ds.shape[1], num_layers) # instances x candidates x layers
                confidences = confidences.transpose(2, 0, 1) # layers x instances x candidates 

                assert confidences.shape == scores.shape
                for layer in range(num_layers):
                    for result_idx, data_idx in enumerate(data_idxs):
                        all_score_h5ds[layer][data_idx] = scores[layer][result_idx]
                        all_confidences_h5ds[layer][data_idx] = confidences[layer][result_idx]


            elif comet_base_name == "wmt22-cometkiwi-da":
                original_layerwise_attn_params = model.layerwise_attention.scalar_parameters
                
                for idx, example in tqdm(enumerate(inputs), total=len(inputs)):
                    example = model.prepare_sample(example, stage="predict")[0]
                    
                    encoder_out = model.encoder(example["input_ids"].to(device), example["attention_mask"].to(device))
                    for layer_idx in range(len(encoder_out["all_layers"])):

                        encoder_out_subset = encoder_out["all_layers"][:layer_idx+1]
                        # Hack the layerwise attention to work on fewer layers
                        model.layerwise_attention.scalar_parameters = model.layerwise_attention.scalar_parameters[:layer_idx+1]
                        model.layerwise_attention.num_layers = layer_idx + 1

                        layerwise_out = model.layerwise_attention(encoder_out_subset, mask=encoder_out["attention_mask"])
                        sentemb = layerwise_out[:, 0, :]
                        ff_out = model.estimator(sentemb)

                        all_score_h5ds[layer_idx][idx] = ff_out.squeeze().cpu().numpy()

                        # Change the layerwise attention back
                        model.layerwise_attention.scalar_parameters = original_layerwise_attn_params
                        model.layerwise_attention.num_layers = len(encoder_out["all_layers"])

            else:
                raise NotImplementedError(f"Comet model {comet_base_name} not implemented.")

    logging.info(f"Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", help="Data directory generated by the pipeline from vilem/scripts.")

    parser.add_argument(
        "split", type=str, help="Data split. Either 'dev' or 'test'.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "--comet_repo", help="Huggingface COMET model name. Must pass --comet_repo or --comet_path")

    parser.add_argument(
        "--comet_path", help="COMET model directory. Must pass --comet_repo or --comet_path")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for PyTorch.")

    parser.add_argument(
        "--comet_batch_size", type=int, default=128, help="COMET batch size.")

    parser.add_argument(
        "--mc_dropout", action="store_true", help="Activate MC dropout.")


    args = parser.parse_args()
    main(args)

# python efficient_reranking/scripts/score_comet_layerwise.py vilem/scripts/data toy output --comet_repo=Unbabel/wmt22-cometkiwi-da
# python efficient_reranking/scripts/score_comet_layerwise.py vilem/scripts/data toy output --comet_path=models-hydrogen/checkpoints/model.ckpt
# python efficient_reranking/scripts/score_comet_layerwise.py vilem/scripts/data toy output --comet_path=models-lithium/checkpoints/model.ckpt
# python efficient_reranking/scripts/score_comet_layerwise.py vilem/scripts/data toy output --comet_path=models-beryllium/checkpoints/model.ckpt
# python efficient_reranking/scripts/score_comet_layerwise.py vilem/scripts/data toy output --comet_path=models-helium/checkpoints/model.ckpt

# python efficient_reranking/scripts/score_comet_layerwise.py vilem/scripts/data dev output --comet_repo=Unbabel/wmt22-cometkiwi-da
# python efficient_reranking/scripts/score_comet_layerwise.py vilem/scripts/data dev output --comet_path=models-hydrogen/checkpoints/model.ckpt
