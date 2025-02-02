import argparse
import logging
import os
import sys

# Logging format borrowed from Fairseq.
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

from pathlib import Path

import comet
import h5py
import torch

from tqdm import tqdm
from transformers import GenerationConfig

from efficient_reranking.lib import datasets, generation, utils

MAX_GENERATION_LENGTH = 256


class MissingArgumentError(ValueError):
    pass


def main(args):
    torch.manual_seed(0)
    work_dir = Path(args.work_dir) / f"{args.src_lang}{args.tgt_lang}" / args.split
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(args.src_lang, args.tgt_lang, args.split)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_h5file = h5py.File(work_dir / (utils.DATA_FILENAME_BASE + ".h5"), 'a')

    logging.info(f"Evaluating candidates with COMET.")

    if args.comet_repo:
        comet_base_name = args.comet_repo.split("/")[-1]
        model_path = comet.download_model(args.comet_repo)
        model = comet.load_from_checkpoint(model_path).eval().to(device)
    elif args.comet_path:
        comet_base_name = os.path.splitext(args.comet_path.split("/")[-1])[0]
        model = comet.load_from_checkpoint(args.comet_path).eval().to(device)
    else:
        raise MissingArgumentError("Must provide --comet_repo or --comet_path.")

    comet_h5ds_name = utils.COMET_SCORES_H5DS_NAME_BASE + comet_base_name
    candidates_h5ds = data_h5file[utils.CANDIDATES_H5DS_NAME]
    all_score_h5ds = []

    for layer_idx in range(len(model.encoder.model.encoder.layer)+1):
        comet_h5ds_name = utils.COMET_SCORES_H5DS_NAME_BASE + comet_base_name + f"_{layer_idx}"
        if comet_h5ds_name in data_h5file:
            if args.overwrite:
                if layer_idx == 0:
                    logging.info(f"Dataset {comet_h5ds_name} exists but overwriting.")
            else:
                logging.info(f"Dataset {comet_h5ds_name} exists, aborting. Use --overwrite to overwrite.")
                return
            all_score_h5ds.append(data_h5file[comet_h5ds_name])
        else:
            all_score_h5ds.append(data_h5file.create_dataset(
                comet_h5ds_name,
                candidates_h5ds.shape,
                float))

    original_layerwise_attn_params = model.layerwise_attention.scalar_parameters

    for i in tqdm(range(candidates_h5ds.shape[0])):
        src = dataset[i]["src"]
        ref = dataset[i]["tgt"]
        tgts = [candidates_h5ds[i, j].decode() for j in range(candidates_h5ds.shape[1])]
        data = [ {"src": src, "mt": tgt} for tgt in tgts]

        with torch.no_grad():
            inputs = model.prepare_sample(data, stage="predict")[0]
            encoder_out = model.encoder(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
            for layer_idx in range(len(encoder_out["all_layers"])):
                encoder_out_subset = encoder_out["all_layers"][:layer_idx+1]
                # Hack the layerwise attention to work on fewer layers
                model.layerwise_attention.scalar_parameters = model.layerwise_attention.scalar_parameters[:layer_idx+1]
                model.layerwise_attention.num_layers = layer_idx + 1

                layerwise_out = model.layerwise_attention(encoder_out_subset, mask=encoder_out["attention_mask"])
                sentemb = layerwise_out[:, 0, :]
                ff_out = model.estimator(sentemb)
                all_score_h5ds[layer_idx][i] = ff_out.squeeze().cpu().numpy()

                # Change the layerwise attention back
                model.layerwise_attention.scalar_parameters = original_layerwise_attn_params
                model.layerwise_attention.num_layers = len(encoder_out["all_layers"])

    logging.info(f"Finished.")

    data_h5file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "src_lang",
        help="Source language pair. Supported languages are 'ende' and those supported by"
             " Facebook M2M100 (https://huggingface.co/facebook/m2m100_418M). 'ende',")

    parser.add_argument(
        "tgt_lang",
        help="Source language pair. Supported languages are 'ende' and those supported by"
             " Facebook M2M100 (https://huggingface.co/facebook/m2m100_418M). 'ende',")

    parser.add_argument(
        "split", type=str, help="Data split. Either 'validation' or 'test'.")

    parser.add_argument(
        "work_dir", help="Working directory for all steps. "
                         "Will be created if doesn't exist.")

    parser.add_argument(
        "--comet_repo", help="Huggingface COMET model name. Must pass --comet_repo or --comet_path")

    parser.add_argument(
        "--comet_path", help="COMET model directory. Must pass --comet_repo or --comet_path")

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for PyTorch.")

    args = parser.parse_args()
    main(args)
