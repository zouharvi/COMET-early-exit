import argparse
import json
import logging
import sys

# Logging format borrowed from Fairseq.
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

from pathlib import Path

import h5py
import torch

from tqdm import tqdm
from transformers import GenerationConfig, M2M100ForConditionalGeneration, AutoTokenizer

from efficient_reranking.lib import datasets, generation, utils

MAX_GENERATION_LENGTH = 256
NUM_CANDIDATES = 128
# For candidate generation with beam search
CANDIDATE_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_beams=NUM_CANDIDATES,
    num_return_sequences=NUM_CANDIDATES,
    early_stopping=True
)


def main(args):
    torch.manual_seed(0)
    work_dir = Path(args.work_dir) / args.split
    work_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Generating candidates...")

    data_path = Path(args.data_dir) / "jsonl" / f"{args.split}.jsonl"
    data_lines = open(data_path).readlines()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M2M100ForConditionalGeneration.from_pretrained(utils.NLLB_MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(utils.NLLB_MODEL_NAME)

    output_path = work_dir / (utils.CANDIDATES_FILENAME + ".h5")
    if output_path.exists():
        if args.overwrite:
            logging.info(f"Output file {output_path} exists but overwriting.")
        else:
            logging.info(f"Output file {output_path} exists, only doing unfinished instances. Use --overwrite to overwrite.")

    with h5py.File(output_path, "a") as output_file:
        # Fetch or create h5 datasets
        if utils.CANDIDATES_TEXT_H5DS_NAME in output_file:
            text_h5ds = output_file[utils.CANDIDATES_TEXT_H5DS_NAME]
            token_logprobs_h5 = output_file[utils.CANDIDATES_TOKEN_LOGPROBS_H5DS_NAME]
        else:
            text_h5ds = output_file.create_dataset(
                utils.CANDIDATES_TEXT_H5DS_NAME,
                (len(data_lines), NUM_CANDIDATES),
                utils.H5_STRING_DTYPE
            )
            token_logprobs_h5 = output_file.create_dataset(
                utils.CANDIDATES_TOKEN_LOGPROBS_H5DS_NAME,
                (len(data_lines), NUM_CANDIDATES),
                utils.H5_VLEN_FLOAT_DTYPE
            )

        for i, data_line in enumerate(tqdm(data_lines)):
            if text_h5ds[i][0] and not args.overwrite:
                continue
            data = json.loads(data_line)
            src_lang, tgt_lang = data["langs"].split("-")
            # The way the tokenizer src_lang is set and tgt_lang is set during generate()
            # is specific to NLLB.
            tokenizer.src_lang = utils.LANG_TO_NLLB[src_lang]
            tgt_lang_token = tokenizer.convert_tokens_to_ids(utils.LANG_TO_NLLB[tgt_lang])
            inputs = tokenizer(data["src"], padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                try:
                    result = model.generate(
                        **inputs,
                        generation_config=CANDIDATE_GENERATION_CONFIG,
                        forced_bos_token_id=tgt_lang_token,
                        output_scores=True,
                        return_dict_in_generate=True)
                except torch.OutOfMemoryError:
                    logging.info(f"Instance {i} failed with out-of-memory error. Skipping.")
                    continue

            texts = tokenizer.batch_decode(result.sequences, skip_special_tokens=True)

            # Get token log probabilities for beam search candidates.
            # NOTE (julius) compute_transition_scores() is slow and was causing OOM
            # errors so wrote a custom version.
            # logprobs = model.compute_transition_scores(
            #     result.sequences,
            #     [scores for scores in result.scores],
            #     beam_indices=result.beam_indices)[:, 1:]
            logprobs = torch.zeros_like(result.sequences[:, 2:], dtype=result.scores[0].dtype)
            for t in range(2, result.sequences.shape[1]):
                beam_scores =  result.scores[t-1].index_select(0, result.beam_indices[:, t-1].clamp(min=0))
                seq_scores = beam_scores.gather(1, result.sequences[:, t:t+1]).squeeze()
                logprobs[:, t-2] = seq_scores

            for j, text in enumerate(texts):
                text_h5ds[i, j] = text
                seq_length = (result.sequences[j] != tokenizer.pad_token_id).sum() - 2
                token_logprobs_h5[i, j] = logprobs[j][:seq_length].cpu().numpy()

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
        "--overwrite", action="store_true", help="Overwrite existing data.")

    parser.add_argument(
        "--subset", type=int, help="Only process the first n items.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for PyTorch.")

    args = parser.parse_args()
    main(args)
