import argparse
import itertools
import json
import logging

from pathlib import Path

import h5py
import torch

from tqdm import tqdm
from transformers import GenerationConfig, M2M100ForConditionalGeneration, AutoTokenizer

from lib import utils

MAX_GENERATION_LENGTH = 256


def get_generation_config(mode, num_candidates, epsilon_cutoff=0.05):
    if mode == "beam":
        return GenerationConfig(
            max_length=MAX_GENERATION_LENGTH,
            num_beams=num_candidates,
            num_return_sequences=num_candidates,
            early_stopping=True
        )
    else:
        return GenerationConfig(
            max_length=MAX_GENERATION_LENGTH,
            num_return_sequences=num_candidates,
            epsilon_cutoff=epsilon_cutoff,
            do_sample=True
        )


def process_result(output, tokenizer, generation_mode):
    """Process generation output to extract the data to save: text, token logprobs, and embeddings."""

    texts = tokenizer.batch_decode(output, skip_special_tokens=True)
    # texts = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    # logprobs = torch.zeros_like(output.sequences[:, 2:], dtype=output.scores[0].dtype)
    # decoder_embeddings = []
    # for t in range(2, output.sequences.shape[1]):
    #     if generation_mode == "beam":
    #         # Get token log probabilities for beam search candidates.
    #         # NOTE (julius) compute_transition_scores() is slow and was causing OOM
    #         # errors so wrote a custom version.
    #         # logprobs = model.compute_transition_scores(
    #         #     result.sequences,
    #         #     [scores for scores in result.scores],
    #         #     beam_indices=result.beam_indices)[:, 1:]
    #             beam_scores = output.scores[t-1].index_select(0, output.beam_indices[:, t-1].clamp(min=0))
    #             seq_scores = beam_scores.gather(1, output.sequences[:, t:t+1]).squeeze()
    #             logprobs[:, t-2] = seq_scores
    #     else:
    #         scores = output.scores[t-1].log_softmax(dim=-1)
    #         logprobs[:, t-2] = scores.gather(1, output.sequences[:, t:t+1]).squeeze()

    #     decoder_embeddings.append(output.decoder_hidden_states[t-1][-1].squeeze())

    # logprobs_list = []
    # for i in range(output.sequences.shape[0]):
    #     seq_length = (output.sequences[i] != tokenizer.pad_token_id).sum() - 2
    #     logprobs_list.append(logprobs[i][:seq_length].cpu().numpy())

    # decoder_embeddings = torch.stack(decoder_embeddings, dim=1)
    # mask = (output.sequences[:, 2:] != tokenizer.pad_token_id).unsqueeze(-1)
    # decoder_embeddings = (decoder_embeddings * mask).sum(dim=1) / mask.sum(1)

    return texts, [None] * len(texts), [None] * len(texts) #logprobs_list, decoder_embeddings


def main(args):
    torch.manual_seed(0)
    work_dir = Path(args.work_dir) / args.split
    work_dir.mkdir(parents=True, exist_ok=True)

    output_path_base = work_dir / (f"{utils.CANDIDATES_FILENAME}_{args.generation_mode}")
    output_path = output_path_base.with_suffix(".h5")
    utils.configure_logger("generate_candidates.py", output_path_base.with_suffix(".log"))

    data_path = Path(args.data_dir) / "jsonl" / f"{args.split}.jsonl"
    data_lines = open(data_path).readlines()


    if output_path.exists() and not args.overwrite:
        raise ValueError(f"Output file {output_path} already exists.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M2M100ForConditionalGeneration.from_pretrained(utils.NLLB_MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(utils.NLLB_MODEL_NAME)

    output_file = h5py.File(output_path, "w")
    # Create h5 datasets
    text_h5ds = output_file.create_dataset(
        utils.CANDIDATES_TEXT_H5DS_NAME,
        (len(data_lines), args.num_candidates),
        utils.H5_STRING_DTYPE
    )
    # token_logprobs_h5ds = output_file.create_dataset(
    #     utils.CANDIDATES_TOKEN_LOGPROBS_H5DS_NAME,
    #     (len(data_lines), args.num_candidates),
    #     utils.H5_VLEN_FLOAT_DTYPE
    # )
    # counts_h5ds = output_file.create_dataset(
    #     utils.CANDIDATES_COUNTS_H5DS_NAME,
    #     (len(data_lines), args.num_candidates),
    #     float
    # )
    # emb_size = model.model.decoder.layers[-1].final_layer_norm.weight.shape[0]
    # embeddings_h5ds = output_file.create_dataset(
    #     utils.EMBEDDINGS_H5DS_NAME,
    #     (len(data_lines), args.num_candidates, emb_size),
    #     float
    # )

    gen_config = get_generation_config(
        args.generation_mode, args.num_candidates, epsilon_cutoff=args.epsilon)

    logging.info(f"Generating candidates...")

    for i, data_line in enumerate(tqdm(data_lines[:args.subset])):
        # if i < 1299:
        #     continue
        data = json.loads(data_line)
        src_lang, tgt_lang = data["langs"].split("-")
        # The way the tokenizer src_lang is set and tgt_lang is set during generate()
        # is specific to NLLB.
        tokenizer.src_lang = utils.LANG_TO_NLLB[src_lang]
        tgt_lang_token = tokenizer.convert_tokens_to_ids(utils.LANG_TO_NLLB[tgt_lang])
        inputs = tokenizer(data["src"], padding=True, return_tensors="pt").to(model.device)
        all_result_data = []
        if args.generation_mode == "beam":
            try:
                with torch.no_grad():
                    result = model.generate(
                        **inputs,
                        generation_config=gen_config,
                        forced_bos_token_id=tgt_lang_token,
                        output_scores=False,
                        output_hidden_states=False,
                        return_dict_in_generate=False)
                result_data = process_result(result, tokenizer, args.generation_mode)
                all_result_data.append(result_data)
            except torch.OutOfMemoryError:
                logging.info(f"Instance {i} failed with out-of-memory error. Skipping.")
                continue
        elif args.generation_mode == "sample":
            with torch.no_grad():
                encoder_outputs = model.model.encoder(**inputs)
            max_batch_size = args.max_batch_size or args.num_candidates
            num_samples_done = 0
            while num_samples_done < args.num_candidates:
                try:
                    batch_size = min(max_batch_size, args.num_candidates - num_samples_done)
                    gen_config = get_generation_config(
                        args.generation_mode,
                        batch_size,
                        epsilon_cutoff=args.epsilon)
                    with torch.no_grad():
                        result = model.generate(
                            encoder_outputs=encoder_outputs.copy(),
                            attention_mask=inputs["attention_mask"],
                            generation_config=gen_config,
                            forced_bos_token_id=tgt_lang_token,
                            output_scores=False,
                            output_hidden_states=False,
                            return_dict_in_generate=False)
                        result_data = process_result(result, tokenizer, args.generation_mode)
                        all_result_data.append(result_data)
                    num_samples_done += batch_size
                except torch.OutOfMemoryError:
                    new_max_batch_size = max_batch_size // 2 + (max_batch_size % 2 > 0)
                    logging.info(
                        f"Instance {i} failed with out-of-memory error. Reducing batch "
                        f"size from {max_batch_size} to {new_max_batch_size}.")
                    max_batch_size = new_max_batch_size

        text_to_idx = {}

        for text, token_logprobs, embedding in zip(*[itertools.chain(*x) for x in zip(*all_result_data)]):
            if text not in text_to_idx:
                text_idx = len(text_to_idx)
                text_to_idx[text] = text_idx

                text_h5ds[i, text_idx] = text
                # token_logprobs_h5ds[i, text_idx] = token_logprobs
                # embeddings_h5ds[i, text_idx] = embedding.cpu().numpy()
            else:
                text_idx = text_to_idx[text]

            #counts_h5ds[i, text_idx] += 1


    output_file.close()
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
        "--subset", type=int, help="Only process the first n items.")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for PyTorch.")

    parser.add_argument(
        "--generation_mode", type=str, default="sample", help="Either 'beam' or 'sample'.")

    parser.add_argument(
        "--num_candidates", type=int, default=200, help="Number of candidates to generate.")

    parser.add_argument(
        "--epsilon", type=float, default=0.05, help="Threshold for epsilon sampling.")

    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite existing data.")

    parser.add_argument(
        "--max_batch_size", type=int, help="Max batch size used during sampling.")


    args = parser.parse_args()
    main(args)

# python scripts/generate_candidates.py vilem/scripts/data test_sample output --generation_mode sample --max_batch_size 50
# python scripts/generate_candidates.py vilem/scripts/data test_sample output --generation_mode beam 