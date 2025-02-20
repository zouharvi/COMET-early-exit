import logging
import sys
import h5py
import numpy as np
from collections import defaultdict

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANG_TO_NLLB = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "uk": "ukr_Cyrl",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "lt": "lit_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ps": "pbt_Arab",
    "fi": "fin_Latn",
    "zu": "zul_Latn",
    "xh": "xho_Latn",
    "ta": "tam_Taml",
    "pl": "pol_Latn",
    "iu": None,
    "km": "khm_Khmr",
    "gu": "guj_Gujr",
    "is": "isl_Latn",
    "ru": "rus_Cyrl",
    "kk": "kaz_Cyrl",
    "cs": "ces_Latn",
    "ha": "hau_Latn",
}


H5_STRING_DTYPE = h5py.special_dtype(vlen=str)
H5_VLEN_FLOAT_DTYPE = h5py.vlen_dtype(np.dtype('float32'))

CANDIDATES_FILENAME = "candidates"
CANDIDATES_TEXT_H5DS_NAME = "text"
CANDIDATES_TOKEN_LOGPROBS_H5DS_NAME = "token_logprobs"
CANDIDATES_COUNTS_H5DS_NAME = "counts"
CANDIDATES_EMBEDDINGS_H5DS_NAME = "embeddings"

# Name of COMET model should be appended to this
COMET_SCORES_FILENAME_BASE = "scores_comet_"
CONFIDENCES_FILENAME_BASE = "confidences_"
COMET_SCORES_H5DS_NAME = "scores"
COMET_CONFIDENCES_H5DS_NAME = "confidences"

EMBEDDINGS_FILENAME_BASE = "embeddings_"
EMBEDDINGS_H5DS_NAME = "embeddings"

SIMILARITIES_FILENAME_BASE = "similarities_"
SIMILARITIES_H5DS_NAME = "similarities"

LOGPROBS_FILENAME_BASE = "logprobs"
SUM_LOGPROBS_H5DS_NAME = "sum"
AVG_LOGPROBS_H5DS_NAME = "avg"

CONFIDENCE_MODELS = ["models-lithium", "models-beryllium", "models-oxygen", "models-nitrogen", "COMET-instant-self-confidence"]
NO_CONFIDENCE_MODELS = ["models-hydrogen", "models-helium"]
VILEM_MODELS = CONFIDENCE_MODELS + NO_CONFIDENCE_MODELS


def configure_logger(name, output_filename):
    file_handler = logging.FileHandler(output_filename, mode='w')
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[file_handler, stream_handler])
    
def average_dicts(dicts_list):
    sum_dict = defaultdict(int)
    num_dicts = len(dicts_list)
    for d in dicts_list:
        for key, value in d.items():
            sum_dict[key] += value
    avg_dict = {key: value / num_dicts for key, value in sum_dict.items()}   
    return avg_dict