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

CONFIDENCE_MODELS = ["models-lithium", "models-beryllium"]
NO_CONFIDENCE_MODELS = ["models-hydrogen", "models-helium"]
VILEM_MODELS = CONFIDENCE_MODELS + NO_CONFIDENCE_MODELS

CORRELATIONS_WITH_LAST_LAYER = {"models-beryllium" : [0.2576426289275995,
                                               0.2875294213811498,
                                               0.587797854276222,
                                               0.7064846671536522,
                                               0.7509703775460114,
                                               0.7663393222749102,
                                               0.775479003982045,
                                               0.7814294583978393,
                                               0.7861331663640891,
                                               0.7918893041572647,
                                               0.8948940180661606,
                                               0.9140959703089833,
                                               0.9317431598905953,
                                               0.9426748534014279,
                                               0.9493236372984017,
                                               0.9540749007605094,
                                               0.95690570886739,
                                               0.971448548895409,
                                               0.9876996438147793,
                                               0.9934140786778413,
                                               0.9962554539564784,
                                               0.9981032372900425,
                                               0.9990964310517031,
                                               1.0],
                                "models-hydrogen": [0.2500587709190746,
                                                    0.3145235936317737,
                                                    0.585871957594635,
                                                    0.697635458164735,
                                                    0.7406413658601467,
                                                    0.7611908504646157,
                                                    0.772903143317165,
                                                    0.7771557991985355,
                                                    0.7801773054442086,
                                                    0.7833344749867298,
                                                    0.867654327063682,
                                                    0.8904675437156966,
                                                    0.9021469358387726,
                                                    0.910414590934367,
                                                    0.9144935865422424,
                                                    0.9182103442677996,
                                                    0.9200030337065261,
                                                    0.9755264168843236,
                                                    0.979480987205641,
                                                    0.994603650497217,
                                                    0.9981397653258224,
                                                    0.9992442614701398,
                                                    0.9996646008158248,
                                                    1.0]}


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