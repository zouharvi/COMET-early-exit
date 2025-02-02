import datasets

# kken is in the WMT18 dataset, but the Hugging Face repo is missing the data, so
# we omit it.
WMT18_LANGUAGE_PAIRS = ["csen", "deen", "eten", "fien", "ruen", "tren", "zhen"]
WMT18_LANGUAGE_PAIRS += [lp[2:4] + lp[0:2] for lp in WMT18_LANGUAGE_PAIRS]


def load_dataset(src_lang, tgt_lang, split=None, subset=None):
    # if language_pair not in WMT18_LANGUAGE_PAIRS:
    #     raise ValueError(
    #         f"Language pair '{language_pair}' not supported by WMT18. "
    #         f"Supported values are {WMT18_LANGUAGE_PAIRS}")
    # if language_pair[0:2] == "en":
    #     language_pair = language_pair[2:4] + language_pair[0:2]

    def dataset_map_fn(example):
        return {"src": example["translation"][src_lang],
                "tgt": example["translation"][tgt_lang]}

    if src_lang == "en":
        return datasets.load_dataset("wmt18", f'{tgt_lang}-{src_lang}', split=split).map(dataset_map_fn)
    else:
        return datasets.load_dataset("wmt18", f'{src_lang}-{tgt_lang}', split=split).map(dataset_map_fn)