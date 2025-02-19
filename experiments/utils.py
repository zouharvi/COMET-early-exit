# %%
import collections

def compute_fertility():
    import numpy as np
    import json
    data_train = [json.loads(line) for line in open("../data/jsonl/train.jsonl")]
    fertility_train = collections.defaultdict(list)
    for line in data_train:
        fertility_train[line["langs"]].append(
            len(line["tgt"])/len(line["src"]))
    return collections.defaultdict(lambda: 1, {l: np.average(v) for l, v in fertility_train.items()})


FERTILITY = collections.defaultdict(
    lambda: 1,
    {'en-iu': 0.7532678052849321,
     'ps-en': 1.107623672092653,
     'en-cs': 0.9603753674875284,
     'ta-en': 0.9052049347448049,
     'en-lt': 0.9897782896996692,
     'en-zh': 0.34178103435228185,
     'en-ru': 1.0687373316205402,
     'en-fi': 1.038892198515933,
     'en-de': 1.1555434418780215,
     'zh-en': 3.9794817826360624,
     'ru-en': 1.063487683592595,
     'de-en': 0.9427733794500234,
     'iu-en': 1.3980283049406592,
     'en-ja': 0.44867591310726956,
     'de-cs': 0.8076087512536432,
     'lt-en': 1.05374171575283,
     'pl-en': 1.022216022407773,
     'de-fr': 1.0585864484950842,
     'kk-en': 1.0798036031581444,
     'ja-en': 2.804085731674332,
     'cs-en': 1.1232106453454278,
     'en-pl': 1.0555176271553215,
     'ha-en': 0.9264788884205395,
     'en-gu': 0.9451926180393779,
     'fi-en': 1.0162270858311704,
     'km-en': 1.041932934755382,
     'is-en': 1.038283887562052,
     'hi-bn': 0.9418982938937649,
     'bn-hi': 0.9884497691494066,
     'en-is': 1.023862721861111,
     'en-kk': 1.0335818208482184,
     'en-ha': 1.1021932686636249,
     'fr-de': 1.0025746457196412,
     'zu-xh': 0.9640429609204979,
     'en-ta': 1.145340858122618,
     'gu-en': 1.0696057645393695,
     'xh-zu': 0.9799569562988472,
     }
)
