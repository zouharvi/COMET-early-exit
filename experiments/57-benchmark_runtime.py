# %%
# %%
import collections
import time
import torch
import comet_early_exit
data = [
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Moh bych obdržet jídlo v 10 do 15 minut?",
    },
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Mohl bych dostat jídlo během 10 či 15 minut?",
    }
] * 50
print(len(data))
# %%

model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-instant-self-confidence"))
model = model.to("cuda:0")
model.eval()
model.layerwise_attention = None
model.hparams.layer = 24
model.layer_index_fixed = model.layer_index_fixed.to("cuda:0")
model.encoder.num_layers

TIMES = collections.defaultdict(list)

with torch.no_grad():
    torch.multiprocessing.set_start_method('spawn', force=True)
    for _ in range(100):
        time_start = time.time()
        example = model.prepare_sample(data, stage="predict")
        TIMES["prepare_sample"].append(time.time() - time_start)

        time_start = time.time()
        src_sentembs = model.get_sentence_embedding(example["src_input_ids"].to("cuda:0"), example["src_attention_mask"].to("cuda:0"))
        mt_sentembs = model.get_sentence_embedding(example["mt_input_ids"].to("cuda:0"), example["mt_attention_mask"].to("cuda:0"))
        TIMES["embedding"].append(time.time() - time_start)

        print(src_sentembs.shape)
        break

        time_start = time.time()
        diff_src = torch.abs(mt_sentembs - src_sentembs)
        prod_src = mt_sentembs * src_sentembs
        embedded_sequences = torch.cat(
            (mt_sentembs, src_sentembs, prod_src, diff_src), dim=2
        ).to("cuda:0")
        batch_size = embedded_sequences.shape[1]
        embedded_sequences = torch.cat([
            embedded_sequences,
            model.layer_index_param.unsqueeze(1).expand(-1, batch_size).unsqueeze(-1),
            model.layer_index_fixed.unsqueeze(1).expand(-1, batch_size).unsqueeze(-1),
        ], dim=-1).to("cuda:0")
        TIMES["arithmetics"].append(time.time() - time_start)

        time_start = time.time()
        output = model.estimator(embedded_sequences)
        TIMES["estimator"].append(time.time() - time_start)

for key, values in TIMES.items():
    print(f"{key:>25} {sum(values):.3f}")
print(f"total {sum([sum(v) for v in TIMES.values()]):.3f}")

# %%

# %%
model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-instant-self-confidence"))
model = model.to("cuda:0")
model.eval()
model.layerwise_attention = None
model.hparams.layer = 24
model.layer_index_fixed = model.layer_index_fixed.to("cuda:0")
model.encoder.num_layers


TIMES = collections.defaultdict(list)

with torch.no_grad():
    for _ in range(100):
        time_start = time.time()
        example = model.prepare_sample(data, stage="predict")
        TIMES["prepare_sample"].append(time.time() - time_start)

        time_start = time.time()
        src_sentembs = model.get_sentence_embedding(example["src_input_ids"].to("cuda:0"), example["src_attention_mask"].to("cuda:0"))
        mt_sentembs = model.get_sentence_embedding(example["mt_input_ids"].to("cuda:0"), example["mt_attention_mask"].to("cuda:0"))
        TIMES["embedding"].append((time.time() - time_start)/2)

        # take only half
        src_sentembs = src_sentembs[:13, :, :]
        mt_sentembs = mt_sentembs[:13, :, :]

        time_start = time.time()
        diff_src = torch.abs(mt_sentembs - src_sentembs)
        prod_src = mt_sentembs * src_sentembs
        embedded_sequences = torch.cat(
            (mt_sentembs, src_sentembs, prod_src, diff_src), dim=2
        ).to("cuda:0")
        batch_size = embedded_sequences.shape[1]
        embedded_sequences = torch.cat([
            embedded_sequences,
            model.layer_index_param[:13].unsqueeze(1).expand(-1, batch_size).unsqueeze(-1),
            model.layer_index_fixed[:13].unsqueeze(1).expand(-1, batch_size).unsqueeze(-1),
        ], dim=-1).to("cuda:0")
        TIMES["arithmetics"].append(time.time() - time_start)

        time_start = time.time()
        output = model.estimator(embedded_sequences)
        TIMES["estimator"].append(time.time() - time_start)

for key, values in TIMES.items():
    print(f"{key:>25} {sum(values):.3f}")
print(f"total {sum([sum(v) for v in TIMES.values()]):.3f}")

# %%

model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-instant-confidence"))
model = model.to("cuda:0")
model.layerwise_attention = None
model.hparams.layer = 24
model.eval()

TIMES = collections.defaultdict(list)

with torch.no_grad():
    for _ in range(100):
        time_start = time.time()
        example = model.prepare_sample(data, stage="predict")
        TIMES["prepare_sample"].append(time.time() - time_start)

        time_start = time.time()
        src_sentembs = model.get_sentence_embedding(example["src_input_ids"].to("cuda:0"), example["src_attention_mask"].to("cuda:0"))
        mt_sentembs = model.get_sentence_embedding(example["mt_input_ids"].to("cuda:0"), example["mt_attention_mask"].to("cuda:0"))
        TIMES["embedding"].append(time.time() - time_start)


        time_start = time.time()
        diff_src = torch.abs(mt_sentembs - src_sentembs)
        prod_src = mt_sentembs * src_sentembs
        embedded_sequences = torch.cat(
            (mt_sentembs, src_sentembs, prod_src, diff_src), dim=1
        ).to("cuda:0")
        TIMES["arithmetics"].append(time.time() - time_start)

        time_start = time.time()
        output = model.estimator(embedded_sequences)
        TIMES["estimator"].append(time.time() - time_start)

for key, values in TIMES.items():
    print(f"{key:>25} {sum(values):.3f}")
print(f"total {sum([sum(v) for v in TIMES.values()]):.3f}")


# %%


model = comet_early_exit.load_from_checkpoint("../lightning_logs/model-helium/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt")
model = model.to("cuda:0")
model.layerwise_attention = None
model.hparams.layer = 24
model.eval()

TIMES = collections.defaultdict(list)
with torch.no_grad():
    for _ in range(100):
        time_start = time.time()
        example = model.prepare_sample(data, stage="predict")
        TIMES["prepare_sample"].append(time.time() - time_start)

        time_start = time.time()
        src_sentembs = model.get_sentence_embedding(example["src_input_ids"].to("cuda:0"), example["src_attention_mask"].to("cuda:0"))
        mt_sentembs = model.get_sentence_embedding(example["mt_input_ids"].to("cuda:0"), example["mt_attention_mask"].to("cuda:0"))
        TIMES["embedding"].append(time.time() - time_start)

        time_start = time.time()
        diff_src = torch.abs(mt_sentembs - src_sentembs)
        prod_src = mt_sentembs * src_sentembs
        embedded_sequences = torch.cat(
            (mt_sentembs, src_sentembs, prod_src, diff_src), dim=1
        ).to("cuda:0")
        TIMES["arithmetics"].append(time.time() - time_start)

        time_start = time.time()
        output = model.estimator(embedded_sequences)
        TIMES["estimator"].append(time.time() - time_start)

for key, values in TIMES.items():
    print(f"{key:>25} {sum(values):.3f}")
print(f"total {sum([sum(v) for v in TIMES.values()]):.3f}")


# %%


model = comet_early_exit.load_from_checkpoint("../lightning_logs/model-helium/checkpoints/epoch=4-step=29320-val_pearson=0.419.ckpt")
model = model.to("cuda:0")
model.layerwise_attention = None
model.hparams.layer = 24
# to enable dropout
model.train()

TIMES = collections.defaultdict(list)
for _ in range(100):
    time_start = time.time()
    example = model.prepare_sample(data, stage="predict")
    TIMES["prepare_sample"].append(time.time() - time_start)

    time_start = time.time()
    src_sentembs = model.get_sentence_embedding(example["src_input_ids"].to("cuda:0"), example["src_attention_mask"].to("cuda:0"))
    mt_sentembs = model.get_sentence_embedding(example["mt_input_ids"].to("cuda:0"), example["mt_attention_mask"].to("cuda:0"))
    TIMES["embedding"].append(time.time() - time_start)

    time_start = time.time()
    diff_src = torch.abs(mt_sentembs - src_sentembs)
    prod_src = mt_sentembs * src_sentembs
    embedded_sequences = torch.cat(
        (mt_sentembs, src_sentembs, prod_src, diff_src), dim=1
    ).to("cuda:0")
    TIMES["arithmetics"].append(time.time() - time_start)

    time_start = time.time()
    output = model.estimator(embedded_sequences)
    TIMES["estimator"].append(time.time() - time_start)

for key, values in TIMES.items():
    print(f"{key:>25} {sum(values):.3f}")
print(f"total {sum([sum(v) for v in TIMES.values()]):.3f}")