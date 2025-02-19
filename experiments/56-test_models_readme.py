# %%

import comet_early_exit

model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-instant-confidence"))
# model = comet_early_exit.load_from_checkpoint("../huggingface/COMET-instant-confidence/checkpoints/model.ckpt")
data = [
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Moh bych obdržet jídlo v 10 do 15 minut?",
    },
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Mohl bych dostat jídlo během 10 či 15 minut?",
    }
]
model_output = model.predict(data, batch_size=8, gpus=1)
print("scores", model_output["scores"])
print("estimated errors", model_output["confidences"])

assert len(model_output["scores"]) == 2 and len(model_output["confidences"]) == 2


# %%

model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-instant-self-confidence"))
# model = comet_early_exit.load_from_checkpoint("../huggingface/COMET-instant-self-confidence/checkpoints/model.ckpt")
data = [
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Moh bych obdržet jídlo v 10 do 15 minut?",
    },
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Mohl bych dostat jídlo během 10 či 15 minut?",
    }
]
model_output = model.predict(data, batch_size=8, gpus=1)

# print predictions at 5th, 12th, and last layer
print("scores", model_output["scores"][0][5], model_output["scores"][0][12], model_output["scores"][0][-1])
print("estimated errors", model_output["confidences"][0][5], model_output["confidences"][0][12], model_output["confidences"][0][-1])

# two top-level outputs
assert len(model_output["scores"]) == 2 and len(model_output["confidences"]) == 2
# each output contains prediction per each layer
assert all(len(l) == 25 for l in model_output["scores"]) and all(len(l) == 25 for l in model_output["confidences"])

# %%

model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-partial"))
# model = comet_early_exit.load_from_checkpoint("../huggingface/COMET-partial/checkpoints/model.ckpt")
data = [
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Mohl bych",
    },
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Mohl bych dostat jídlo",
    },
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Mohl bych dostat jídlo během 10 či 15 minut?",
    }
]
model_output = model.predict(data, batch_size=8, gpus=1)
print("scores", model_output["scores"])