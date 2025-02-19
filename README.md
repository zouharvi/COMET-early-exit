# COMET-early-exit

This repository contains code for the paper [Early-Exit and Instant Confidence Translation Quality Estimation](TODO) by Vilém Zouhar, Maike Züfle, Beni Egressy, Julius Cheng, Jan Niehues.

> **Abstract:** 
> Quality estimation is omnipresent in machine translation, for both evaluation and generation.
> Unfortunately, quality estimation models are often opaque and computationally expensive, making them impractical to be part of large-scale pipelines.
> In this work, we tackle two connected challenges:
> (1) reducing the cost of quality estimation at scale, and (2) developing an inexpensive uncertainty estimation method for quality estimation.
> To address the latter, we introduce Instant Confidence COMET, an uncertainty-aware quality estimation model that matches the performance of previous
> approaches at a fraction of their costs.
> We extend this to Early-Exit COMET, a quality estimation model that can compute quality scores and associated confidences already at early model layers, allowing us to early-exit computations and reduce evaluation costs.
> We also apply our model to machine translation reranking.
> We combine Early-Exit COMET with an upper confidence bound bandit algorithm to find the best candidate from a large pool without having to run the full evaluation model on all candidates.
> In both cases (evaluation and reranking) our methods reduce the required compute by 50% with very little degradation in performance.

<img src="meta/14-plot_conf_individual.svg" width="500em">

## Running pre-trained models

The implementation for the various COMET models is kept in [comet_early_exit](comet_early_exit).
To run our models, you need to first install this version of COMET either with:
```bash
pip install "git+https://github.com/zouharvi/COMET-early-exit#egg=comet-early-exit&subdirectory=comet_early_exit"
```
or in editable mode:
```bash
git clone https://github.com/zouharvi/COMET-early-exit.git
cd COMET-early-exit
pip3 install -e comet_early_exit
```

Then, this package can be used in Python with `comet_early_exit` package.
The package name changed intentionally from Unbabel's package name such that they are not mutually exclusive.
```python
import comet_early_exit
model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-instant-confidence"))
```

We offer three public models on HuggingFace: [COMET-instant-confidence](TODO), [COMET-instant-self-confidence](TODO), and [COMET-partial](TODO) train in various regimes on direct assessment up to WMT2022.
All models are reference-less, requiring only the source and the translation.

### COMET-instant-confidence

Behaves like standard quality estimation, but outputs two numbers: `scores` (as usual) and `confidences`, which is the estimated mean absolute error from the human score.
Thus, contrary to expectations, higher "confidence" correponds to less correct QE estimation.
```python
model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-instant-confidence"))
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
```
Outputs:
```
TODO
```

### COMET-instant-self-confidence

```python
model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-instant-self-confidence"))
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
assert all(len(l) for l in model_output["scores"] == 24) and all(len(l) for l in model_output["confidences"] == 24)
```
Outputs:
```
TODO
```

### COMET-partial

This model is described in the appendix in the paper.
It is able to score even incomplete translations:
```python
model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("zouharvi/COMET-partial"))
data = [
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Mohl bych",
    },
    {
        "src": "Can I receive my food in 10 to 15 minutes?",
        "mt": "Mohl bych dostat jídlo během 10 či 15 minut?",
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
```
Outputs:
```
TODO
```


## Replicating experiments in the paper

The [experiments/](experiments) directory contains scripts to run the experiments.
This is intentionally separate from the [comet_early_exit/](comet_early_exit/) package, which is a fork of [Unbabel's COMET](https://github.com/Unbabel/COMET/) from version 2.2.4.
For training your own models, as described in the paper, please see [experiments/04-launch_comet.sh](experiments/04-launch_comet.sh).

Description of plotting and other experiments is WIP.

## Citation

```
@misc{zouhar2025earlyexit,
      title={Early-Exit and Instant Confidence Translation Quality Estimation}, 
      author={Vilém Zouhar and Maike Züfle and Beni Egressy and Julius Cheng and Jan Niehues},
      year={2025},
      eprint={TODO},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={TODO},
}
```