# COMET-early-exit-experiments

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

Then, this package can be used in Python:

```python
import comet_early_exit
model = comet_early_exit.load_from_checkpoint(comet_early_exit.download_model("TODO"))
```

TODO

## Training your own models

TODO

## Replicating experiments in the paper

TODO


## Misc

Cite as:
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