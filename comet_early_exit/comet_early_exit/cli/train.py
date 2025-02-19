#!/usr/bin/env python3

# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

Command for training new Metrics.
=================================

e.g:
```
    comet-train --cfg configs/models/regression_metric.yaml --seed_everything 12
```

For more details run the following command:
```
    comet-train --help
```
"""
import json
import logging
import warnings

import torch
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping, LearningRateMonitor,ModelCheckpoint
)
from pytorch_lightning.trainer.trainer import Trainer

from comet_early_exit.models import (
    EarlyExitRegression, EarlyExitConfRegression,
    EarlyExitConfMultiRegression, EarlyExitConfMultiRegressionExtra,
    EarlyExitMultiRegression,
    InstantConfRegression, DirectUncertaintyPrediction,
    RankingMetric, ReferencelessRegression,
    RegressionMetric, UnifiedMetric
)

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training COMET models.")
    parser.add_argument(
        "--seed_everything",
        type=int,
        default=12,
        help="Training Seed.",
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_subclass_arguments(RegressionMetric, "regression_metric")
    parser.add_subclass_arguments(
        ReferencelessRegression, "referenceless_regression_metric"
    )
    parser.add_subclass_arguments(EarlyExitRegression, "earlyexit_metric")
    parser.add_subclass_arguments(EarlyExitMultiRegression, "earlyexitmulti")
    parser.add_subclass_arguments(EarlyExitConfRegression, "earlyexitconf_metric")
    parser.add_subclass_arguments(EarlyExitConfMultiRegression, "earlyexitconfmulti")
    parser.add_subclass_arguments(EarlyExitConfMultiRegressionExtra, "earlyexitconfmulti_extra")
    parser.add_subclass_arguments(RankingMetric, "ranking_metric")
    parser.add_subclass_arguments(UnifiedMetric, "unified_metric")
    parser.add_subclass_arguments(InstantConfRegression, "instantconf_metric")
    parser.add_subclass_arguments(DirectUncertaintyPrediction, "direct_uncertainty_prediction")
    parser.add_subclass_arguments(EarlyStopping, "early_stopping")
    parser.add_subclass_arguments(ModelCheckpoint, "model_checkpoint")
    parser.add_subclass_arguments(Trainer, "trainer")
    parser.add_argument(
        "--load_from_checkpoint",
        help="Loads a model checkpoint for fine-tuning",
        default=None,
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="Strictly enforce that the keys in checkpoint_path match the keys returned by this module's state dict.",
    )
    return parser


def initialize_trainer(configs) -> Trainer:
    checkpoint_callback = ModelCheckpoint(
        **namespace_to_dict(configs.model_checkpoint.init_args)
    )
    early_stop_callback = EarlyStopping(
        **namespace_to_dict(configs.early_stopping.init_args)
    )
    trainer_args = namespace_to_dict(configs.trainer.init_args)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer_args["callbacks"] = [early_stop_callback, checkpoint_callback, lr_monitor]
    print(json.dumps(trainer_args, indent=4, default=lambda x: x.__dict__))
    trainer = Trainer(**trainer_args)
    return trainer


def initialize_model(configs):
    if configs.regression_metric is not None:
        print(
            json.dumps(
                configs.regression_metric.init_args,
                indent=4,
                default=lambda x: x.__dict__,
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = RegressionMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.regression_metric.init_args),
            )
        else:
            model = RegressionMetric(
                **namespace_to_dict(configs.regression_metric.init_args)
            )
    elif configs.referenceless_regression_metric is not None:
        print(
            json.dumps(
                configs.referenceless_regression_metric.init_args,
                indent=4,
                default=lambda x: x.__dict__,
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = ReferencelessRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.referenceless_regression_metric.init_args),
            )
        else:
            model = ReferencelessRegression(
                **namespace_to_dict(configs.referenceless_regression_metric.init_args)
            )
    elif configs.ranking_metric is not None:
        print(
            json.dumps(
                configs.ranking_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = ReferencelessRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.ranking_metric.init_args),
            )
        else:
            model = RankingMetric(**namespace_to_dict(configs.ranking_metric.init_args))
    elif configs.earlyexit_metric is not None:
        print(
            json.dumps(
                configs.earlyexit_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = EarlyExitRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.earlyexit_metric.init_args),
            )
        else:
            model = EarlyExitRegression(**namespace_to_dict(configs.earlyexit_metric.init_args))
    elif configs.earlyexitconf_metric is not None:
        print(
            json.dumps(
                configs.earlyexitconf_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = EarlyExitConfRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.earlyexitconf_metric.init_args),
            )
        else:
            model = EarlyExitConfRegression(**namespace_to_dict(configs.earlyexitconf_metric.init_args))
    elif configs.instantconf_metric is not None:
        print(
            json.dumps(
                configs.instantconf_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = InstantConfRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.instantconf_metric.init_args),
            )
        else:
            model = InstantConfRegression(**namespace_to_dict(configs.instantconf_metric.init_args))
    elif configs.direct_uncertainty_prediction is not None:
        print(
            json.dumps(
                configs.direct_uncertainty_prediction.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = DirectUncertaintyPrediction.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.direct_uncertainty_prediction.init_args),
            )
        else:
            model = DirectUncertaintyPrediction(**namespace_to_dict(configs.direct_uncertainty_prediction.init_args))
    elif configs.earlyexitconfmulti_extra is not None:
        print(
            json.dumps(
                configs.earlyexitconfmulti_extra.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = EarlyExitConfMultiRegressionExtra.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.earlyexitconfmulti_extra.init_args),
            )
        else:
            model = EarlyExitConfMultiRegressionExtra(**namespace_to_dict(configs.earlyexitconfmulti_extra.init_args))
    elif configs.earlyexitconfmulti is not None:
        print(
            json.dumps(
                configs.earlyexitconfmulti.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = EarlyExitConfMultiRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.earlyexitconfmulti.init_args),
            )
        else:
            model = EarlyExitConfMultiRegression(**namespace_to_dict(configs.earlyexitconfmulti.init_args))
    elif configs.earlyexitmulti is not None:
        print(
            json.dumps(
                configs.earlyexitmulti.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = EarlyExitMultiRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.earlyexitmulti.init_args),
            )
        else:
            model = EarlyExitMultiRegression(**namespace_to_dict(configs.earlyexitmulti.init_args))
    elif configs.unified_metric is not None:
        print(
            json.dumps(
                configs.unified_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = UnifiedMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.unified_metric.init_args),
            )
        else:
            model = UnifiedMetric(**namespace_to_dict(configs.unified_metric.init_args))
    else:
        raise Exception("Model configurations missing!")

    return model


def train_command() -> None:
    parser = read_arguments()
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)

    trainer = initialize_trainer(cfg)
    model = initialize_model(cfg)
    # Related to train/val_dataloaders:
    # 2 workers per gpu is enough! If set to the number of cpus on this machine
    # it throws another exception saying its too many workers.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Consider increasing the value of the `num_workers` argument` .*",
    )
    trainer.fit(model)


if __name__ == "__main__":
    train_command()
