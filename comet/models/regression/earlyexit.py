# -*- coding: utf-8 -*-
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

r"""
EarlyExitRegression
========================
    Early exit referenceless regression Metric that learns to predict a quality assessment by
    looking at source and translation.
"""
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import torch

from comet.models.regression.regression_metric import RegressionMetric
from comet.models.utils import Prediction, Target
from comet.modules import FeedForward
from comet.models.pooling_utils import average_pooling, max_pooling
from comet.models.metrics import EarlyExitMetrics
from torch import nn
import pytorch_lightning as ptl
from ..predict_pbar import PredictProgressBar
from torch.utils.data import DataLoader



class EarlyExitRegression(RegressionMetric):
    """EarlyExitRegression:

    Args:
        nr_frozen_epochs (Union[float, int]): Number of epochs (% of epoch) that the
            encoder is frozen. Defaults to 0.9.
        keep_embeddings_frozen (bool): Keeps the encoder frozen during training. Defaults
            to True.
        optimizer (str): Optimizer used during training. Defaults to 'AdamW'.
        warmup_steps (int): Warmup steps for LR scheduler.
        encoder_learning_rate (float): Learning rate used to fine-tune the encoder model.
            Defaults to 3.0e-06.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 3.0e-05.
        layerwise_decay (float): Learning rate % decay from top-to-bottom encoder layers.
            Defaults to 0.95.
        encoder_model (str): Encoder model to be used. Defaults to 'XLM-RoBERTa'.
        pretrained_model (str): Pretrained model from Hugging Face. Defaults to
            'microsoft/infoxlm-large'.
        pool (str): Type of sentence level pooling (options: 'max', 'cls', 'avg').
            Defaults to 'avg'
        layer (Union[str, int]): Encoder layer to be used for regression ('mix'
            for pooling info from all layers). Defaults to 'mix'.
        layer_transformation (str): Transformation applied when pooling info from all
            layers (options: 'softmax', 'sparsemax'). Defaults to 'sparsemax'.
        layer_norm (bool): Apply layer normalization. Defaults to 'False'.
        loss (str): Loss function to be used. Defaults to 'mse'.
        dropout (float): Dropout used in the top-layers. Defaults to 0.1.
        batch_size (int): Batch size used during training. Defaults to 4.
        train_data (Optional[List[str]]): List of paths to training data. Each file is
            loaded consecutively for each epoch. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data.
            Validation results are averaged across validation set. Defaults to None.
        hidden_sizes (List[int]): Hidden sizes for the Feed Forward regression.
        activations (str): Feed Forward activation function.
        final_activation (str): Feed Forward final activation.
        local_files_only (bool): Whether or not to only look at local files.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 1e-06,
        learning_rate: float = 1.5e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-large",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        layer_transformation: str = "softmax",
        layer_norm: bool = True,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        hidden_sizes: List[int] = [2048, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super(RegressionMetric, self).__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            pool=pool,
            layer=layer,
            layer_transformation=layer_transformation,
            layer_norm=layer_norm,
            loss=loss,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            validation_data=validation_data,
            class_identifier="earlyexit_metric",
            load_pretrained_weights=load_pretrained_weights,
            local_files_only=local_files_only,
        )
        self.save_hyperparameters()
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units * 4,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
            out_dim=1,
        )

    def requires_references(self) -> bool:
        return False
    
    def enable_context(self):
        if self.pool == "avg":
            self.use_context = True

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """This method will be called by dataloaders to prepared data to input to the
        model.

        Args:
            sample (List[dict]): Batch of train/val/test samples.
            stage (str): model stage (options: 'fit', 'validate', 'test', or
                'predict'). Defaults to 'fit'.

        Returns:
            Model inputs and depending on the 'stage' training labels/targets.
        """
        inputs = {k: [str(dic[k]) for dic in sample] for k in sample[0] if k != "score"}
        src_inputs = self.encoder.prepare_sample(inputs["src"])
        mt_inputs = self.encoder.prepare_sample(inputs["mt"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        model_inputs = {**src_inputs, **mt_inputs}

        if stage == "predict":
            return model_inputs

        # duplicate the scores to the number of output layers because we do a prediction from each layer
        scores = [[float(s["score"])]*self.encoder.num_layers for s in sample]
        # transpose to be [samples/batch, layers]
        targets = Target(score=torch.tensor(scores, dtype=torch.float).T)

        if "system" in inputs:
            targets["system"] = inputs["system"]

        return model_inputs, targets
    

    def init_metrics(self):
        """Initializes train/validation metrics."""
        self.train_metrics = EarlyExitMetrics(num_layers=self.encoder.num_layers, prefix="train")
        self.val_metrics = nn.ModuleList([
            EarlyExitMetrics(num_layers=self.encoder.num_layers, prefix="val")
            for d in self.hparams.validation_data
        ])

    def forward(
        self,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt_input_ids: torch.tensor,
        mt_attention_mask: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """EarlyExitRegression model forward method.

        Args:
            src_input_ids [torch.tensor]: input ids from source sentences.
            src_attention_mask [torch.tensor]: Attention mask from source sentences.
            mt_input_ids [torch.tensor]: input ids from MT.
            mt_attention_mask [torch.tensor]: Attention mask from MT.

        Return:
            Prediction object with translation scores.
        """
        src_sentembs = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        mt_sentembs = self.get_sentence_embedding(mt_input_ids, mt_attention_mask)

        # paralel computation from all layers
        diff_src = torch.abs(mt_sentembs - src_sentembs)
        prod_src = mt_sentembs * src_sentembs
        embedded_sequences = torch.cat(
            (mt_sentembs, src_sentembs, prod_src, diff_src), dim=2
        )
        # last dimension is [1]
        output = self.estimator(embedded_sequences).squeeze(dim=2)

        return Prediction(score=output)

    def read_training_data(self, path: str) -> List[dict]:
        """Method that reads the training data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        df = df[["src", "mt", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype("float16")
        return df.to_dict("records")

    def read_validation_data(self, path: str) -> List[dict]:
        """Method that reads the validation data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        columns = ["src", "mt", "score"]
        # If system in columns we will use this to calculate system-level accuracy
        if "system" in df.columns:
            columns.append("system")
            df["system"] = df["system"].astype(str)

        df = df[columns]
        df["score"] = df["score"].astype("float16")
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        return df.to_dict("records")


    # override functions from base because we compute embeddings slightly differently
    def compute_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Function that extracts sentence embeddings for
        a single sentence.

        Args:
            tokens (torch.Tensor): sequences [batch_size x seq_len].
            attention_mask (torch.Tensor): attention_mask [batch_size x seq_len].
            token_type_ids (torch.Tensor): Model token_type_ids [batch_size x seq_len].
                Optional

        Returns:
            torch.Tensor [batch_size x hidden_size] with sentence embeddings.
        """
        encoder_out = self.encoder(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )
        # NOTE: here we drop support for layerwise attention, but potentially can bring it back
        # if self.layerwise_attention:
        #     embeddings = self.layerwise_attention(
        #         encoder_out["all_layers"], attention_mask
        #     )

        # elif self.hparams.layer >= 0 and self.hparams.layer < self.encoder.num_layers:

        # NOTE: if we use some intermediary computation, we mess up the graph?
        return torch.stack([x[:,0,:] for x in encoder_out["all_layers"]])

        sentembs = []
        for embeddings in encoder_out["all_layers"]:
            if self.hparams.pool == "default":
                sentemb = encoder_out["sentemb"]

            elif self.hparams.pool == "max":
                sentemb = max_pooling(
                    input_ids, embeddings, self.encoder.tokenizer.pad_token_id
                )

            elif self.hparams.pool == "avg":
                sentemb = average_pooling(
                    input_ids,
                    embeddings,
                    attention_mask,
                    self.encoder.tokenizer.pad_token_id,
                    self.encoder.tokenizer.sep_token_id, 
                    self.use_context,
                )

            elif self.hparams.pool == "cls":
                sentemb = embeddings[:, 0, :]

            else:
                raise Exception("Invalid pooling technique.")
            
            sentembs.append(sentemb)

        return torch.stack(sentembs)


    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Pytorch Lightning predict step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
            dataloader_idx (int): Integer displaying which dataloader this sample is
                coming from.

        Return:
            Predicion object
        """
        model_outputs = Prediction(scores=self(**batch).score)
        if self.mc_dropout:
            raise Exception("MC Dropout not available for Early Exit now")
        
        return model_outputs



    def predict(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 16,
        gpus: int = 1,
        devices: Union[List[int], str, int] = None,
        mc_dropout: int = 0,
        progress_bar: bool = True,
        accelerator: str = "auto",
        num_workers: int = None,
        length_batching: bool = True,
    ) -> Prediction:
        """
        This is a stripped down version of the predict method of base.py without metadata and length batching but
        with support for early exit model.
        """
        if mc_dropout > 0:
            self.set_mc_dropout(mc_dropout)
        assert gpus <= 1

        if gpus > 0 and devices is not None:
            assert len(devices) == gpus, AssertionError(
                "List of devices must be same size as `gpus` or None if `gpus=0`"
            )
        elif gpus > 0:
            devices = gpus
        else: # gpu = 0
            devices = "auto"
        

        self.eval()
        dataloader = DataLoader(
            dataset=samples,
            batch_size=batch_size,
            collate_fn=self.prepare_for_inference,
            num_workers=2,
            multiprocessing_context="fork" if torch.backends.mps.is_available() else None,
        )
        callbacks = []

        if progress_bar:
            enable_progress_bar = True
            callbacks.append(PredictProgressBar())
        else:
            enable_progress_bar = False

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer = ptl.Trainer(
            devices=devices,
            logger=False,
            callbacks=callbacks,
            accelerator=accelerator if gpus > 0 else "cpu",
            strategy="auto" if gpus < 2 else "ddp",
            enable_progress_bar=enable_progress_bar,
        )
        predictions = trainer.predict(
            self, dataloaders=dataloader, return_predictions=True
        )

        scores = torch.cat([pred["scores"] for pred in predictions], dim=1).T.tolist()
        output = Prediction(scores=scores, system_score=np.average(scores, axis=0))

        return output