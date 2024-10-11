from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.opt import OPTPreTrainedModel, OPTModel
from transformers.models.gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel
from transformers.modeling_outputs import SequenceClassifierOutput, SequenceClassifierOutputWithPast, TokenClassifierOutput
import scipy.stats


def l1_loss_fn(score, target, ignore_index=-100):
    mask = (target == ignore_index)
    out = torch.nn.functional.l1_loss(score, target=target, reduction="none")

    return out[~mask]


def mse_loss_fn(score, target, ignore_index=-100):
    mask = (target == ignore_index)
    out = torch.nn.functional.mse_loss(score, target, reduction="none")

    return out[~mask]


def gaussian_loss_fn(score, target, eps=1e-4, quantile=None, kl_weight=0.0, return_kl=False, ignore_index=-100):
    mask = (target == ignore_index)
    # little different from the rest, score is Nx2, quantile is ignored, this is just a negative log likelihood of a Gaussian distribution
    assert (
        score.ndim == 2 and score.shape[-1] == 2
    ), "score has the wrong shape, expected Nx2 input but got {}".format(score.shape)
    assert (
        target.ndim == 1
    ), "target has the wrong shape, expected 1-d vector, got {}".format(target.shape)
    mu = score[:, 0]
    var = score[:, 1]

    assert (
        mu.shape == var.shape and mu.shape == target.shape
    ), "mean, std and target have non-compatible shapes, got {} {} {}".format(
        mu.shape, var.shape, target.shape
    )

    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    loss = 0.5 * torch.log(var) + 0.5 * (target - mu) ** 2 / (var) + 0.5 * math.log(2 * math.pi)
    assert target.shape == loss.shape, "loss should be a 1-d vector got {}".format(
        loss.shape
    )
    
    kl = 0.5 * (-torch.log(var) + var + mu ** 2 - 1)
    
    if kl_weight > 0:
        loss += kl_weight * kl
    
    if return_kl:
        return loss, kl

    return loss[~mask]


def pinball_loss_fn(score, target, quantiles, ignore_index=-100):
    mask = (target == ignore_index)

    assert (
        score.ndim == 2
    ), "score has the wrong shape, expected 2d input but got {}".format(score.shape)
    assert (
        target.ndim == 1
    ), "target has the wrong shape, expected 1-d vector, got {}".format(target.shape)

    target = target.reshape([-1, 1])
    delta_score = target - score
    loss = torch.maximum(delta_score * quantiles, delta_score * (quantiles - 1.0))
    return loss[~mask, :]


def mse_pinball_loss_fn(score, target, quantiles, pinball_reduction="sum", ignore_index=-100):
    mask = (target == ignore_index)

    num_quantiles = quantiles.shape[0]
    
    assert (
        score.ndim == 2
    ), "score has the wrong shape, expected 2d input but got {}".format(score.shape)
    assert (
        score.shape[-1] == num_quantiles + 1
    ), f"score shape does not match with number of quantiles, score is {score.shape} but number of quantiles is {num_quantiles}"
    assert (
        target.ndim == 1
    ), "target has the wrong shape, expected 1-d vector, got {}".format(target.shape)
    
    regression_loss = torch.nn.functional.mse_loss(score[:, 0], target, reduction="none")[~mask]
    pinball_loss = pinball_loss_fn(score[:, 1:], target, quantiles, ignore_index=ignore_index)
    if pinball_reduction == "sum":
        pinball_loss = pinball_loss.sum(-1)
    elif pinball_reduction == "mean":
        pinball_loss = pinball_loss.mean(-1)

    loss = regression_loss + pinball_loss
    return loss


def gaussian_pinball_loss_fn(score, target, eps=1e-4, quantile=None, ignore_index=-100):
    # little different from the rest, score is Nx2, quantile is ignored, this is just a negative log likelihood of a Gaussian distribution
    assert (
        score.ndim == 2 and score.shape[-1] == 2
    ), "score has the wrong shape, expected Nx2 input but got {}".format(score.shape)
    assert (
        target.ndim == 1
    ), "target has the wrong shape, expected 1-d vector, got {}".format(target.shape)

    gaussian_loss = gaussian_loss_fn(score, target, ignore_index=ignore_index)
    pinball_loss = pinball_loss_fn(score[:, [0]], target, torch.FloatTensor([0.5]).to(score.device), ignore_index=ignore_index).sum(-1) + pinball_loss_fn(score[:, [0]] + torch.sqrt(score[:, [1]]), target, torch.FloatTensor([1-scipy.stats.norm.sf(1)], ignore_index=ignore_index).to(score.device)).sum(-1)

    loss = gaussian_loss + pinball_loss

    return loss


class BertForQuantileRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        if self.config.regression_type == "regression":
            self.num_labels = 1
        elif self.config.regression_type == "gaussian_regression":
            self.num_labels = 2
            self.kl_weight = 0.0
            if self.config.kl_weight > 0:
                self.kl_weight = self.config.kl_weight
            self.var_nonlin = torch.nn.functional.softplus   
        elif self.config.regression_type == "iqr_regression":
            self.num_labels = len(self.config.quantiles)
            self.quantiles = self.config.quantiles
        elif self.config.regression_type == "mse_pinball_regression":
            self.num_labels = 1 + len(self.config.quantiles)
            self.quantiles = self.config.quantiles
        elif self.config.regression_type == "gaussian_pinball_regression":
            self.num_labels = 2
            self.var_nonlin = torch.nn.functional.softplus
        else:
            self.num_labels = self.config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if self.config.regression_type == "gaussian_regression":
            logits[:, 1] = self.var_nonlin(logits[:, 1].clone())
        elif self.config.regression_type == "gaussian_pinball_regression":
            logits[:, 1] = self.var_nonlin(logits[:, 1].clone())
        
        loss = None
        if labels is not None:
            if self.config.regression_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.regression_type == "gaussian_regression":
                loss_fct = gaussian_loss_fn
                loss = loss_fct(logits, labels, kl_weight=self.kl_weight).mean()
            elif self.config.regression_type == "iqr_regression":
                loss_fct = pinball_loss_fn
                loss = loss_fct(logits, labels, quantiles=torch.FloatTensor(self.quantiles).to(logits.device)).mean()
            elif self.config.regression_type == "mse_pinball_regression":
                loss_fct = mse_pinball_loss_fn
                loss = loss_fct(logits, labels, quantiles=torch.FloatTensor(self.quantiles).to(logits.device)).mean()
            elif self.config.regression_type == "gaussian_pinball_regression":
                loss_fct = gaussian_pinball_loss_fn
                loss = loss_fct(logits, labels).mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
class OPTForQuantileRegression(OPTPreTrainedModel):
    def __init__(self, config):
        config.dropout = 0.0
        super().__init__(config)
        
        if self.config.regression_type == "regression":
            self.num_labels = 1
        elif self.config.regression_type == "gaussian_regression":
            self.num_labels = 2
            self.kl_weight = 0.0
            if self.config.kl_weight > 0:
                self.kl_weight = self.config.kl_weight
            self.var_nonlin = torch.nn.functional.softplus   
        elif self.config.regression_type == "iqr_regression":
            self.num_labels = len(self.config.quantiles)
            self.quantiles = self.config.quantiles
        elif self.config.regression_type == "mse_pinball_regression":
            self.num_labels = 1 + len(self.config.quantiles)
            self.quantiles = self.config.quantiles
        elif self.config.regression_type == "gaussian_pinball_regression":
            self.num_labels = 2
            self.var_nonlin = torch.nn.functional.softplus
        else:
            self.num_labels = self.config.num_labels
        self.config = config

        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        if self.config.regression_type == "gaussian_regression":
            pooled_logits[:, 1] = self.var_nonlin(pooled_logits[:, 1].clone())
        elif self.config.regression_type == "gaussian_pinball_regression":
            pooled_logits[:, 1] = self.var_nonlin(pooled_logits[:, 1].clone())
            
        loss = None
        if labels is not None:
            if self.config.regression_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.regression_type == "gaussian_regression":
                loss_fct = gaussian_loss_fn
                loss = loss_fct(pooled_logits, labels, kl_weight=self.kl_weight).mean()
            elif self.config.regression_type == "iqr_regression":
                loss_fct = pinball_loss_fn
                loss = loss_fct(pooled_logits, labels, quantiles=torch.FloatTensor(self.quantiles).to(pooled_logits.device)).mean()
            elif self.config.regression_type == "mse_pinball_regression":
                loss_fct = mse_pinball_loss_fn
                loss = loss_fct(pooled_logits, labels, quantiles=torch.FloatTensor(self.quantiles).to(pooled_logits.device)).mean()
            elif self.config.regression_type == "gaussian_pinball_regression":
                loss_fct = gaussian_pinball_loss_fn
                loss = loss_fct(pooled_logits, labels).mean()

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value
        
        
class GPTNeoXForQuantileRegression(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        if self.config.regression_type == "regression":
            self.num_labels = 1
        elif self.config.regression_type == "gaussian_regression":
            self.num_labels = 2
            self.kl_weight = 0.0
            if self.config.kl_weight > 0:
                self.kl_weight = self.config.kl_weight
            self.var_nonlin = torch.nn.functional.softplus   
        elif self.config.regression_type == "iqr_regression":
            self.num_labels = len(self.config.quantiles)
            self.quantiles = self.config.quantiles
        elif self.config.regression_type == "mse_pinball_regression":
            self.num_labels = 1 + len(self.config.quantiles)
            self.quantiles = self.config.quantiles
        elif self.config.regression_type == "gaussian_pinball_regression":
            self.num_labels = 2
            self.var_nonlin = torch.nn.functional.softplus
        else:
            self.num_labels = self.config.num_labels
        self.config = config

        self.gpt_neox = GPTNeoXModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        if self.config.regression_type == "gaussian_regression":
            pooled_logits[:, 1] = self.var_nonlin(pooled_logits[:, 1].clone())
        elif self.config.regression_type == "gaussian_pinball_regression":
            pooled_logits[:, 1] = self.var_nonlin(pooled_logits[:, 1].clone())

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.regression_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.regression_type == "gaussian_regression":
                loss_fct = gaussian_loss_fn
                loss = loss_fct(pooled_logits, labels, kl_weight=self.kl_weight).mean()
            elif self.config.regression_type == "iqr_regression":
                loss_fct = pinball_loss_fn
                loss = loss_fct(pooled_logits, labels, quantiles=torch.FloatTensor(self.quantiles).to(pooled_logits.device)).mean()
            elif self.config.regression_type == "mse_pinball_regression":
                loss_fct = mse_pinball_loss_fn
                loss = loss_fct(pooled_logits, labels, quantiles=torch.FloatTensor(self.quantiles).to(pooled_logits.device)).mean()
            elif self.config.regression_type == "gaussian_pinball_regression":
                loss_fct = gaussian_pinball_loss_fn
                loss = loss_fct(pooled_logits, labels).mean()

        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class GPTNeoXForTokenQuantileRegression(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if self.config.regression_type == "regression":
            self.num_labels = 1
        elif self.config.regression_type == "gaussian_regression":
            self.num_labels = 2
            self.kl_weight = 0.0
            if self.config.kl_weight > 0:
                self.kl_weight = self.config.kl_weight
            self.var_nonlin = torch.nn.functional.softplus   
        elif self.config.regression_type == "iqr_regression":
            self.num_labels = len(self.config.quantiles)
            self.quantiles = self.config.quantiles
        elif self.config.regression_type == "mse_pinball_regression":
            self.num_labels = 1 + len(self.config.quantiles)
            self.quantiles = self.config.quantiles
        elif self.config.regression_type == "gaussian_pinball_regression":
            self.num_labels = 2
            self.var_nonlin = torch.nn.functional.softplus
        else:
            self.num_labels = self.config.num_labels
        self.config = config

        self.gpt_neox = GPTNeoXModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        if self.config.regression_type == "gaussian_regression":
            logits[..., 1] = self.var_nonlin(logits[..., 1].clone())
        elif self.config.regression_type == "gaussian_pinball_regression":
            logits[..., 1] = self.var_nonlin(logits[..., 1].clone())

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            # the label is already shifted by one position, i.e. the logloss of the next token
            # the logits starts from position 0 and is actually one token longer
            # label and logits should have been padded to the same length
            # we would like to shift the logits by one position (to the left) so that logits are aligned with the token
            # as we cannot predict the probability of the token without seeing it
            shift_logits = logits[:, 1:, :].contiguous()
            labels = labels[:, :-1].contiguous()

            # TODO: one issue is that there are masked labels (how to deal with it?)
            if self.config.regression_type == "regression":
                loss_fct = mse_loss_fn
                loss = loss_fct(shift_logits.view(-1), labels.view(-1)).mean()
            elif self.config.regression_type == "gaussian_regression":
                loss_fct = gaussian_loss_fn
                loss = loss_fct(shift_logits.view(-1, self.num_labels), labels.view(-1), kl_weight=self.kl_weight).mean()
            elif self.config.regression_type == "iqr_regression":
                loss_fct = pinball_loss_fn
                loss = loss_fct(shift_logits.view(-1, self.num_labels), labels.view(-1), quantiles=torch.FloatTensor(self.quantiles).to(logits.device)).mean()
            elif self.config.regression_type == "mse_pinball_regression":
                loss_fct = mse_pinball_loss_fn
                loss = loss_fct(shift_logits.view(-1, self.num_labels), labels.view(-1), quantiles=torch.FloatTensor(self.quantiles).to(logits.device)).mean()
            elif self.config.regression_type == "gaussian_pinball_regression":
                loss_fct = gaussian_pinball_loss_fn
                loss = loss_fct(shift_logits.view(-1, self.num_labels), labels.view(-1)).mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )