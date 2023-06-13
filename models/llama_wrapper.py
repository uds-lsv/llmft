import math 
import deepspeed
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
from transformers.models.llama import LlamaModel, LlamaForSequenceClassification, LlamaForCausalLM

from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast

from transformers.utils import logging

from transformers.deepspeed import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)


class LlamaWithLMClassifier(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def _init_weights(self, module):
        super()._init_weights(module)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # logits.shape = (bsz, seq_len, vocab_size)
        logits = self.lm_head(outputs[0])

        # In the classification setting we only care about the last prediction
        # Get the position of the last non-padding token
        sequence_lengths = torch.ne(
            input_ids, self.config.pad_token_id).sum(-1) - 1
        logits = logits[torch.arange(
            input_ids.shape[0], device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            logits = logits.contiguous()
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            labels = labels.contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
