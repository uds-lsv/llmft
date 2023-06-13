import math 
import deepspeed
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
from transformers.models.gpt_neox import GPTNeoXForCausalLM

from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast

from transformers.utils import logging

from transformers.deepspeed import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)


class GPTNeoXWithLMClassifier(GPTNeoXForCausalLM):

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        logits = self.embed_out(hidden_states)

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