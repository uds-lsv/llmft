import math
import deepspeed
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
from transformers.models.opt import OPTModel, OPTForSequenceClassification, OPTForCausalLM

from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast

from transformers.utils import logging

from transformers.deepspeed import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)


class LoRAAdapter(nn.Module):
    # LoRA adapter
    def __init__(self, hidden_size, adapter_dim, lora_alpha, dropout, training):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim, bias=True)
        self.non_linearity = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_size, bias=True)
        self.dropout = dropout
        self.training = training

        if lora_alpha == -1:
            # make the scale trainable
            self.scale = torch.nn.Parameter(torch.ones(1))
        else:
            self.scale = lora_alpha

        # init weights
        # we follow https://arxiv.org/abs/2110.04366 here
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)

        # LoRA initializes the second matrix with zeros
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _init_weights(self):
        # we will run this only when using deepspeed ZeRO stage 3
        # as in that case the initialization above will be ignored and we need to gather weights before initializing them

        with deepspeed.zero.GatheredParameters([self.down.weight, self.down.bias, self.up.weight, self.up.bias], modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down.bias)
                # LoRA initializes the second matrix with zeros
                nn.init.zeros_(self.up.weight)
                nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x = self.down(x)
        x = self.non_linearity(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.up(x)
        x = torch.mul(x, self.scale)
        return x


class MLPAdapter(nn.Module):
    def __init__(self, hidden_size, adapter_dim, dropout, training):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim, bias=True)
        self.non_linearity = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_size, bias=True)
        self.dropout = dropout
        self.training = training
        self.scale = torch.nn.Parameter(torch.ones(1))

        # init weights
        # we follow https://arxiv.org/abs/2110.04366 here
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.bias)

    def _init_weights(self):
        # we will run this only when using deepspeed ZeRO stage 3
        # as in that case the initialization above will be ignored and we need to gather weights before initializing them

        with deepspeed.zero.GatheredParameters([self.down.weight, self.down.bias, self.up.weight, self.up.bias], modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down.bias)
                nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x = self.down(x)
        x = self.non_linearity(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.up(x)
        x = torch.mul(x, self.scale)
        return x


class ScalingAdapter(nn.Module):
    # An adapter that simply scales the input element-wise by some learnable vector
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.ones(
                hidden_size,
            ),
        )

    def forward(self, x):
        return torch.mul(x, self.scale)


class ClassificationHead(nn.Module):
    # A classification head that can be put on top of an OPT model
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.word_embed_proj_dim,
                               config.word_embed_proj_dim, bias=True)
        self.activation = nn.ReLU()  # we use ReLU instead of tanh
        self.output = nn.Linear(config.word_embed_proj_dim,
                                config.num_labels, bias=True)
        self.dropout = nn.Dropout(config.dropout)

        # initialize weights
        self.dense.weight.data.normal_(
            mean=0.0, std=config.init_std)
        self.dense.bias.data.zero_()
        self.output.weight.data.normal_(
            mean=0.0, std=config.init_std)
        self.output.bias.data.zero_()

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class OPTAttention(transformers.models.opt.modeling_opt.OPTAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        adapter_type: str = None,
        adapter_dim: int = None,
        lora_alpha: float = None,
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)
        self.adapter_type = adapter_type
        self.adapter_dim = adapter_dim
        self.lora_alpha = lora_alpha

        if self.adapter_type == "lora":
            self.query_adapter = LoRAAdapter(
                embed_dim, adapter_dim, lora_alpha, dropout=self.dropout, training=self.training)
            self.value_adapter = LoRAAdapter(
                embed_dim, adapter_dim, lora_alpha, dropout=self.dropout, training=self.training)
        elif self.adapter_type == "ia3":
            self.key_adapter = ScalingAdapter(embed_dim)
            self.value_adapter = ScalingAdapter(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        if self.adapter_type == "lora":
            query_states = query_states + self.query_adapter(hidden_states)

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if self.adapter_type == "lora":
                value_states = value_states + self.value_adapter(hidden_states)
            elif self.adapter_type == "ia3":
                key_states = key_states + self.key_adapter(hidden_states)
                value_states = value_states + self.value_adapter(hidden_states)

            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(
            query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(
                torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(
                1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(
            bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(transformers.models.opt.modeling_opt.OPTDecoderLayer):
    def __init__(self, config):
        super().__init__(config)

        self.adapter_type = None
        if config.use_adapters:
            self.adapter_type = config.adapter_type

            if config.adapter_type in ["lora", "ia3"]:
                self.self_attn = OPTAttention(
                    embed_dim=self.embed_dim,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                    is_decoder=True,
                    adapter_type=config.adapter_type,
                    adapter_dim=config.adapter_dim,
                    lora_alpha=config.lora_alpha
                )

            self.attention_adapter = None
            self.fc_adapter = None
            if config.adapter_type == "ia3":
                self.fc_adapter = ScalingAdapter(config.ffn_dim)
            elif config.adapter_type in ["parallel-fc", "sequential-fc"]:
                self.fc_adapter = MLPAdapter(
                    self.embed_dim, config.adapter_dim, dropout=self.dropout, training=self.training)
            elif config.adapter_type in ["parallel-attn", "sequential-attn"]:
                self.attention_adapter = MLPAdapter(
                    self.embed_dim, config.adapter_dim, dropout=self.dropout, training=self.training)
            elif config.adapter_type in ["parallel", "sequential"]:
                self.attention_adapter = MLPAdapter(
                    self.embed_dim, config.adapter_dim, dropout=self.dropout, training=self.training)
                self.fc_adapter = MLPAdapter(
                    self.embed_dim, config.adapter_dim, dropout=self.dropout, training=self.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Parallel adapters apply the adapter to the input of the attention block
        adapter_output = None
        if self.adapter_type in ["parallel", "parallel-attn"]:
            adapter_output = self.attention_adapter(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)

        # we add the result of the parallel adapter to the result of the attn block
        if adapter_output is not None:
            hidden_states = hidden_states + adapter_output

        # Sequential adapters apply the adapter to the output of the attention block
        if self.adapter_type in ["sequential", "sequential-attn"]:
            hidden_states = self.attention_adapter(hidden_states)

        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        # Parallel adapters apply the adapter to the input of the fc block
        adapter_output = None
        if self.adapter_type in ["parallel", "parallel-fc"]:
            adapter_output = self.fc_adapter(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        # (IA)^3 applies element-wise multiplication here
        if self.adapter_type == "ia3":
            hidden_states = self.fc_adapter(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)

        # we add the result of the parallel adapter to the result of the fc block
        if adapter_output is not None:
            hidden_states = hidden_states + adapter_output

        # Sequential adapters apply the adapter to the output of the fc block
        if self.adapter_type in ["sequential", "sequential-fc"]:
            hidden_states = self.fc_adapter(hidden_states)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# monkey-patch OPTDecoderLayer

# This is a hacky way to add adapters to an existing model such that it is still
# correctly initialized with from_pretrained()

# this overwrites the value of the original OPTAttention and OPTDecoderLayer module
# to the ones defined above (with adapter)
transformers.models.opt.modeling_opt.OPTAttention = OPTAttention
transformers.models.opt.modeling_opt.OPTDecoderLayer = OPTDecoderLayer


class OPTWithClassifier(OPTForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.soft_prompt_embeddings = None

        if config.classifier_type == "linear":
            # That's the OPT default (also GPT-2 etc.)
            self.score = nn.Linear(
                config.word_embed_proj_dim, self.num_labels, bias=False)
        elif config.classifier_type == "fully-connected":
            # That's what masked-LMs are using
            self.score = ClassificationHead(config)

        # Add embeddings for soft prompts
        if config.use_soft_prompt:
            self.num_soft_prompt_tokens = config.num_soft_prompt_tokens
            self.soft_prompt_embeddings = torch.nn.Parameter(
                torch.zeros(
                    1,
                    config.num_soft_prompt_tokens,
                    config.word_embed_proj_dim,
                ),
            )

            # initialize soft prompt embeddings randomly
            self.soft_prompt_embeddings.data.normal_(mean=0.0, std=0.001)

            # initialize soft prompt embeddings from embedding matrix
            # TODO(mm): when using deepspeed zero3 we can't do this because embeddings will be an empty tensor
            # embeds = self.model.decoder.embed_tokens.weight
            # self.soft_prompt_embeddings.data = torch.mean(
            #     embeds, 0, keepdim=True).reshape(1, 1, -1).expand(-1, config.num_soft_prompt_tokens, -1)

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
        soft_prompt_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.soft_prompt_embeddings is not None:
            batch_size, seq_length = input_ids.size()
            dim = self.soft_prompt_embeddings.size(-1)

            # replace soft prompt placeholder embeddings with trainable embeddings
            # hacky solution thanks to Vilém Zouhar. Are there other solutions?

            # get word embeddings for inputs
            # positional embeddings will be added during the forward path
            embeds = self.model.decoder.embed_tokens(input_ids)

            # repeat prompt embeddings for every batch
            soft_prompt_embeddings = self.soft_prompt_embeddings.expand(
                batch_size, -1, -1)

            # get the indices to replace (per sequence in the batch)
            _, indices = soft_prompt_mask.nonzero(as_tuple=True)
            indices = indices.reshape(batch_size, -1)

            # indexing magic
            # index every sequence in the batch
            sequence_index = torch.arange(batch_size).to(indices.device)
            # we will flatten the embeddings tensor so we have to shift the indices of the softprompts
            # by the sequence length
            offset = torch.stack(
                [sequence_index for _ in range(self.num_soft_prompt_tokens)], dim=1) * seq_length
            indices = indices + offset

            # replace values
            embeds = embeds.flatten(start_dim=0, end_dim=1)
            embeds[indices] = soft_prompt_embeddings
            embeds = embeds.reshape(
                batch_size, seq_length, dim)  # reshape back

            transformer_outputs = self.model(
                inputs_embeds=embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        else:
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
                # find the last padding token
                sequence_lengths = torch.ne(
                    input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(
            batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
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


class OPTWithLMClassifier(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def _init_weights(self, module):
        super()._init_weights(module)

        if is_deepspeed_zero3_enabled():
            # TODO(mm): also need to do this for soft prompts
            if isinstance(module, LoRAAdapter):
                module._init_weights()
            elif isinstance(module, MLPAdapter):
                module._init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
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

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        # we overwrite this function to support untying the input and output embeddings
        if not self.config.untie_embeddings:
            # default implementation from hf transformers
            if self.config.torchscript:
                output_embeddings.weight = nn.Parameter(
                    input_embeddings.weight.clone())
            else:
                output_embeddings.weight = input_embeddings.weight

            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = nn.functional.pad(
                    output_embeddings.bias.data,
                    (
                        0,
                        output_embeddings.weight.shape[0] -
                        output_embeddings.bias.shape[0],
                    ),
                    "constant",
                    0,
                )
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings
        else:
            # do nothing
            print("**** Untying input and output embeddings ****")
