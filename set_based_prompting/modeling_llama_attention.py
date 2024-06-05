import math
import types
import warnings
import typing

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.utils import ModelOutput

VERBOSE = False


def forwardLlamaAttention(
    self,
    hidden_states: torch.Tensor,
    attention_mask: typing.Optional[torch.Tensor] = None,
    position_ids: typing.Optional[torch.LongTensor] = None,
    past_key_value: typing.Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor], typing.Optional[typing.Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()
    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, None
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def forwardLlamaModel(
    self,
    input_ids: typing.Union[torch.LongTensor,None] = None,
    attention_mask: typing.Optional[torch.Tensor] = None,
    position_ids: typing.Optional[torch.LongTensor] = None,
    past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None,
    inputs_embeds: typing.Optional[torch.FloatTensor] = None,
    use_cache: typing.Optional[bool] = None,
    output_attentions: typing.Optional[bool] = None,
    output_hidden_states: typing.Optional[bool] = None,
    return_dict: typing.Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if False:  # self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif False:  # self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
        if VERBOSE:
            print(f"4D attention mask {attention_mask.shape}")

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def forwardLlamaForCausalLM(
    self,
    input_ids: typing.Union[torch.LongTensor,None] = None,
    attention_mask: typing.Optional[torch.Tensor] = None,
    position_ids: typing.Optional[torch.LongTensor] = None,
    past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None,
    inputs_embeds: typing.Optional[torch.FloatTensor] = None,
    labels: typing.Optional[torch.LongTensor] = None,
    use_cache: typing.Optional[bool] = None,
    output_attentions: typing.Optional[bool] = None,
    output_hidden_states: typing.Optional[bool] = None,
    return_dict: typing.Optional[bool] = None,
):
    # NOTE: print debugging info
    if VERBOSE:
        print(f"Position IDs={position_ids}")
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

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

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(
            self.vocab_size // self.config.pretraining_tp, dim=0
        )
        logits = [
            F.linear(hidden_states, lm_head_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

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


def expand_attention_mask(attention_mask):
    """
    Expands an attention mask from shape (bsz, 1, tgt_seq_len, src_seq_len) to
    (bsz, 1, tgt_seq_len+1, src_seq_len+1) with the last new column as zeros and the last new row fully as ones.

    Parameters:
    - attention_mask: A torch.Tensor of shape (bsz, 1, tgt_seq_len, src_seq_len)

    Returns:
    - A torch.Tensor of shape (bsz, 1, tgt_seq_len+1, src_seq_len+1)
    """
    bsz, _, tgt_seq_len, src_seq_len = attention_mask.shape
    new_attention_mask = torch.zeros(
        bsz,
        1,
        tgt_seq_len + 1,
        src_seq_len + 1,
        device=attention_mask.device,
        dtype=torch.int32,
    )

    # Copy the existing attention mask to the new mask's top left
    new_attention_mask[:, :, :tgt_seq_len, :src_seq_len] = attention_mask

    # Set the last row fully to ones
    new_attention_mask[:, :, -1, :] = 1

    return new_attention_mask


def _update_model_kwargs_for_generation(
    self,
    outputs: ModelOutput,
    model_kwargs:typing.Dict[str, typing.Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
) -> typing.Dict[str, typing.Any]:

    # update past_key_values
    model_kwargs["past_key_values"] = self._extract_past_from_model_output(
        outputs, standardize_cache_format=standardize_cache_format
    )
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat(
            [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
        )

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 4:
                # NOTE: this if-statement loop can be refactored to avoid redundant code
                next_position_id = (
                    model_kwargs["position_ids"][0][-1] + 1
                    if "position_ids" in model_kwargs
                    else attention_mask.shape[-1]
                )

                # given an attention_mask of shape (bsz, 1, tgt_seq_len, src_seq_len) in model kwargs, generate
                # new 2D attention_mask of ones with shape (bsz, src_seq_len+1) for subsequent forward() calls
                # EDIT:
                # Convert 4D mask to 2D mask, while recording any padding tokens
                model_kwargs["attention_mask"] = (
                    torch.any(attention_mask[0][0] != 0, dim=0)
                    .to(attention_mask.dtype)
                    .unsqueeze(0)
                )
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        model_kwargs["attention_mask"],
                        model_kwargs["attention_mask"].new_ones(
                            (model_kwargs["attention_mask"].shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )  # Add a new 1 to the end for the newly generated token
                model_kwargs["position_ids"] = torch.tensor([[next_position_id]]).to(
                    model_kwargs["position_ids"].device
                )
            else:
                # Extend the length of the attention mask by 1 to reflect the fact that we have generated one additional token
                assert len(attention_mask.shape) == 2
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
                if "position_ids" in model_kwargs:
                    next_position_id = (
                        model_kwargs["position_ids"][0][-1] + 1
                        if "position_ids" in model_kwargs
                        else attention_mask.shape[-1]
                    )
                    model_kwargs["position_ids"] = torch.tensor(
                        [[next_position_id]]
                    ).to(model_kwargs["position_ids"].device)
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [
                    decoder_attention_mask,
                    decoder_attention_mask.new_ones(
                        (decoder_attention_mask.shape[0], 1)
                    ),
                ],
                dim=-1,
            )

    return model_kwargs

# Used for transformers version git+https://github.com/huggingface/transformers@1c31b7aa3bb4e7ef24c77596d2a76f45a770159f
def get_2D_attention_accepting_model_llama(model):
    model._update_model_kwargs_for_generation = types.MethodType(
        _update_model_kwargs_for_generation, model
    )
    return model
