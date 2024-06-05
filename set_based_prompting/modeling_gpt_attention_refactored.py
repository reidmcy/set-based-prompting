import types
import typing

import torch
import transformers.utils
from torch import nn

VERBOSE = False

def _attn(self, query, key, value, attention_mask=None, head_mask=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )

    # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        # Note: this is where the 2D causal mask is generated, we override this to replace the causal mask with attention_mask_2D, if it is given
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )

        # Causal_mask has shape (1, 1, query_length, key_length)
        if (
            attention_mask is not None
            and attention_mask.shape[-1] == causal_mask.shape[-1] ** 2
        ):
            causal_mask = attention_mask.view(
                causal_mask.shape
            )  # .to(causal_mask.dtype)
            attn_weights = attn_weights + causal_mask
        else:
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            attn_weights = torch.where(
                causal_mask, attn_weights.to(attn_weights.dtype), mask_value
            )

    if attention_mask is not None and len(attention_mask.shape) == 2:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


def _update_model_kwargs_for_generation(
    self,
    outputs: transformers.utils.ModelOutput,
    model_kwargs: typing.Dict[str, typing.Any],
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
                # given an attention_mask of shape (bsz, 1, tgt_seq_len, src_seq_len) in model kwargs
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
            else:
                assert len(attention_mask.shape) == 2
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
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


def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past_key_values:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    elif (
        attention_mask is not None
        and position_ids is not None
        and len(attention_mask.shape) == 4
    ):
        # EDIT: Don't set position_ids = None if the user passed in both position_ids and a 2D attention_mask
        assert position_ids.shape[-1] == attention_mask.shape[-1]
    else:
        position_ids = None

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    )
    return model_inputs


def get_2D_attention_accepting_model_gpt(model):
    model._update_model_kwargs_for_generation = types.MethodType(
        _update_model_kwargs_for_generation, model
    )
    model.prepare_inputs_for_generation = types.MethodType(
        prepare_inputs_for_generation, model
    )
    for hidden_layer in range(len(model.transformer.h)):
        model.transformer.h[hidden_layer].attn._attn = types.MethodType(
            _attn, model.transformer.h[hidden_layer].attn
        )
    return model
