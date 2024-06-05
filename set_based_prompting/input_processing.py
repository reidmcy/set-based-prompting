import json
import pandas as pd
import typing
import os
import numpy as np
import accelerate
import getpass

import torch
import transformers
import transformers.tokenization_utils_base

from .attention_mask_editing import (
    get_attention_mask_2d_n_options,
    get_position_ids_nopad_n_options,
    get_position_ids_padded_n_options,
    get_attention_mask_2d_recursive,
    get_position_ids_nopad_recursive,
)
from .data_types import OrderIndependentResult, SplitPrompt
from .evalCSQA import eval_accuracy, load_csqa_order_independent_prompts
from .modeling_gpt_attention_refactored import get_2D_attention_accepting_model_gpt
from .modeling_llama_attention import get_2D_attention_accepting_model_llama
from .utils import get_HF_TOKEN


def get_2D_attention_accepting_model(model):
    if isinstance(model, transformers.GPT2LMHeadModel):
        return get_2D_attention_accepting_model_gpt(model)
    elif isinstance(model, transformers.LlamaForCausalLM):
        # print(f"Modify llama model to accept 2D attention mask")
        return get_2D_attention_accepting_model_llama(model)
    elif isinstance(model, transformers.MistralForCausalLM) or isinstance(
        model, transformers.MixtralForCausalLM
    ):
        return get_2D_attention_accepting_model_llama(model)
    else:
        raise ValueError(f"model_type={model.__class__} not recognized")


def get_tokenized_input_prompt(
    tokA: transformers.BatchEncoding,
    tokParallel: typing.List[transformers.BatchEncoding],
    tokD: transformers.BatchEncoding,
) -> typing.Dict[str, transformers.tokenization_utils_base.EncodingFast]:
    tokAll = {"input_ids": tokA["input_ids"], "attention_mask": tokA["attention_mask"]}
    for tokOption in tokParallel:
        tokAll["input_ids"] = torch.cat(
            (tokAll["input_ids"], tokOption["input_ids"][0].unsqueeze(0)),
            dim=1,  # type: ignore
        )  # type: ignore
        tokAll["attention_mask"] = torch.cat(
            (tokAll["attention_mask"], tokOption["attention_mask"][0].unsqueeze(0)),  # type: ignore
            dim=1,
        )  # type: ignore
    tokAll["input_ids"] = torch.cat(
        (tokAll["input_ids"], tokD["input_ids"][0].unsqueeze(0)),
        dim=1,  # type: ignore
    )  # type: ignore
    tokAll["attention_mask"] = torch.cat(
        (tokAll["attention_mask"], tokD["attention_mask"][0].unsqueeze(0)),
        dim=1,  # type: ignore
    )  # type: ignore
    return tokAll


def get_tokenized_input_prompt_recursive(
    tokSequences: typing.List[
        typing.Union[
            transformers.BatchEncoding, typing.List[transformers.BatchEncoding]
        ]
    ],
) -> typing.Dict[str, transformers.tokenization_utils_base.EncodingFast]:
    """
    # flatten tokSequences into a single token object
    """
    tokAll = {
        "input_ids": torch.zeros((1, 0), dtype=torch.int),
        "attention_mask": torch.zeros((1, 0), dtype=torch.int),
    }
    for toks in tokSequences:
        if not isinstance(toks, list):
            tokAll["input_ids"] = torch.cat(
                (tokAll["input_ids"], toks["input_ids"][0].unsqueeze(0)),
                dim=1,  # type: ignore
            )  # type: ignore
            tokAll["attention_mask"] = torch.cat(
                (tokAll["attention_mask"], toks["attention_mask"][0].unsqueeze(0)),
                dim=1,  # type: ignore
            )  # type: ignore
        else:
            for tokOption in toks:
                tokAll["input_ids"] = torch.cat(
                    (tokAll["input_ids"], tokOption["input_ids"][0].unsqueeze(0)),
                    dim=1,  # type: ignore
                )  # type: ignore
                tokAll["attention_mask"] = torch.cat(
                    (
                        tokAll["attention_mask"],
                        tokOption["attention_mask"][0].unsqueeze(0),
                    ),  # type: ignore
                    dim=1,
                )  # type: ignore
    return tokAll


def genOrderIndependentOutputSingleString(
    input_text: str,
    model,
    tokenizer,
    max_new_tokens=10,
    is_order_independent=True,
    reverse_parallel_substrings_order=False,
    torch_device="cpu",
):
    # @param input_text: "A|B;C;D|E" where A,B,C,D,E are strings, and | and ; are delimiters. The number of substrings between | and | is arbitrary.
    # The substrings between | and |, delimited by ";", are processed in parallel by the model
    # if "<|start_2d|>" occurs exactly once in the input_text string
    if input_text.count("<|start_2d|>") < 1:
        raise ValueError(
            f"input_text={input_text} must contain at least one '<|start_2d|>'"
        )
    elif input_text.count("<|start_2d|>") == 1:
        prefix, suffix = input_text.split("<|start_2d|>")
        parallel_substrings, suffix = suffix.split("<|end_2d|>")
        parallel_substrings = parallel_substrings.split("<|split_2d|>")
        print(prefix, parallel_substrings, suffix)
        return genOrderIndependentOutput(
            prefix,
            parallel_substrings,
            suffix,
            model,
            tokenizer,
            max_new_tokens,
            is_order_independent,
            reverse_parallel_substrings_order,
            torch_device,
        )
    else:
        inputStrings = []
        for i, part in enumerate(input_text.split("<|start_2d|>")):
            if i == 0:
                inputStrings.append(part)
            else:
                inputStrings.append(part.split("<|end_2d|>")[0].split("<|split_2d|>"))
                inputStrings.append(part.split("<|end_2d|>")[1])
        return genOrderIndependentOutputRecursive(
            inputStrings,
            model,
            tokenizer,
            max_new_tokens,
            is_order_independent,
            reverse_parallel_substrings_order,
            torch_device,
        )


def genOrderIndependentOutputRecursive(
    inputStrings: typing.List[typing.Union[str, typing.List[str]]],
    model: typing.Union[transformers.PreTrainedModel, transformers.GPT2LMHeadModel],
    tokenizer: typing.Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ],
    max_new_tokens: int = 10,
    is_order_independent: bool = True,
    reverse_parallel_substrings_order: bool = False,
    torch_device: str = "cpu",
    modify_model: bool = True,
    edit_position: bool = True,
    edit_attention: bool = True,
    pad_attention: bool = False,
):
    assert pad_attention == False
    # Tokenize input text
    tokSequences = []
    for i, seq in enumerate(inputStrings):
        if isinstance(seq, list):
            if reverse_parallel_substrings_order:
                seq = seq[::-1]
            tokSequences.append(
                [
                    tokenizer(
                        input_text, return_tensors="pt", add_special_tokens=(i == 0)
                    ).to(torch_device)
                    for input_text in seq
                ]
            )
        else:
            tokSequences.append(
                tokenizer(seq, return_tensors="pt", add_special_tokens=(i == 0)).to(
                    torch_device
                )
            )

    tokAll = get_tokenized_input_prompt_recursive(tokSequences)
    inputPromptLen = len(
        tokenizer.decode(tokAll["input_ids"][0], skip_special_tokens=True)
    )
    if not is_order_independent:
        # Run tests with no attention mask nor position_id intervention
        generated = model.generate(
            tokAll["input_ids"],
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        # Generate a 2D attention mask such that all substrings are processed in parallel
        position_ids = get_position_ids_nopad_recursive(tokSequences)
        if not edit_position:
            position_ids = torch.arange(0, len(tokAll["input_ids"])).unsqueeze(0)
        position_ids = position_ids.to(torch_device)
        attention_mask_2d = get_attention_mask_2d_recursive(tokSequences, tokAll).to(
            torch_device
        )
        if not edit_attention:
            attention_mask_2d = None
        # Modify the given model to accept a 2D attention mask as input
        if modify_model:
            model = get_2D_attention_accepting_model(model)
        generated = model.generate(
            tokAll["input_ids"],
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask_2d,
            position_ids=position_ids,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)[
        inputPromptLen:
    ]
    # Return output of model generation, and generated text
    return generated, text


def genOrderIndependentOutput(
    prefix: str,
    parallel_substrings: typing.List[str],
    suffix: str,
    model: typing.Union[transformers.PreTrainedModel, transformers.GPT2LMHeadModel],
    tokenizer: typing.Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ],
    max_new_tokens: int = 10,
    is_order_independent: bool = True,
    reverse_parallel_substrings_order: bool = False,
    torch_device: str = "cpu",
    modify_model: bool = True,
    edit_position: bool = True,
    edit_attention: bool = True,
    pad_attention: bool = False,
):
    model_device = model.device
    # Tokenize input text
    tokA = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
        return_token_type_ids=False,
    ).to(torch_device)
    tokD = tokenizer(
        suffix,
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to(torch_device)
    if reverse_parallel_substrings_order:
        parallel_substrings = parallel_substrings[::-1]
    tokParallel = [
        tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(
            torch_device
        )
        for input_text in parallel_substrings
    ]

    tokAll = get_tokenized_input_prompt(tokA, tokParallel, tokD)
    assert len(tokA["attention_mask"][0]) + sum(
        [len(tokOption["attention_mask"][0]) for tokOption in tokParallel]
    ) + len(tokD["attention_mask"][0]) == len(tokAll["attention_mask"][0])
    s = len(tokAll["input_ids"][0])
    # inputPromptLen=len(prefix)+sum([len(s) for s in parallel_substrings])+len(suffix)
    inputPromptLen = len(
        tokenizer.decode(tokAll["input_ids"][0], skip_special_tokens=True)
    )

    if not is_order_independent:
        # Run tests with no attention mask nor position_id intervention
        generated = model.generate(
            tokAll["input_ids"],
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        # Pad all parallel substrings to the same length (if pad_attention=True), then generate a 2D attention mask such that all substrings are processed in parallel
        get_position_ids = (
            get_position_ids_padded_n_options
            if pad_attention
            else get_position_ids_nopad_n_options
        )
        position_ids, tokParallel, tokAll = get_position_ids(
            tokA, tokParallel, tokD, tokenizer=tokenizer
        )
        if not edit_position:
            position_ids = torch.arange(0, len(tokAll["input_ids"])).unsqueeze(0)
        position_ids = position_ids.to(torch_device)
        attention_mask_2d = get_attention_mask_2d_n_options(
            tokA, tokParallel, tokD, tokAll
        ).to(torch_device)
        if not edit_attention:
            attention_mask_2d = None
        # Modify the given model to accept a 2D attention mask as input
        if modify_model:
            model = get_2D_attention_accepting_model(model)
        generated = model.generate(
            tokAll["input_ids"],
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask_2d,
            position_ids=position_ids,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)[
        inputPromptLen:
    ]
    # Return output of model generation, and generated text
    return generated, text


def load_model(
    model_name, torch_device: typing.Literal["auto", "cpu", "cuda"] = "auto", device_id: typing.Optional[int] = None
) -> typing.Tuple[
    typing.Union[transformers.PreTrainedModel, transformers.GPT2LMHeadModel],
    typing.Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ],
]:
    HF_TOKEN = get_HF_TOKEN()
    cache_dir = "/n/holylabs/LABS/dwork_lab/Everyone/cache/transformers"
    if getpass.getuser() == "rmcilroyyoung":
        cache_dir = "/n/holylabs/LABS/dwork_lab/Lab/reidmcy/hf_models"
    if not os.path.exists(cache_dir):
        cache_dir = None

    if device_id is not None:
        assert torch_device == "cuda", f"device_id={device_id} requires torch_device='cuda'"
        map_device = f"cuda:{device_id}"
    else:
        map_device = torch_device

    if model_name == "gpt2":
        model = transformers.GPT2LMHeadModel.from_pretrained(
            "gpt2", device_map=map_device
        )
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:  # e.g model_name == "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=HF_TOKEN,
            use_fast=True,
            cache_dir=cache_dir,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=HF_TOKEN,
            device_map=map_device,
            cache_dir=cache_dir,
        )
    return model, tokenizer  # type: ignore


def get_model_outputs(
    model,
    tokenizer,
    prompts_parallel,
    is_order_independent,
    torch_device,
    reverse_parallel_substrings_order,
    edit_position,
    edit_attention,
    pad_attention,
    max_new_tokens,
):
    modelOutputs = []
    for prefix, parallel_substrings, suffix in prompts_parallel:
        g, t = genOrderIndependentOutput(
            prefix,
            parallel_substrings,
            suffix,
            model,
            tokenizer,
            is_order_independent=is_order_independent,
            torch_device=torch_device,
            reverse_parallel_substrings_order=reverse_parallel_substrings_order,
            edit_position=edit_position,
            edit_attention=edit_attention,
            pad_attention=pad_attention,
            max_new_tokens=max_new_tokens,
        )
        modelOutputs.append(t)
    return modelOutputs


def save_model_outputs(modelOutputs, model_name, is_order_independent):
    suffix = "_order_independent" if is_order_independent else ""
    with open(f"data/{model_name}_outputs{suffix}.json", "w") as f:
        json.dump(modelOutputs, f)


def eval_model_csqa_accuracy(
    model,
    tokenizer,
    is_order_independent,
    torch_device,
    num_samples,
    model_name=None,
    reverse_parallel_substrings_order=False,
    edit_position=True,
    edit_attention=True,
    pad_attention=False,
    max_new_tokens=10,
    suffix=None,
):
    prompts_parallel, labels, incorrect_answers = load_csqa_order_independent_prompts(
        suffix
    )
    modelOutputs = get_model_outputs(
        model,
        tokenizer,
        prompts_parallel[:num_samples],
        is_order_independent,
        torch_device,
        reverse_parallel_substrings_order,
        edit_position,
        edit_attention,
        pad_attention,
        max_new_tokens,
    )
    eval_accuracy(modelOutputs, labels[:num_samples], incorrect_answers[:num_samples])
    if model_name is None:
        model_name = model.__class__
    save_model_outputs(modelOutputs, model_name, is_order_independent)


def gen_file_outputs(
    model: typing.Union[transformers.PreTrainedModel, transformers.GPT2LMHeadModel],
    tokenizer: typing.Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ],
    torch_device: str,
    input_file_name: str,
    output_file_name: str,
    max_new_tokens: int,
):
    input_file = open(input_file_name)
    input_text_prompts = json.load(input_file)
    outputs = []
    for prompt in input_text_prompts:
        input_text = prompt["prompt"]
        textOD = genOrderIndependentOutputSingleString(
            input_text,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            is_order_independent=False,
            reverse_parallel_substrings_order=False,
            torch_device=torch_device,
        )[1]
        textOD_reverse = genOrderIndependentOutputSingleString(
            input_text,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            is_order_independent=False,
            reverse_parallel_substrings_order=True,
            torch_device=torch_device,
        )[1]
        outputs.append(
            {
                "prompt": input_text,
                "output_order_dependent": textOD,
                "output_order_dependent_rev": textOD_reverse,
            }
        )
    for i, prompt in enumerate(input_text_prompts):
        input_text = prompt["prompt"]
        textOI = genOrderIndependentOutputSingleString(
            input_text,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            is_order_independent=True,
            reverse_parallel_substrings_order=False,
            torch_device=torch_device,
        )[1]
        outputs[i]["output_order_independent"] = textOI
    with open(output_file_name, "w") as f:
        json.dump(outputs, f)


def calc_perplexity(input_ids, attention_mask, model):
    # Compute the perplexity of the model for the given input sequence
    # Perplexity as defined here https://huggingface.co/docs/transformers/perplexity
    out = model(
        input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=True
    )
    loss = out.loss.item()
    perplexity = torch.exp(out.loss).item()
    return loss, perplexity


def calc_perplexity_option(input_ids, attention_mask, option, model, tokenizer):
    """
    Compute the perplexity of the sequence defined by input_ids and the tokenized option string
    """
    label_input_ids = tokenizer(option, return_tensors="pt")["input_ids"]
    nTokLabel = label_input_ids.shape[1]
    label_input_ids = torch.cat(
        (input_ids, label_input_ids),
        dim=1,  # type: ignore
    )  # type: ignore
    # Add nTokLabel ones (in triangular format) to the 2D attention mask
    label_attention_mask = None
    if attention_mask is not None:
        n = attention_mask.shape[-1]
        label_attention_mask = torch.cat(
            (
                attention_mask,
                torch.zeros(
                    (attention_mask.shape[0], 1, attention_mask.shape[-1], nTokLabel),
                    dtype=torch.int,
                ),
            ),
            dim=3,
        )
        n = torch.ones(1, 1, nTokLabel, n + nTokLabel)
        n[0, 0, :, -nTokLabel:] = torch.tril(torch.ones((1, 1, nTokLabel, nTokLabel)))
        label_attention_mask = torch.cat((label_attention_mask, n), dim=2)
    option_loss, option_perplexity = calc_perplexity(
        label_input_ids, label_attention_mask, model
    )
    return option_perplexity


def calc_options_perplexity_dict(
        input_ids:torch.Tensor,
        attention_mask:torch.Tensor,
          metadata:dict, model: typing.Union[transformers.PreTrainedModel, transformers.GPT2LMHeadModel],
    tokenizer: typing.Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ],) -> typing.Dict[str, float]:
    label, incorrect_answers = metadata["label"], metadata["incorrect_answers"]
    # Compute perplexity for both the correct and incorrect answers
    prompt_loss, prompt_perplexity = calc_perplexity(input_ids, attention_mask, model)
    option_perplexities = {"prompt": prompt_perplexity}
    for option in [label] + incorrect_answers:
        option_perplexity = calc_perplexity_option(
            input_ids, attention_mask, option, model, tokenizer
        )
        option_perplexities[option] = option_perplexity
    return option_perplexities


def order_independent_query(
    prefix: str,
    parallel_substrings: typing.List[str],
    suffix: str,
    model: typing.Union[transformers.PreTrainedModel, transformers.GPT2LMHeadModel],
    tokenizer: typing.Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ],
    max_new_tokens: int = 10,
    is_order_independent: bool = True,
    #torch_device: str = "cpu",
    modify_model: bool = True,
    edit_position: bool = True,
    edit_attention: bool = True,
    pad_attention: bool = False,
    include_perplexity: bool = False,
    metadata: typing.Union[typing.Dict[str, typing.Any], None] = None,

) -> OrderIndependentResult:
    model_device = model.device
    # Tokenize input text
    tokA = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
        return_token_type_ids=False,
    ).to(model_device)
    tokD = tokenizer(
        suffix,
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to(model_device)
    tokParallel = [
        tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(
            model_device
        )
        for input_text in parallel_substrings
    ]

    tokAll = get_tokenized_input_prompt(tokA, tokParallel, tokD)
    assert len(tokA["attention_mask"][0]) + sum(
        [len(tokOption["attention_mask"][0]) for tokOption in tokParallel]
    ) + len(tokD["attention_mask"][0]) == len(tokAll["attention_mask"][0])
    s = len(tokAll["input_ids"][0])
    # inputPromptLen=len(prefix)+sum([len(s) for s in parallel_substrings])+len(suffix)
    inputPromptLen = len(
        tokenizer.decode(tokAll["input_ids"][0], skip_special_tokens=True)
    )
    input_ids = tokAll["input_ids"]
    attention_mask_2d = None

    if not is_order_independent:
        # Run tests with no attention mask nor position_id intervention
        edit_attention, edit_position = False, False
        generated = model.generate(
            tokAll["input_ids"],
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        # Pad all parallel substrings to the same length (if pad_attention=True), then generate a 2D attention mask such that all substrings are processed in parallel
        get_position_ids = (
            get_position_ids_padded_n_options
            if pad_attention
            else get_position_ids_nopad_n_options
        )
        position_ids, tokParallel, tokAll = get_position_ids(
            tokA, tokParallel, tokD, tokenizer=tokenizer
        )
        if not edit_position:
            position_ids = torch.arange(0, len(tokAll["input_ids"][0])).unsqueeze(0)
        position_ids = position_ids.to(model_device)
        attention_mask_2d = get_attention_mask_2d_n_options(
            tokA, tokParallel, tokD, tokAll, device=model_device
        )
        if not edit_attention:
            attention_mask_2d = None
        # Modify the given model to accept a 2D attention mask as input
        if modify_model:
            model = get_2D_attention_accepting_model(model)
        generated = model.generate(
            tokAll["input_ids"],
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask_2d,
            position_ids=position_ids,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    assert isinstance(
        generated, transformers.utils.ModelOutput
    ), f"generated={generated} is not a ModelOutput, but a {type(generated)}"
    text = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)[
        inputPromptLen:
    ]
    is_correct_answer, correct_answer_prob = None, None
    raw_output_contains_correct_answer_only = None
    label_scores = {}
    label_perplexities = {}
    if metadata is not None and "label" in metadata and "incorrect_answers" in metadata:
        # Compute the probability that the label, rather than any of the incorrect answers, is selected
        class_ids = torch.LongTensor(
            tokenizer(
                [metadata["label"]] + metadata["incorrect_answers"], padding=True
            ).input_ids
        )
        # Generate the logits for each token in the generated output sequence.
        # `scores` has size [batch, seq_length, vocab_size]
        scores = torch.stack(generated.scores, dim=1).to("cpu")  # type: ignore
        # We don't care about the logits for the special tokens
        scores[:, :, tokenizer.all_special_ids] = torch.nan

        expanded_class_ids = class_ids.T.expand(1, -1, -1)

        # pad scores with nans to be at least as long as the class_ids
        if scores.shape[1] < expanded_class_ids.shape[1]:
            scores = torch.cat(
                [
                    scores,
                    torch.full(
                        (scores.shape[0], expanded_class_ids.shape[1] - scores.shape[1], scores.shape[2]),
                        float("nan"),
                    ),
                ],
                dim=1,
            )

        score_of_labels = scores.gather(dim=2, index=expanded_class_ids)
        # For each output sequence, compute the mean across the non-special tokens, then softmax across output sequences
        mean_logit_scores_per_label = score_of_labels.nanmean(dim=1)
        # Create dictionary of key=label, value=mean score across logits in each output option sequence
        label_scores = {
            k: float(v)
            for k, v in zip(
                [metadata["label"]] + metadata["incorrect_answers"],
                list(mean_logit_scores_per_label[0].numpy()),
            )
        }
        probabilities = mean_logit_scores_per_label.softmax(1)
        max_probability_index = torch.argmax(probabilities, dim=1)[0]
        is_correct_answer = bool((max_probability_index == 0).item())
        correct_answer_prob = probabilities[0, 0].item()
        raw_output_contains_correct_answer_only = metadata[
            "label"
        ] in generated and all(
            [o not in generated for o in metadata["incorrect_answers"]]
        )
        label_perplexities = {}
        if include_perplexity:
            label_perplexities = calc_options_perplexity_dict(
                tokAll["input_ids"].cpu().detach(),
                attention_mask_2d if is_order_independent else None,
                metadata,
                model,
                tokenizer,
            )


    # Return output of model generation, and generated text
    return OrderIndependentResult(
        prompt=SplitPrompt.from_split_text(prefix, parallel_substrings, suffix),
        model=model.__class__.__name__,
        max_new_tokens=max_new_tokens,
        order_independent_output=is_order_independent,
        pad_attention=pad_attention,
        text_output=text,
        raw_output=generated,
        input_ids=input_ids,
        is_correct_answer=is_correct_answer,
        correct_answer_prob=correct_answer_prob,
        raw_output_contains_correct_answer_only=raw_output_contains_correct_answer_only,
        edit_position=edit_position,
        edit_attention=edit_attention,
        label_scores=label_scores,
        label_perplexities=label_perplexities,
    )


def record_accuracy(
    output_file_name: str,
    accuracy_file: str,
    accuracy_scoring_mode: typing.Union[
        typing.Literal["pct_raw_output_contains_correct_answer_only"],
        typing.Literal["max_token_prob"],
    ],
) -> None:
    """
    @param output_file_name: name of file containing model outputs
    @param accuracy_file: path to file to record experiment results
    @param accuracy_scoring_mode: "pct_raw_output_contains_correct_answer_only" or "max_token_prob"
    """
    assert accuracy_scoring_mode in [
        "pct_raw_output_contains_correct_answer_only",
        "max_token_prob",
    ]
    accuracy_key = (
        "raw_output_contains_correct_answer_only"
        if accuracy_scoring_mode == "pct_raw_output_contains_correct_answer_only"
        else "is_correct_answer"
    )
    df = pd.DataFrame(
        columns=[
            "model_name",
            "max_new_tokens",
            "output_filename",
            "output_type",
            "accuracy",
            "n_samples",
            "accuracy_scoring_mode",
        ]
    )
    if not os.path.isfile(accuracy_file):
        print(f"Create file {accuracy_file} to store accuracy records")
        df.to_csv(accuracy_file, index=False)

    records = []
    with open(output_file_name) as f:
        for line in f:
            records.append(json.loads(line))

    for output_type in records[0]["responses"]:
        for model_name in np.unique([r["model"] for r in records]):
            # record accuracy for each output_type
            print(f"Eval for output_type={output_type}")
            max_new_tokens = records[0]["responses"][output_type]["max_new_tokens"]
            for i, r in enumerate(records):
                if accuracy_key not in r["responses"][output_type]:
                    print(
                        f"{accuracy_key} not in row {i} with keys",
                        r["responses"][output_type].keys(),
                    )
            accuracy = np.mean(
                [
                    r["responses"][output_type][accuracy_key]
                    for r in records
                    if r["model"] == model_name
                ]
            )
            pd.DataFrame.from_dict(
                {
                    "model_name": [model_name],
                    "max_new_tokens": [max_new_tokens],
                    "output_file_name": [output_file_name],
                    "output_type": [output_type],
                    "accuracy": [accuracy],
                    "n_samples": [
                        len([r for r in records if r["model"] == model_name])
                    ],
                    "accuracy_scoring_mode": [accuracy_scoring_mode],
                }
            ).to_csv(accuracy_file, mode="a", index=False, header=False)
