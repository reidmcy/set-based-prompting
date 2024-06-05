import argparse
import json

import transformers

from .utils import get_HF_TOKEN


def load_csqa_prompts():
    with open("prompts_csqa.json", "r") as f:
        prompts = json.load(f)
    with open("labels_csqa.json", "r") as f:
        labels = json.load(f)
    return prompts, labels


def load_csqa_order_independent_prompts(suffix=None):
    prompts, labels = load_csqa_prompts()
    prompts_parallel = []
    suffix = " Answer: " if suffix is None else suffix
    for p in prompts:
        prefix = p.split("\n")[0] + "\n"
        parallel = p.split("\n- ")[1:]
        parallel = ["\n- " + s for s in parallel]
        prompts_parallel.append([prefix, parallel, " Answer: "])
    incorrect_answers = []
    for p, label in zip(prompts, labels):
        options = p.split("\n- ")[1:]
        incorrect_answers.append([o for o in options if o != label])
    return prompts_parallel, labels, incorrect_answers


def eval_accuracy(modelOutputs, labels, incorrect_answers):
    correct = 0
    for i in range(len(modelOutputs)):
        if labels[i] in modelOutputs[i] and all(
            [ia not in modelOutputs[i] for ia in incorrect_answers[i]]
        ):
            correct += 1
    print(f"{correct}/{len(modelOutputs)}={correct/len(modelOutputs)} answers correct")
    return correct / len(modelOutputs)


def load_model(model_name, torch_device="cpu"):
    HF_TOKEN = get_HF_TOKEN()

    if model_name == "gpt2":
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_name == "llama":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=HF_TOKEN, use_fast=True
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=HF_TOKEN,
            device_map=torch_device,
        )
    return model, tokenizer


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


def main(model_name, is_order_independent, torch_device, num_samples):
    """
    Save model outputs to file and report accuracy for a given model
    """
    print(
        f"Generate model outputs for {num_samples} samples using model {model_name} on torch device {torch_device} with order independent={is_order_independent}"
    )
    model, tokenizer = load_model(model_name, torch_device)
    eval_model_accuracy(
        model, tokenizer, is_order_independent, torch_device, num_samples, model_name
    )


#  python evalCSQA.py --model-name gpt2 --torch-device cpu --num-samples 10
#  python evalCSQA.py --model-name gpt2 --torch-device cpu --num-samples 10 --is-order-independent
if __name__ == "__main__":
    # take in flags from command line, model_name string, is_order_independent boolean, torch_device string, and num_samples int
    parser = argparse.ArgumentParser(
        description="Run attention mask editing tests on a given model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="name of the model to test",
        choices=["gpt2", "llama"],
        default="gpt2",
    )
    parser.add_argument(
        "--is-order-independent",
        type=bool,
        help="whether to run order independent tests",
        default=False,
    )
    parser.add_argument(
        "--torch-device", type=str, help="device to run tests on", default="cpu"
    )
    parser.add_argument(
        "--num-samples", type=int, help="number of samples to run", default=9741
    )

    args = parser.parse_args()
    model_name = args.model_name
    is_order_independent = args.is_order_independent
    print(f"is_order_independent={is_order_independent}")
    torch_device = args.torch_device
    num_samples = args.num_samples
    main(model_name, is_order_independent, torch_device, num_samples)
