import argparse
import json

import tqdm
import enum
import itertools
import numpy as np

import os.path
import shutil
import typing
import order_independent_llm

class AttentionVariation(enum.Enum):
    normal = 'normal'
    normal_padded = 'normal_padded'
    normal_reversed = 'normal_reversed'
    order_independent = 'order_independent'
    order_independent_padded = 'order_independent_padded'
    only_parallel_attention = 'only_parallel_attention'
    only_parallel_attention_reversed = 'only_parallel_attention_reversed'
    only_parallel_position = 'only_parallel_position'
    only_parallel_position_reversed = 'only_parallel_position_reversed'
    normal_permuted = 'normal_permuted'



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run attention mask editing tests on a given model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="name of the model to test",
        choices=[
            "gpt2",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/llama-2-70b-chat-hf",
            "WizardLM/WizardLM-7B-V1.0",
            "lmsys/vicuna-7b-v1.5",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Meta-Llama-3-70B",
            "meta-llama/Meta-Llama-3-70B-Instruct",
        ],
        default="gpt2",
    )

    parser.add_argument(
        "--torch-device",
        type=str,
        help="device to run tests on",
        default="auto",
        choices=["cuda", "cpu", "auto"],
    )

    parser.add_argument(
        "--cuda-device-id",
        type=int,
        help="cuda device id to run tests on",
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="max number of tokens to generate given each prompt",
        default=10,
    )
    parser.add_argument(
        "--infile",
        type=str,
        help="path/to/file containing formatted input prompts",
        required=True,
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="path/to/file that should contain generated outputs",
        required=True,
    )
    parser.add_argument(
        "--pad-attention",
        help="whether to pad all parallel substrings to the same length",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--add-only-parallel-attention",
        help="whether to generate outputs that modify the attention mask but not positional encoding",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--add-only-parallel-position",
        help="whether to generate outputs that modify the positional encoding but not attention mask",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--record-accuracy",
        help="whether to evaluate and record the accuracy of model output answers",
        required=False,
        action="store_true",
    )
    parser.add_argument(  # only used if --record-accuracy
        "--accuracy-file",
        type=str,
        help="path/to/csv to record experiment accuracy results",
        required=False,
        default="records.csv",
    )
    parser.add_argument(
        "--num-normal-ordering-permutations",
        type=int,
        default=None,
        required=False,
        help="If set, will generate normal output for X! orderings of the parallel substrings",
    )

    parser.add_argument(
        "--temp_file",
        help="Write intermediate results to a temporary file, prevents overwriting of output file if already created.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--append-temp-file",
        help="Append to temporary file, starting from the last successful prompt, instead of start of input file.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="top k tokens to consider for token probabilities",
        default=100,
    )
    parser.add_argument(
        "--include-probs",
        help="Include probabilities in output",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    model_name: str = args.model_name
    torch_device: typing.Literal["auto", "cpu", "cuda"] = args.torch_device
    cuda_device_id: typing.Union[None, int] = args.cuda_device_id
    input_file_name: str = args.infile
    output_file_name: str = args.outfile
    max_new_tokens: int = args.max_new_tokens
    pad_attention: bool = args.pad_attention
    only_parallel_attention: bool = args.add_only_parallel_attention
    only_parallel_position: bool = args.add_only_parallel_position
    record_accuracy: bool = args.record_accuracy
    accuracy_file: str = args.accuracy_file
    temp_file: bool = args.temp_file
    append_temp_file:bool = args.append_temp_file
    top_k: int = args.top_k
    include_probs:bool = args.include_probs
    num_normal_ordering_permutations: typing.Union[None,int] = args.num_normal_ordering_permutations

    if cuda_device_id is not None:
        assert torch_device == "cuda", "cuda_device_id is only used when torch_device is 'cuda'"

    conditions = [
        AttentionVariation.normal,
        AttentionVariation.normal_reversed,
        AttentionVariation.order_independent,
    ]
    if pad_attention:
        conditions.append(AttentionVariation.normal_padded)
        conditions.append(AttentionVariation.order_independent_padded)
    if only_parallel_attention:
        conditions.append(AttentionVariation.only_parallel_attention)
        conditions.append(AttentionVariation.only_parallel_attention_reversed)
    if only_parallel_position:
        conditions.append(AttentionVariation.only_parallel_position)
        conditions.append(AttentionVariation.only_parallel_position_reversed)
    parallel_orderings:typing.List[typing.Union[None,typing.List[int]]] = [None]*len(conditions)
    if num_normal_ordering_permutations  is not None:
        order_independent_llm.print_with_timestamp(
            f"Warning: num_normal_ordering_permutations is set, will generate normal output for {num_normal_ordering_permutations}! orderings of the parallel substrings"
        )
        # We want to generate output for num_normal_ordering_permutations! normal orderings
        perm_list = list(itertools.permutations(list(range(0,num_normal_ordering_permutations))))
        for i, perm in enumerate(perm_list, start=1):
            parallel_orderings.append(list(perm))
            conditions.append(AttentionVariation.normal_permuted)

    run_experiment(
        model_name,
        torch_device,
        cuda_device_id,
        input_file_name,
        output_file_name,
        max_new_tokens,
        conditions,
        parallel_orderings,
        record_accuracy,
        accuracy_file,
        temp_file,
        append_temp_file,
        top_k,
        include_probs
    )


def run_experiment(
    model_name : str,
    torch_device : typing.Literal["auto", "cpu", "cuda"],
    cuda_device_id : typing.Union[None, int],
    input_file_name : str,
    output_file_name : str,
    max_new_tokens : int,
    conditions: typing.List[AttentionVariation],
    parallel_orderings: typing.List[typing.Union[None, typing.List[int]]],
    record_accuracy : bool,
    accuracy_file : str,
    temp_file : bool,
    append_temp_file : bool,
    top_k:int,
    include_probs:bool
) -> None:
    order_independent_llm.print_with_timestamp(
        f"Running on {input_file_name} with model {model_name} and torch device {torch_device} with max_new_tokens {max_new_tokens}"
    )

    order_independent_llm.print_with_timestamp(
        f"Running with conditions: {', '.join([c.value for c in conditions])}"
    )

    original_output_file_name = output_file_name

    if temp_file:
        if os.path.exists(output_file_name):
            order_independent_llm.print_with_timestamp(
                f"Output file {output_file_name} already exists, skipping..."
            )
            return
        else:
            output_file_name = output_file_name + "_tmp"

    order_independent_llm.print_with_timestamp("Loading model and tokenizer...")

    model, tokenizer = order_independent_llm.load_model(model_name, torch_device, cuda_device_id)

    prompts = order_independent_llm.SplitPrompt.from_json_file(input_file_name)

    open_mode = 'wt'
    if temp_file and append_temp_file:
        prompts = order_independent_llm.filter_prompts(output_file_name, prompts)
        open_mode = 'at'

    order_independent_llm.print_with_timestamp(
        f"Found {len(prompts)} input prompts, running..."
    )

    order_independent_llm.print_with_timestamp(f"Writing results to {original_output_file_name}...")

    with open(output_file_name, open_mode) as fout:
        for prompt in tqdm.tqdm(
            prompts, desc=f"[{order_independent_llm.get_timestamp_str()}] Running {model_name} {os.path.basename(input_file_name).split('.')[0]}", total=len(prompts), leave=True
        ):
            prefix, parallel_substrings, suffix = prompt.gen_split_text()

            responses = {}
            for i,condition in enumerate(conditions):
                resp = generate_response(
                    condition,
                    prefix,
                    parallel_substrings,
                    suffix,
                    model,
                    tokenizer,
                    max_new_tokens,
                    parallel_ordering=parallel_orderings[i],
                    metadata=prompt.metadata,
                )
                key=condition.value
                if condition==AttentionVariation.normal_permuted:
                    ordering = parallel_orderings[i]
                    if ordering is None:
                        key+="_normal"
                    else:
                        key+="_"+''.join(map(str, ordering))
                responses[key] = resp.to_json_dict()
                if include_probs:
                    responses[key]['probs'] = resp.get_per_token_probs(tokenizer, top_k).to_json_dict()

            prompt_output = {
                "prompt": prompt.text,
                "model": model_name,
                "prompt_metadata": prompt.metadata,
                "responses": responses,
            }

            json.dump(
                prompt_output,
                fout,
            )
            fout.write("\n")
            fout.flush()

    if temp_file:
        shutil.move(output_file_name, original_output_file_name)

    if record_accuracy:
        order_independent_llm.print_with_timestamp(
            f"Recording accuracy of model outputs, accuracy records saved to {accuracy_file}"
        )
        order_independent_llm.record_accuracy(
            output_file_name, accuracy_file, "pct_raw_output_contains_correct_answer_only"
        )

    order_independent_llm.print_with_timestamp(
        f"Finished running {len(prompts)} prompts, results saved to {original_output_file_name}"
    )

def generate_response(
    condition: AttentionVariation,
    prefix: str,
    parallel_substrings: typing.List[str],
    suffix: str,
    model,
    tokenizer,
    max_new_tokens: int,
    parallel_ordering: typing.Union[None, typing.List[int]],
    metadata: typing.Union[None, dict] = None,
)->order_independent_llm.OrderIndependentResult:

    if condition in [
        AttentionVariation.normal,
        AttentionVariation.normal_padded,
        AttentionVariation.normal_reversed,
        AttentionVariation.normal_permuted,
    ]:
        is_order_independent = False
    else:
        is_order_independent = True
    if condition in [
        AttentionVariation.normal_reversed,
        AttentionVariation.only_parallel_attention_reversed,
        AttentionVariation.only_parallel_position_reversed,
    ] :
        substrings = parallel_substrings[::-1]
    elif condition in [AttentionVariation.normal_permuted]:
        substrings = list(np.asarray(parallel_substrings)[parallel_ordering])
    else:
        substrings = parallel_substrings

    if condition in [
        AttentionVariation.normal_padded,
        AttentionVariation.order_independent_padded,
    ]:
        pad_attention = True
    else:
        pad_attention = False

    if condition in [
        AttentionVariation.only_parallel_attention,
        AttentionVariation.only_parallel_attention_reversed,
    ]:
        edit_position = False
        edit_attention = True
    elif condition in [
        AttentionVariation.only_parallel_position,
        AttentionVariation.only_parallel_position_reversed,
    ]:
        edit_position = True
        edit_attention = False
    else:
        edit_position = True
        edit_attention = True


    return order_independent_llm.order_independent_query(
            prefix=prefix,
            parallel_substrings=substrings,
            suffix=suffix,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            #torch_device=torch_device,
            is_order_independent=is_order_independent,
            edit_position= edit_position,
            edit_attention = edit_attention,
            pad_attention=pad_attention,
            metadata=metadata,
        )
if __name__ == "__main__":
    main()
