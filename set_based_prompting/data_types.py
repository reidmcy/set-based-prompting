import dataclasses
import json
import re
import typing
import pandas
import os.path

import torch
import transformers.generation.utils

# note that this is a simple regex, nested splits are not supported, and are considered undefined behavior
split_regex = re.compile(
    pattern=r".*<\|start_2d\|>.+<\|split_2d\|>.+<\|end_2d\|>.*", flags=re.DOTALL
)

inner_mid_outer_split = re.compile(
    pattern=r"(.*)<\|start_2d\|>(.+)<\|end_2d\|>(.*)", flags=re.DOTALL
)


@dataclasses.dataclass
class SplitPrompt:
    text: str
    metadata: typing.Union[typing.Dict[str, typing.Any], None]

    def gen_split_text(self) -> typing.Tuple[str, typing.List[str], str]:
        split_possible = split_regex.match(self.text)
        if split_possible is None:
            raise ValueError(
                f"Prompt:'''\n{self.text}\n''' does not contain the necessary split delimiters, it must be of the form '<|start_2d|>...<|split_2d|>...<|end_2d|>'"
            )

        sections = inner_mid_outer_split.match(self.text)
        assert (
            sections is not None
        ), f"Prompt {self.text} is doing something weird (nesting?), it should have matched the regex"

        prefix = sections.group(1)
        parallel_substrings = sections.group(2).split("<|split_2d|>")
        suffix = sections.group(3)

        return prefix, parallel_substrings, suffix

    @classmethod
    def from_split_text(
        cls, prefix: str, parallel_substrings: typing.List[str], suffix: str
    ) -> "SplitPrompt":
        return cls(
            text=prefix
            + "<|start_2d|>"
            + "<|split_2d|>".join(parallel_substrings)
            + "<|end_2d|>"
            + suffix,
            metadata=None,
        )
    @classmethod
    def from_json_file(cls, json_file: str) -> typing.List["SplitPrompt"]:
        with open(json_file, "rt") as fin:
            prompts = json.load(fin)
        ret_prompts: list["SplitPrompt"] = []
        for i, prompt in enumerate(prompts):
            if "prompt_metadata" not in prompt:
                prompt["prompt_metadata"] = None
            if len(prompt.keys()) > 2:
                raise ValueError(
                    f"Prompt number {i+1} '''\n{json.dumps(prompt)}\n''' has too many keys, only prompt and prompt_metadata are allowed"
                )
            ret_prompts.append(
                cls(text=prompt["prompt"], metadata=prompt["prompt_metadata"])
            )
        return ret_prompts


def filter_prompts(
    completes_file: str,
    prompts: typing.List[SplitPrompt],
) -> typing.List[SplitPrompt]:
    try:
        with open(completes_file, "rt") as f:
            complete_promps = [json.loads(l)["prompt"] for l in f]
    except FileNotFoundError:
        return [p for p in prompts]
    except json.JSONDecodeError:
        os.remove(completes_file)
        return [p for p in prompts]
    return [p for p in prompts if p.text not in complete_promps]


@dataclasses.dataclass
class TokenProbs:
    token_ids: typing.List[int]
    token_decoded: typing.List[str]
    token_probs: typing.List[float]
    token_probs_dict: typing.List[typing.Dict[str, float]]
    top_k: int

    def to_json_dict(self) -> dict:
        return {
            "token_ids": self.token_ids,
            "token_decoded": self.token_decoded,
            "token_probs": self.token_probs,
            "token_probs_dict": self.token_probs_dict,
            "top_k": self.top_k,
        }


@dataclasses.dataclass
class OrderIndependentResult:
    prompt: SplitPrompt
    model: str
    max_new_tokens: int
    order_independent_output: bool
    pad_attention: bool
    text_output: str
    edit_position: bool
    edit_attention: bool
    raw_output: typing.Union[
        torch.LongTensor, transformers.generation.utils.GenerateOutput
    ]
    input_ids: torch.Tensor
    label_scores: typing.Dict[str,float]
    label_perplexities: typing.Dict[str,float]
    is_correct_answer: typing.Union[bool, None] = None
    correct_answer_prob: typing.Union[float, None] = None
    raw_output_contains_correct_answer_only: typing.Union[bool, None] = None


    def get_per_token_probs(
        self,
        tokenizer: typing.Union[
            transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
        ],
        top_k: int = 100,
    ) -> TokenProbs:
        raw_output = self.raw_output
        assert isinstance(
            raw_output, transformers.utils.ModelOutput
        ), f"Raw output is not a GenerateOutput, it is a {type(self.raw_output)}"
        assert raw_output.scores is not None, "Raw output does not have scores"

        scores = torch.nn.functional.softmax(torch.stack(raw_output.scores), dim=2)
        seq = raw_output.sequences[0, -len(scores) :]
        tokens = seq.cpu().numpy().tolist()
        token_probs = []
        top_k_indices = torch.topk(scores, top_k, dim=2)[1]
        for score, token in zip(scores[:, 0, :], tokens):
            token_probs.append(score[token].item())

        token_probs_dict = []
        for i, top_k_index in enumerate(top_k_indices[:, 0, :]):
            token_probs_dict.append(
                {
                    tokenizer.convert_ids_to_tokens(int(t)): scores[i, 0, t].item()
                    for t in top_k_index.cpu().numpy()
                }
            )
        return TokenProbs(
            token_ids=tokens,
            token_decoded=tokenizer.convert_ids_to_tokens(tokens), # type: ignore
            token_probs=token_probs,
            token_probs_dict=token_probs_dict,
            top_k=top_k,
        )

    def to_json_dict(
        self,
    ) -> typing.Dict[str, typing.Union[str,int, float, None, typing.Dict[str, float]]]:#-> typing.Dict[str, typing.Union[str, int, bool, typing.List[float]]:
        return {
            #"prompt": self.prompt.text,
            # "model": self.model,
            "max_new_tokens": self.max_new_tokens,
            "order_independent_output": self.order_independent_output,
            "pad_attention": self.pad_attention,
            "text_output": self.text_output,
            "is_correct_answer": self.is_correct_answer,
            "correct_answer_prob": self.correct_answer_prob,
            "raw_output_contains_correct_answer_only": self.raw_output_contains_correct_answer_only,
            "edit_position": self.edit_position,
            "edit_attention": self.edit_attention,
            "label_scores": self.label_scores,
            "label_perplexities": self.label_perplexities,
        }


def load_to_dataframe(
    path: str,
    fail_on_empty: bool = True,
    include_probs: bool = True,
) -> pandas.DataFrame:
    lines = []
    with open(path, "rt") as fin:
        for line_raw in fin:
            line_data = json.loads(line_raw)
            for response_type, response_data in line_data["responses"].items():
                line_dict = {
                    "response_type": response_type,
                    "model": line_data["model"],
                }
                line_dict.update(response_data)
                probs = response_data.get("probs", None)
                if probs is not None and include_probs:
                    line_dict["probs"]= probs['token_probs']
                    line_dict["probs_top_k"]= probs['token_probs_dict']

                else:
                    line_dict['probs'] = None

                for key, value in line_data["prompt_metadata"].items():
                    line_dict[f"meta_{key}"] = value
                line_dict["prompt"] = line_data["prompt"]
                line_dict["file_name"] = (
                    os.path.basename(path).replace(".jsonl", "").replace(".json", "")
                )
                lines.append(line_dict)
    if len(lines) <= 0 and fail_on_empty:
        raise ValueError(f"File {path} does not contain any data")
    return pandas.DataFrame(lines)
