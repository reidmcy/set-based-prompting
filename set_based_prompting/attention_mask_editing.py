import torch
import os
import warnings
import logging
import typing
import transformers

def _quiet_tf() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)
    import tensorflow as tf

    tf.get_logger().setLevel("INFO")
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.ERROR)

_quiet_tf()
import tensorflow as tf # noqa: E402

def get_attention_mask_2d_n_options(tokA, tokMCQOptions, tokD, tokAll, device: typing.Optional[torch.device] = None) -> torch.Tensor:
    """
    Outputs a [1,1,s,s] attention mask where s is the total number of tokens in tokA, tokMCQOptions, tokD
        Values are False where tokens are masked, True where attention should be paid
    """
    nTokA = len(tokA["attention_mask"][0])
    nTokD = len(tokD["attention_mask"][0])
    nTokOptions = max(
        [len(tokMCQOption["attention_mask"][0]) for tokMCQOption in tokMCQOptions]
    )
    nTokPerOption = [
        len(tokMCQOption["attention_mask"][0]) for tokMCQOption in tokMCQOptions
    ]
    nTokAll = (
        nTokA
        + sum(
            [len(tokMCQOption["attention_mask"][0]) for tokMCQOption in tokMCQOptions]
        )
        + nTokD
    )
    assert nTokAll == len(tokAll["attention_mask"][0])
    causal_mask = torch.tril(torch.ones((nTokAll, nTokAll), dtype=torch.bool))

    # keep tensorflow from grapping all the GPUs
    with tf.device('cpu'):
        mask = tf.Variable(causal_mask)

        # All tokens in later MCQ sequences should ignore tokens in earlier MCQ sequences
        for i in range(1, len(tokMCQOptions)):
            # mask[nTokA+i*nTokOptions:nTokA+nOptions*nTokOptions, nTokA+(i-1)*nTokOptions : nTokA+i*nTokOptions].assign(tf.zeros([(nOptions-i)*nTokOptions, nTokOptions], tf.bool))
            mask[
                nTokA + sum(nTokPerOption[:i]) : nTokA + sum(nTokPerOption),
                nTokA + sum(nTokPerOption[: i - 1]) : nTokA + sum(nTokPerOption[:i]),
            ].assign(tf.zeros([sum(nTokPerOption[i:]), nTokPerOption[i - 1]], tf.bool))
        # All tokens should ignore the padding tokens, which occur at indices where tokAll['attention_mask'][0]==0
        paddingTokenIndices = torch.where(tokAll["attention_mask"][0] == 0)[0]
        for ptI in paddingTokenIndices.cpu().numpy():
            mask[:, ptI].assign(tf.zeros([nTokAll], tf.bool))

        mask = tf.convert_to_tensor([tf.Variable(mask)])
        mask = torch.tensor(mask.numpy())
        mask = mask.view(1, 1, nTokAll, nTokAll)
        assert mask.shape == (1, 1, nTokAll, nTokAll)
        return mask.to(torch.float32).to(tokA["input_ids"].device)

def get_attention_mask_2d_recursive(tokSequences, tokAll):
    """
    Outputs a [1,1,s,s] attention mask where s is the total number of tokens in tokA, tokMCQOptions, tokD
        Values are False where tokens are masked, True where attention should be paid
    """
    nTokAll = len(tokAll["attention_mask"][0])
    causal_mask = torch.tril(torch.ones((nTokAll, nTokAll), dtype=torch.bool))

    mask = tf.Variable(causal_mask)

    # All tokens in later MCQ sequences should ignore tokens in earlier MCQ sequences
    startIdx=0
    for tokSequence in tokSequences:
        if not isinstance(tokSequence,list):
            startIdx += len(tokSequence["attention_mask"][0])
            continue
        nTokPerOption = [
            len(tokMCQOption["attention_mask"][0]) for tokMCQOption in tokSequence
        ]
        for i in range(1, len(tokSequence)):
            mask[
                startIdx + sum(nTokPerOption[:i]) : startIdx + sum(nTokPerOption),
                startIdx + sum(nTokPerOption[: i - 1]) : startIdx + sum(nTokPerOption[:i]),
            ].assign(tf.zeros([sum(nTokPerOption[i:]), nTokPerOption[i - 1]], tf.bool))
        startIdx += sum(nTokPerOption)

    # All tokens should ignore the padding tokens, which occur at indices where tokAll['attention_mask'][0]==0
    paddingTokenIndices = torch.where(tokAll["attention_mask"][0] == 0)[0]
    for ptI in paddingTokenIndices.cpu().numpy():
        mask[:, ptI].assign(tf.zeros([nTokAll], tf.bool))

    mask = tf.convert_to_tensor([tf.Variable(mask)])
    mask = torch.tensor(mask.numpy())
    mask = mask.view(1, 1, nTokAll, nTokAll)
    assert mask.shape == (1, 1, nTokAll, nTokAll)
    return mask.to(torch.float32).to(tokAll["input_ids"].device)

def get_position_ids_nopad_n_options(
    tokA, tokMCQOptions, tokD, ordering=None, tokenizer=None
):
    nTokA = len(tokA["attention_mask"][0])
    nTokD = len(tokD["attention_mask"][0])
    nTokOptions = max(
        [len(tokMCQOption["attention_mask"][0]) for tokMCQOption in tokMCQOptions]
    )
    nTokAll = (
        nTokA
        + sum(
            [len(tokMCQOption["attention_mask"][0]) for tokMCQOption in tokMCQOptions]
        )
        + nTokD
    )
    if ordering is None:
        ordering = list(range(len(tokMCQOptions)))
    input_ids, attention_mask = tokA["input_ids"][0], tokA["attention_mask"][0]
    for i in ordering:
        input_ids = torch.cat((input_ids, tokMCQOptions[i]["input_ids"][0]))
        attention_mask = torch.cat(
            (attention_mask, tokMCQOptions[i]["attention_mask"][0])
        )
    input_ids = torch.cat((input_ids, tokD["input_ids"][0]))
    attention_mask = torch.cat((attention_mask, tokD["attention_mask"][0]))
    tokAll = {
        "input_ids": input_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
    }
    assert nTokAll == len(tokAll["attention_mask"][0])
    position_ids = torch.arange(0, nTokA)
    for i in range(len(tokMCQOptions)):
        position_ids = torch.cat(
            (
                position_ids,
                torch.arange(nTokA, nTokA + len(tokMCQOptions[i]["attention_mask"][0])),
            )
        )
    position_ids = torch.cat(
        (position_ids, torch.arange(nTokA + nTokOptions, nTokA + nTokOptions + nTokD))
    ).unsqueeze(0)

    return position_ids, tokMCQOptions, tokAll

def get_position_ids_nopad_recursive(
    tokSequences: typing.List[typing.Union[transformers.BatchEncoding, typing.List[transformers.BatchEncoding]]],
):
    '''
    @ tokSequences is a list of [tok | List[tok]]]

    TODO: new get_position_ids_nopad_recursive
    TODO: new get_attention_mask_2d_recursive
    TODO: new get_tokenized_input_prompt
    '''
    position_ids = torch.arange(0,0)
    startTokIndex = 0
    for toks in tokSequences:
        if not isinstance(toks,list):
            position_ids = torch.cat(
                (position_ids, torch.arange(startTokIndex, startTokIndex + len(toks["attention_mask"][0])))
            )
            startTokIndex += len(toks["attention_mask"][0])
        else:
            for tokMCQOption in toks:
                position_ids = torch.cat(
                    (position_ids, torch.arange(startTokIndex, startTokIndex + len(tokMCQOption["attention_mask"][0])))
                )
            startTokIndex += max(
                [len(tokMCQOption["attention_mask"][0]) for tokMCQOption in toks]
            )
    return position_ids.unsqueeze(0)


def get_position_ids_padded_n_options(
    tokA, tokMCQOptions, tokD, ordering=None, tokenizer=None
):
    nTokA = len(tokA["attention_mask"][0])
    nTokD = len(tokD["attention_mask"][0])
    nTokOptions = max(
        [len(tokMCQOption["attention_mask"][0]) for tokMCQOption in tokMCQOptions]
    )
    nTokAll = nTokA + len(tokMCQOptions) * nTokOptions + nTokD
    if ordering is None:
        ordering = list(range(len(tokMCQOptions)))
    for i in range(len(tokMCQOptions)):
        nTokOption = len(tokMCQOptions[i]["attention_mask"][0])
        if nTokOption < nTokOptions:
            # pad tokMCQOption
            torch_device = tokMCQOptions[i]["input_ids"][0].device
            tokMCQOptions[i]["input_ids"] = torch.cat(
                (
                    tokMCQOptions[i]["input_ids"][0],
                    torch.tensor(
                        [tokenizer.pad_token_id] * (nTokOptions - nTokOption)
                    ).to(torch_device),
                )
            ).unsqueeze(0)
            tokMCQOptions[i]["attention_mask"] = torch.cat(
                (
                    tokMCQOptions[i]["attention_mask"][0],
                    torch.zeros(nTokOptions - nTokOption).to(torch_device),
                )
            ).unsqueeze(0)
    input_ids, attention_mask = tokA["input_ids"][0], tokA["attention_mask"][0]
    for i in ordering:
        input_ids = torch.cat((input_ids, tokMCQOptions[i]["input_ids"][0]))
        attention_mask = torch.cat(
            (attention_mask, tokMCQOptions[i]["attention_mask"][0])
        )
    input_ids = torch.cat((input_ids, tokD["input_ids"][0]))
    attention_mask = torch.cat((attention_mask, tokD["attention_mask"][0]))
    tokAll = {
        "input_ids": input_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
    }
    assert nTokAll == len(tokAll["attention_mask"][0])
    position_ids = torch.arange(0, nTokA)
    for i in range(len(tokMCQOptions)):
        position_ids = torch.cat(
            (position_ids, torch.arange(nTokA, nTokA + nTokOptions))
        )
    position_ids = torch.cat(
        (position_ids, torch.arange(nTokA + nTokOptions, nTokA + nTokOptions + nTokD))
    ).unsqueeze(0)

    return position_ids, tokMCQOptions, tokAll
