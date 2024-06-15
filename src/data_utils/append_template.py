import torch
from transformers import PreTrainedTokenizerFast, BatchEncoding


def append_template_to_text(
        tokenizer: PreTrainedTokenizerFast,
        encoded_input: BatchEncoding,
        encoded_template: BatchEncoding,
        max_length: int,
        last_token_id: int,
) -> BatchEncoding:
    """
    Append template to encoded_input. Template is appended to the place of eos token, then append rest of input
    (could be padded).

    :param tokenizer: tokenizer
    :param encoded_input: encoded_input
    :param encoded_template: encoded template
    :param max_length: maximum length for the output
    :param last_token_id: id of the last token in the template
    :return: encoded input with template, respecting the maximum length
    """
    last_special_token_id = torch.where(encoded_input.input_ids == last_token_id)[1]
    if len(last_special_token_id) > 1:
        last_special_token_id = last_special_token_id[-1]
    elif len(last_special_token_id) == 0:
        last_special_token_id = encoded_input.input_ids.shape[1]

    # Calculate how many tokens are needed for the template.
    template_tokens = encoded_template.input_ids.shape[1]
    # Calculate available tokens without the template.
    available_tokens = max_length - template_tokens

    if available_tokens > last_special_token_id:
        input_ids = torch.cat(
            (encoded_input.input_ids[:, :last_special_token_id],
             encoded_template.input_ids,
             encoded_input.input_ids[:, last_special_token_id]), dim=1
        )
        attention_mask = torch.cat(
            (encoded_input.attention_mask[:, :last_special_token_id],
             encoded_template.attention_mask,
             encoded_input.attention_mask[:, last_special_token_id]), dim=1
        )
    else:
        new_input_length = max_length - template_tokens - 1
        input_ids = torch.cat(
            (encoded_input.input_ids[:, :new_input_length],
             encoded_template.input_ids,
             encoded_input.input_ids[:, last_special_token_id]), dim=1
        )
        attention_mask = torch.cat(
            (encoded_input.attention_mask[:, :new_input_length],
             encoded_template.attention_mask,
             encoded_input.attention_mask[:, last_special_token_id]), dim=1
        )

    # Pad to max length
    padding_length = max_length - input_ids.shape[1]
    input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
    attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length), value=0)

    return BatchEncoding(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )
