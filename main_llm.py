import logging
import os

import torch
import wandb

from src.data_utils.data_loader import build_llm_data_loader
from src.llm_prompting.llm_classifier import llm_classify
from src.model.model_tokenizer_loader import load_model_and_tokenizer_llm
from src.utils.args_utils import init_args_llm
from src.utils.logger_utils import init_logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # For FastTokenizers


def main():
    init_logging()
    args = init_args_llm()

    if args.token is not None:
        os.environ["HF_TOKEN"] = args.token

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            tags=[args.tag] if args.tag else [],
        )

    logging.info("Loading tokenizer and model...")
    model, tokenizer = load_model_and_tokenizer_llm(
        model_path=args.model,
        load_in_8bits=args.load_in_8bits,
        token=args.token,
        use_cpu=args.use_cpu,
    )
    logging.info("Tokenizer and model loaded")

    tokenizer.padding_side = "left"

    if tokenizer.chat_template is None:
        if "microsoft/Orca" in args.model:
            tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    if "orca" in args.model.lower():
        assistant_text = "<|im_start|> assistant\n"
    elif "llama-3" in args.model.lower():
        assistant_text = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama" in args.model.lower():
        assistant_text = "[/INST]"
    else:
        raise ValueError("Response template not defined for this model.")

    logging.info("Creating Data loader")
    data_loader = build_llm_data_loader(
        language=args.language,
        task=args.task,
        max_test_data=args.max_test_data,
        tokenizer=tokenizer,
        few_shot_prompt=args.few_shot_prompt,
    )
    logging.info("Data loader created")

    device = torch.device("cpu" if args.use_cpu else "cuda" if torch.cuda.is_available() else "cpu")

    llm_classify(
        model=model,
        tokenizer=tokenizer,
        data_loader=data_loader,
        no_wandb=args.no_wandb,
        assistant_text=assistant_text,
        device=device,
    )

    if not args.no_wandb:
        wandb.finish()
    logging.info("This is the end...")


if __name__ == '__main__':
    main()
