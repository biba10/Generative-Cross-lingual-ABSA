import functools
import logging
import os
import shutil

import torch
import wandb
from peft import AutoPeftModelForCausalLM, LoraConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from src.data_utils.data_utils import data_collate_llm_dataset
from src.data_utils.llm_dataset import LLMDataset
from src.llm_prompting.llm_classifier import llm_classify
from src.utils.args_utils import init_args_llm
from src.utils.config import ABSA_DEV, ABSA_TEST, ABSA_TRAIN, DATA_DIR_PATH
from src.utils.logger_utils import init_logging

SUPPORTED_GPUS_TF32 = ["A100", "A6000", "RTX 30", "RTX 40", "A30", "A40"]


def check_ampere_gpu() -> bool:
    """Check if the GPU supports NVIDIA Ampere or later and enable FP32 in PyTorch if it does."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.info("No GPU detected, running on CPU.")
        return False
    # Get current device named
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    for supported_gpu in SUPPORTED_GPUS_TF32:
        if supported_gpu in device_name:
            logging.info("Detected ampere GPU: %s", device_name)
            return True
    logging.info("Detected non-ampere GPU: %s", device_name)
    return False


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_target_modules(model) -> list[str]:
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


def main():
    init_logging()
    args = init_args_llm()

    args.instruction_tuning = True

    if args.token is not None:
        os.environ["HF_TOKEN"] = args.token

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            tags=[args.tag] if args.tag else [],
        )

    use_cpu = True if args.use_cpu else True if not torch.cuda.is_available() else False
    device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # load model in 4-bit precision
        bnb_4bit_quant_type="nf4",  # pre-trained model should be quantized in 4-bit NF format
        bnb_4bit_use_double_quant=True,  # Using double quantization as mentioned in QLoRA paper
        bnb_4bit_compute_dtype=torch.bfloat16,  # During computation, pre-trained model should be loaded in BF16 format
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config if not use_cpu else None,
        device_map="auto" if not use_cpu else "cpu",
        use_cache=False,
        low_cpu_mem_usage=True,
        token=args.token,
    )

    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, token=args.token)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    resized_embeddings = False
    if tokenizer.pad_token is None:
        if args.batch_size > 1:
            logging.info("Resizing embeddings...")
            tokenizer.pad_token = "[PAD]"
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            resized_embeddings = True
        else:
            logging.info("Setting pad token to eos token...")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        logging.info("Pad token already set...")

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        target_modules=find_target_modules(model),
        modules_to_save=None if not resized_embeddings else ["lm_head", "embed_tokens"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if tokenizer.chat_template is None:
        if "microsoft/Orca" in args.model:
            tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    data_path_train = os.path.join(DATA_DIR_PATH, args.language, ABSA_TRAIN)
    train_dataset = LLMDataset(
        data_path=str(data_path_train),
        task=args.task,
        tokenizer=tokenizer,
        language=args.language,
        max_data=args.max_train_data,
        instruction_tuning=True,
    )

    data_path_dev = os.path.join(DATA_DIR_PATH, args.language, ABSA_DEV)
    dev_dataset = LLMDataset(
        data_path=str(data_path_dev),
        task=args.task,
        tokenizer=tokenizer,
        language=args.language,
        max_data=args.max_dev_data,
        instruction_tuning=True,
    )

    test_language = args.target_language if args.target_language is not None else args.language
    data_path_test = os.path.join(DATA_DIR_PATH, test_language, ABSA_TEST)
    test_dataset = LLMDataset(
        data_path=str(data_path_test),
        task=args.task,
        tokenizer=tokenizer,
        language=test_language,
        max_data=args.max_test_data,
        instruction_tuning=True,
        testing=True,
    )

    output_dir = "output"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    ampere_gpu = False
    if not use_cpu:
        ampere_gpu = check_ampere_gpu()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=args.epochs,
        optim="paged_adamw_32bit",
        report_to=["wandb"] if not args.no_wandb else [],
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        bf16=ampere_gpu,
        tf32=ampere_gpu,
        save_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        use_cpu=use_cpu,
        remove_unused_columns=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        disable_tqdm=True,
        group_by_length=True,
        dataloader_drop_last=False,
    )

    if "orca" in args.model.lower():
        response_template = tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)[2:]
        assistant_text = "<|im_start|> assistant\n"
    elif "llama-3" in args.model.lower():
        response_template = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
        assistant_text = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama" in args.model.lower():
        response_template = tokenizer.encode(" [/INST]", add_special_tokens=False)[1:]
        assistant_text = "[/INST]"
    else:
        raise ValueError("Response template not defined for this model.")

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        dataset_text_field="input_ids",
        max_seq_length=1024,
        data_collator=collator,
    )

    best_model_dir = "best_model"
    logging.info("Training...")
    trainer.train()
    trainer.save_model(best_model_dir)
    logging.info("Training finished")

    if args.load_in_8bits:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # load best model
    model = AutoPeftModelForCausalLM.from_pretrained(
        best_model_dir,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    tokenizer.padding_side = "left"
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=functools.partial(data_collate_llm_dataset, tokenizer=tokenizer),
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    llm_classify(
        model=model,
        tokenizer=tokenizer,
        data_loader=test_dataloader,
        no_wandb=args.no_wandb,
        assistant_text=assistant_text,
        device=device,
    )

    if not args.no_wandb:
        wandb.finish()
    logging.info("This is the end...")


if __name__ == '__main__':
    main()
