import logging

import pytorch_lightning as pl
import torch
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast,
                          PreTrainedTokenizerBase, AutoModelForCausalLM, BitsAndBytesConfig)

from src.model.model_generative import ABSAModelGenerative


def load_model_and_tokenizer_llm(
        model_path: str,
        load_in_8bits: bool,
        token: str,
        use_cpu: bool = False,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load model and tokenizer from path. Load model in 8-bit mode.

    :param model_path: path to pre-trained model or shortcut name
    :param load_in_8bits: if True, model is loaded in 8-bit mode
    :param token: token
    :param use_cpu: if True, CPU is used
    :return: model and tokenizer
    """
    bnb_config = None
    if not use_cpu:
        if load_in_8bits:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,  # load model in 8-bit precision
                low_cpu_mem_usage=True,
            )
            logging.info("Loading model in 8-bit mode")
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # load model in 4-bit precision
                bnb_4bit_quant_type="nf4",  # pre-trained model should be quantized in 4-bit NF format
                bnb_4bit_use_double_quant=True,  # Using double quantization as mentioned in QLoRA paper
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logging.info("Loading model in 4-bit mode")
    else:
        logging.info("Using CPU")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        token=token,
        low_cpu_mem_usage=True if not use_cpu else False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

    return model, tokenizer


def load_model_and_tokenizer(
        model_path: str,
        model_max_length: int,
        max_seq_length_label: int,
        optimizer: str,
        learning_rate: float,
        beam_size: int,
        constrained_decoding: bool,
) -> tuple[pl.LightningModule, PreTrainedTokenizerFast]:
    """
    Load model and tokenizer from path. Add special tokens to tokenizer.

    :param model_path: path to pre-trained model or shortcut name
    :param model_max_length: maximal length of the sequence
    :param max_seq_length_label: maximal length of the label
    :param optimizer: optimizer
    :param learning_rate: learning rate
    :param beam_size: beam size
    :param constrained_decoding: if True, constrained decoding is used
    :return: model and tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model_max_length, use_fast=True)

    model = _load_generative_model(
        model_path=model_path,
        tokenizer=tokenizer,
        optimizer=optimizer,
        learning_rate=learning_rate,
        beam_size=beam_size,
        constrained_decoding=constrained_decoding,
        max_seq_length=max_seq_length_label,
    )
    return model, tokenizer


def _load_generative_model(
        model_path: str,
        tokenizer: PreTrainedTokenizerFast,
        optimizer: str,
        learning_rate: float,
        beam_size: int,
        constrained_decoding: bool,
        max_seq_length: int,
) -> pl.LightningModule:
    """
    Load generative model from path. Add special tokens to tokenizer.

    :param model_path: path to pre-trained model or shortcut name
    :param tokenizer: pre-trained tokenizer
    :param optimizer: optimizer
    :param learning_rate: learning rate
    :param beam_size: beam size
    :param constrained_decoding: if True, constrained decoding is used
    :param max_seq_length: maximum sequence length for generation
    :return: model
    """
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    absa_model = ABSAModelGenerative(
        learning_rate=learning_rate,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        beam_size=beam_size,
        constrained_decoding=constrained_decoding,
        max_length=max_seq_length,
    )

    return absa_model
