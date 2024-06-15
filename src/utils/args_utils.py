import argparse
import logging

from src.utils.config import (TEST_LANG_OPTIONS, TRAIN_LANG_OPTIONS, MODE_OPTIONS, ADAFACTOR_OPTIMIZER, ADAMW_OPTIMIZER,
                              LANG_ENGLISH, MODE_DEV)
from src.utils.tasks import Task

MIN_SEQ_LENGTH = 32


def init_args_llm() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--language",
        type=str,
        default=LANG_ENGLISH,
        help="Language of test dataset.",
        choices=TEST_LANG_OPTIONS,
    )

    parser.add_argument(
        "--target_language",
        type=str,
        default=None,
        help="Language of test dataset.",
        choices=TEST_LANG_OPTIONS,
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token for loading model.",
    )

    parser.add_argument(
        "--load_in_8bits",
        default=False,
        action="store_true",
        help="Use 8-bit precision.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Orca-2-13b",
        help="Path to pre-trained model or shortcut name.",
    )

    parser.add_argument(
        "--task",
        type=Task,
        choices=list(Task),
        default=Task.TASD,
        help="Task.",
    )

    parser.add_argument(
        "--max_test_data",
        default=0,
        type=int,
        help="Amount of data that will be used for testing",
    )

    parser.add_argument(
        "--max_train_data",
        default=0,
        type=int,
        help="Amount of data that will be used for training",
    )

    parser.add_argument(
        "--max_dev_data",
        default=0,
        type=int,
        help="Amount of data that will be used for validation",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )

    parser.add_argument(
        "--no_wandb",
        default=False,
        action="store_true",
        help="Do not use WandB.",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for WandB.",
    )

    parser.add_argument(
        "--use_cpu",
        default=False,
        action="store_true",
        help="Use CPU even if GPU is available.",
    )

    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="WandB entity name.",
    )

    parser.add_argument(
        "--wandb_project_name",
        default=None,
        help="WandB project name.",
    )

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument(
        "--few_shot_prompt",
        default=False,
        action="store_true",
        help="Use few shot.",
    )

    group.add_argument(
        "--instruction_tuning",
        default=False,
        action="store_true",
        help="Use instruction tuning.",
    )

    args = parser.parse_args()

    if args.wandb_entity is None or args.wandb_project_name is None:
        args.no_wandb = True
        logging.info("WandB is disabled.")

    logging.info("Arguments: %s", args)

    return args


def init_args_prompting() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--language",
        type=str,
        default=LANG_ENGLISH,
        help="Language of test dataset.",
        choices=TEST_LANG_OPTIONS,
    )

    parser.add_argument(
        "--few_shot",
        required=False,
        action='store_true',
        help="If used, few shot prompt is used",
    )

    parser.add_argument(
        "--model_version",
        type=str,
        default="gpt-3.5-turbo",
        help="Only applicable to chatgpt model for now.",
    )

    parser.add_argument(
        "--credentials_file_path",
        default=None,
        type=str,
        help="Path to the credentials file for ChatGPT"
             " if not use default path './private/credentials_chatgpt.txt for ChatGPT",
    )

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size",
    )

    parser.add_argument(
        "--max_test_data",
        default=-1,
        type=float,
        help="Amount of data that will be used for testing, "
             "if (-inf, 0> than it is ignored, "
             "if (0,1> than percentage of the value is used as training data, "
             "if (1, inf) absolute number of training examples is used as training data",
    )

    parser.add_argument(
        "--temperature",
        default=0.9,
        type=float,
        help="Temperature of the model, how much is creative, currently only applicable to ChatGPT",
    )

    parser.add_argument(
        "--top_p",
        default=0.95,
        type=float,
        help="top_p parameter of the model, currently only applicable to ChatGPT",
    )

    parser.add_argument(
        "--max_tokens",
        default=1024,
        type=int,
        help="max_tokens parameter of the model, currently only applicable to ChatGPT",
    )

    parser.add_argument("--task", type=Task, choices=list(Task), default=Task.TASD, help="Task.")

    parser.add_argument(
        "--reeval_file_path",
        type=str,
        default=None,
        help="Path to file with *results_predictions, If argument is passed,"
             " the predictions are loaded and evaluated again, otherwise the predictions are created",
    )

    args = parser.parse_args()

    logging.info("Arguments: %s", args)

    return args


def init_args() -> argparse.Namespace:
    """
    Initialize arguments for the script.

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="t5-base",
        help="Path to pre-trained model or shortcut name.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size."
    )
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximal sequence length.")
    parser.add_argument(
        "--max_seq_length_label",
        type=int,
        default=256,
        help="Maximal sequence length for the label.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--target_language",
        type=str,
        default=LANG_ENGLISH,
        help="Language of test dataset (target language).",
        choices=TEST_LANG_OPTIONS,
    )
    parser.add_argument(
        "--source_language",
        type=str,
        default=LANG_ENGLISH,
        help="Language of training dataset (source language).",
        choices=TRAIN_LANG_OPTIONS,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=[ADAMW_OPTIMIZER, ADAFACTOR_OPTIMIZER],
        default=ADAMW_OPTIMIZER,
        help="Optimizer.",
    )
    parser.add_argument(
        "--mode", type=str, choices=MODE_OPTIONS, default=MODE_DEV,
        help="Mode - 'dev' evaluates model on validation data after each epoch. The validation set is used for "
             "selecting the best model (based on 'checkpoint_monitor'), which is then evaluated on the test set "
             "of the target language."
             "'test' does not evaluate the model on the validation set. The model, fine-tuned exactly for the number "
             "of epochs, is evaluated on the test set. "

    )
    parser.add_argument(
        "--checkpoint_monitor", type=str,
        choices=["val_loss", "acte_f1", "tasd_f1", "e2e_f1", "acsa_f1"],
        default="val_loss",
        help="Metric based on which the best model will be stored according to the performance on validation data in "
             "'dev' mode"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Accumulate gradient batches. It is used when there is insufficient memory for training"
             " for the required effective batch size.",
    )
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for beam search decoding.")
    parser.add_argument("--task", type=Task, choices=list(Task), default=Task.TASD, help="Task.")
    parser.add_argument(
        "--target_language_few_shot",
        type=int,
        default=None,
        help="Number of examples for training for target language. None means no examples, 0 means all examples.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Do not use WandB.")
    parser.add_argument("--wandb_entity", default=None, help="WandB entity name.")
    parser.add_argument("--wandb_project_name", default=None, help="WandB project name.")
    parser.add_argument("--tag", type=str, default=None, help="Tag for WandB.")
    parser.add_argument(
        "--constrained_decoding",
        action="store_true",
        help="Use constrained decoding. It has an effect only when used with sequence-to-sequence models.",
    )
    args = parser.parse_args()

    logging.info("Arguments: %s", args)

    if args.max_seq_length < MIN_SEQ_LENGTH:
        logging.warning("Please use at least %d for max_seq_length.", MIN_SEQ_LENGTH)
        exit(1)

    if args.max_seq_length_label < MIN_SEQ_LENGTH:
        logging.warning("Please use at least %d for max_seq_length_label.", MIN_SEQ_LENGTH)
        exit(1)

    if args.wandb_entity is None or args.wandb_project_name is None:
        args.no_wandb = True
        logging.info("WandB is disabled.")

    logging.info("Train language: %s", args.source_language)
    logging.info("Test language: %s", args.target_language)

    return args
