import argparse
import datetime
import logging
import os
from pathlib import Path

import wandb
from pytorch_lightning.loggers import Logger, WandbLogger, CSVLogger


def get_trainer_logger(args: argparse.Namespace) -> Logger:
    """
    Initialize logger for Trainer – WandB if api key is available, CSV otherwise.

    :param args: parsed arguments used for WandB logger config
    :return: logger (WandB or CSV)
    """
    if args.no_wandb:
        logging.info("Using CSV logger")
        return CSVLogger(save_dir=os.getcwd(), name="logs")

    logging.info("Trying to use wandb logger")
    try:
        logger = WandbLogger(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            tags=[args.tag] if args.tag else [],
            settings=wandb.Settings(_except_exit=False),
        )
    except Exception:
        # Log the exception
        logging.error("Wandb exception: ", exc_info=True)
        logging.info("WandB login failed, using CSV logger")
        logger = CSVLogger(save_dir=os.getcwd(), name="logs")

    return logger


def generate_file_name(config, f1=None):
    time = get_actual_time()
    model_name = Path(config['model_name']).name
    model_ver = config['model_version']

    max_test_data = config['max_test_data']
    if max_test_data > 1:
        max_test_data = int(max_test_data)

    dataset_name = config['language']

    name_file = model_name + "_" \
                + model_ver + "_" \
                + dataset_name \
                + "_prompt-" + str(config['prompt_type']) \
                + "_MXT-" + str(max_test_data)

    name_file += "_" + time
    name_file = name_file.replace('.', '-')
    if f1 is not None:
        name_file = name_file + "_F1-%.4f" % f1

    return name_file


def get_actual_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H.%M_%S.%f")


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def init_logging() -> None:
    """Initialize logging."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
