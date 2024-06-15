import logging
import os
import re

import pytorch_lightning as pl
import torch

from src.data_utils.data_loader import SADataLoader
from src.model.model_tokenizer_loader import load_model_and_tokenizer
from src.utils.args_utils import init_args
from src.utils.config import MODE_TEST, MODE_DEV, LANG_ENGLISH
from src.utils.logger_utils import get_trainer_logger, init_logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # For FastTokenizers


def main():
    """Main function."""
    init_logging()
    args = init_args()
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logging.info("Using GPU")
    else:
        logging.info("Using CPU")

    logging.info("Loading logger...")
    trainer_logger = get_trainer_logger(args)
    logging.info("Logger loaded")

    logging.info("Loading model and tokenizer...")
    absa_model, tokenizer = load_model_and_tokenizer(
        model_path=args.model,
        model_max_length=max(args.max_seq_length, args.max_seq_length_label),
        max_seq_length_label=args.max_seq_length_label if args.target_language != LANG_ENGLISH else 448,  # one en test example is much longer
        optimizer=args.optimizer,
        learning_rate=args.lr,
        beam_size=args.beam_size,
        constrained_decoding=args.constrained_decoding,
    )
    logging.info("Model and tokenizer loaded")

    logging.info("Creating data loader...")
    data_loader = SADataLoader(
        source_language=args.source_language,
        target_language=args.target_language,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        max_seq_len_text=args.max_seq_length,
        max_seq_len_label=args.max_seq_length_label,
        mode=args.mode,
        task=args.task,
        target_language_few_shot=args.target_language_few_shot,
    )

    data_loader.setup()
    logging.info("Data loader created")

    logging.info("Initializing trainer...")
    # Callback is used only in 'dev' mode
    if args.mode == MODE_DEV:
        if args.checkpoint_monitor == "val_loss":
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filename="best-checkpoint-{epoch}",
                save_top_k=1,
                verbose=True,
                monitor="val_loss",
                mode="min",
            )
        else:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filename="best-checkpoint-{epoch}",
                save_top_k=1,
                verbose=True,
                monitor=args.checkpoint_monitor,
                mode="max",
            )
        callbacks = [checkpoint_callback]
    else:
        checkpoint_callback = None
        callbacks = []

    # If mode is 'test', no validation is performed
    limit_val_batches = 0 if args.mode == MODE_TEST else 1.0
    num_sanity_val_steps = 0 if args.mode == MODE_TEST else 2

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        logger=trainer_logger,
        log_every_n_steps=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="auto",
    )
    logging.info("Trainer initialized")

    logging.info("Training...")
    trainer.fit(absa_model, data_loader)
    logging.info("Training finished")

    logging.info("Testing...")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=trainer_logger,
    )

    if args.mode == MODE_DEV and checkpoint_callback is not None:
        logging.info("Model saved to: %s", str(checkpoint_callback.best_model_path))
        # Find the best epoch from checkpoint - first number after 'epoch=', convert to int
        best_epoch = int(re.search(r"epoch=(\d+)", checkpoint_callback.best_model_path).group(1))
        logging.info("Best epoch: %d", best_epoch)
        trainer_logger.log_metrics({"best_epoch": best_epoch})
        # Take best score from checkpoint as double
        best_score = checkpoint_callback.best_model_score.item()
        logging.info("Best score: %f", best_score)
        trainer_logger.log_metrics({"best_score": best_score})

    if args.mode == MODE_TEST:
        trainer.test(absa_model, data_loader)
    elif args.mode == MODE_DEV:
        # Test on best checkpoint
        trainer.test(absa_model, dataloaders=data_loader, ckpt_path=checkpoint_callback.best_model_path)

    logging.info("Testing finished")


if __name__ == '__main__':
    main()
