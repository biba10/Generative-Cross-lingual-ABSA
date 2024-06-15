import functools
import logging
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase

from src.data_utils.data_utils import data_collate_llm_dataset
from src.data_utils.dataset_seq2seq import Seq2SeqDataset
from src.data_utils.llm_api_dataset import data_collate_llm_api_dataset, LLMApiDataset
from src.data_utils.llm_dataset import LLMDataset
from src.utils.config import ABSA_TRAIN, ABSA_TEST, DATA_DIR_PATH, ABSA_DEV
from src.utils.tasks import Task


def build_llm_api_prompt_data_loader(
        language: str,
        batch_size: int,
        task: Task,
        max_test_data: int,
) -> DataLoader:
    data_path_test = os.path.join(DATA_DIR_PATH, language, ABSA_TEST)
    dataset = LLMApiDataset(
        data_path=str(data_path_test),
        task=task,
        max_test_data=max_test_data,
    )

    return DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=data_collate_llm_api_dataset
        )


def build_llm_data_loader(
        language: str,
        task: Task,
        tokenizer: PreTrainedTokenizerBase,
        max_test_data: int,
        few_shot_prompt: bool = False,
) -> DataLoader:
    """Create data loader for LLM.

    :param language: language of the dataset
    :param task: task to solve
    :param tokenizer: tokenizer
    :param max_test_data: max test data
    :param few_shot_prompt: whether to use few-shot prompting

    :return: data loader
    """
    data_path_test = os.path.join(DATA_DIR_PATH, language, ABSA_TEST)
    dataset = LLMDataset(
        data_path=str(data_path_test),
        tokenizer=tokenizer,
        task=task,
        max_data=max_test_data,
        language=language,
        few_shot_prompt=few_shot_prompt,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=functools.partial(data_collate_llm_dataset, tokenizer=tokenizer),
        drop_last=False,
    )


class SADataLoader(pl.LightningDataModule):
    """Data loader for sentiment analysis."""

    def __init__(
            self,
            source_language: str,
            target_language: str,
            batch_size: int,
            tokenizer: PreTrainedTokenizerFast,
            max_seq_len_text: int,
            max_seq_len_label: int,
            mode: str,
            task: Task,
            target_language_few_shot: int | None,
    ) -> None:
        """
        Initialize data loader for SemEval 2016 task 5 dataset with given arguments.

        :param source_language: language of the train dataset
        :param target_language: language of the test dataset
        :param batch_size: train and validation batch size
        :param tokenizer: tokenizer
        :param max_seq_len_text: maximum length of the text sequence
        :param max_seq_len_label: maximum length of the label sequence
        :param mode: mode - 'dev' splits training data to train and dev sets, 'test' uses all training data for training
        :param task: task to solve
        :param target_language_few_shot: number of examples for few-shot learning in target language
        """
        super().__init__()
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_seq_len_text = max_seq_len_text
        self._max_seq_len_label = max_seq_len_label
        self._source_language = source_language
        self._target_language = target_language
        self._mode = mode
        self._task = task
        self._target_language_few_shot = target_language_few_shot

    def setup(self, stage=None) -> None:
        """
        Setup data loader.

        :param stage: stage ('fit' for training or 'test' for testing)
        :return: None
        """
        if stage == "fit" or stage is None:
            # Load train dataset
            data_path_train = os.path.join(DATA_DIR_PATH, self._source_language, ABSA_TRAIN)
            data_path_dev = os.path.join(DATA_DIR_PATH, self._source_language, ABSA_DEV)
            train_dataset = Seq2SeqDataset(
                data_path=data_path_train,
                tokenizer=self._tokenizer,
                max_seq_len_text=self._max_seq_len_text,
                max_seq_len_label=self._max_seq_len_label,
                task=self._task,
            )
            dev_dataset = Seq2SeqDataset(
                data_path=data_path_dev,
                tokenizer=self._tokenizer,
                max_seq_len_text=self._max_seq_len_text,
                max_seq_len_label=self._max_seq_len_label,
                task=self._task,
            )

            self._train_dataset = train_dataset
            self._dev_dataset = dev_dataset
            if self._source_language != self._target_language and self._target_language_few_shot is not None:
                data_path_train = os.path.join(DATA_DIR_PATH, self._target_language, ABSA_TRAIN)
                train_dataset_target = Seq2SeqDataset(
                    data_path=data_path_train,
                    tokenizer=self._tokenizer,
                    max_seq_len_text=self._max_seq_len_text,
                    max_seq_len_label=self._max_seq_len_label,
                    task=self._task,
                    few_shot=self._target_language_few_shot,
                )

                self._train_dataset = torch.utils.data.ConcatDataset(
                    [
                        self._train_dataset,
                        train_dataset_target,
                    ]
                )

                logging.info("Train data length: %d", len(self._train_dataset))
                logging.info("Dev data length: %d", len(self._dev_dataset))

            logging.info("Train data all length: %d", len(self._train_dataset))

        # Load test dataset
        if stage == "test" or stage is None:
            data_path_test = os.path.join(DATA_DIR_PATH, self._target_language, ABSA_TEST)
            self._test_dataset = Seq2SeqDataset(
                data_path=data_path_test,
                tokenizer=self._tokenizer,
                max_seq_len_text=self._max_seq_len_text,
                max_seq_len_label=self._max_seq_len_label,
                task=self._task,
            )
            logging.info("Test data length: %d", len(self._test_dataset))

    def train_dataloader(self) -> DataLoader:
        """
        Get train data loader.

        :return: train data loader
        """
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=0,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Get dev data loader.

        :return: dev data loader
        """
        return DataLoader(
            self._dev_dataset,
            batch_size=self._batch_size,
            num_workers=0,
        )

    def test_dataloader(self):
        """
        Get test data loader.

        :return: test data loader
        """
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            num_workers=0,
        )
