import argparse
import ast
import csv
import logging
import time
from abc import ABC, abstractmethod

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.llm_evaluation import evaluate_llm
from src.llm_prompting.llm_access import (load_credentials_open_ai, init_open_ai, classify_sentiment_chatgpt)
from src.llm_prompting.templates.prompt_templates import (BASIC_PROMPT_TASD, BASIC_PROMPT_ACTE, BASIC_PROMPT_E2E,
                                                          BASIC_PROMPT_ACSA, FEW_SHOT_PROMPTS)
from src.utils.config import DEFAULT_CREDENTIALS_FILE_PATH_CHATGPT
from src.utils.prompt import print_time_info, get_table_result_string
from src.utils.tasks import Task


def get_predictions_from_file(file_path: str):
    predictions_df = pd.read_csv(file_path, sep='\t', header=0)

    review_texts = []
    predictions_text = []
    gold_labels = []

    for index, row in predictions_df.iterrows():
        text = row['text_review']
        label = row['label']
        pred_text = row['text_predicted']

        review_texts.append(text)
        predictions_text.append(pred_text)
        gold_labels.append(label)

    return review_texts, predictions_text, gold_labels


class LargeLanguageModelClassifier(ABC):
    def __init__(
            self,
            data_loader: DataLoader,
            prediction_file: str,
            language: str,
            args: argparse.Namespace,
            max_retry_attempts: int = 10,
            sleep_time_retry_attempt_s: int = 10,
            no_wandb: bool = False,
            failed_prediction_file: str | None = None,
    ):
        """

        :param data_loader: DataLoader
        :param max_retry_attempts: maximum number of attempts for one example
        :param sleep_time_retry_attempt_s: sleep time after failed request in seconds
        """
        self._args = args
        self._language = language
        self._prediction_file = prediction_file
        self._sleep_time_retry_attempt_s = sleep_time_retry_attempt_s
        self._max_retry_attempts = max_retry_attempts
        self._data_loader = data_loader
        self._no_wandb = no_wandb
        self._failed_prediction_file = failed_prediction_file if failed_prediction_file is not None else prediction_file + "_failed_texts.txt"

        self._prompt = self.build_prompt()
        logging.info("Built prompt: %s", str(self._prompt))

    def _get_predictions(self):
        review_texts = []
        gold_labels = []
        predictions_text = []

        failed_pred_texts = []

        logging.info("Writing results in file: %s", str(self._prediction_file))

        with open(self._prediction_file, "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(["text_review", "label", "text_predicted"])
            f.flush()

            for i, data in enumerate(tqdm(self._data_loader)):
                texts = data["text"]
                labels = data["labels"]
                for text, label in zip(texts, labels):
                    pred_text = self.get_prediction(text)
                    if pred_text is None:
                        logging.info("We were not able to get prediction for text: %s", str(text))
                        failed_pred_texts.append(text)
                        continue
                    else:
                        pass
                    review_texts.append(text)
                    predictions_text.append(pred_text)
                    gold_labels.append(label)

                    writer.writerow([str(text), str(label), str(pred_text)])
                    f.flush()

        logging.info("Number of text that we were not able to classify: %s", str(len(failed_pred_texts)))
        if len(failed_pred_texts) > 0:
            logging.info("Writing failed examples into file: %s", str(self._failed_prediction_file))
            try:
                with open(self._failed_prediction_file, "w", encoding='utf-8', newline='') as f_failed:
                    writer_failed = csv.writer(f_failed, delimiter='\t', lineterminator='\n')
                    writer_failed.writerow(["failed_text"])
                    f_failed.flush()

                    for failed_text in failed_pred_texts:
                        writer.writerow([str(failed_text)])
            except Exception as e:
                logging.error("Error during writing failed examples: %s", str(e))

        return review_texts, predictions_text, gold_labels

    def perform_evaluation(self, reeval_file_path: str | None = None):
        t0 = time.time()
        if reeval_file_path is not None:
            results_file = reeval_file_path + "_reeval.txt"
            review_texts, predictions_text, gold_labels_text = get_predictions_from_file(reeval_file_path)
            gold_labels = convert_str_to_list(gold_labels_text)
            predictions = convert_str_to_list(predictions_text)
        else:
            results_file = self._prediction_file + "_res.txt"
            review_texts, predictions_text, gold_labels = self._get_predictions()
            predictions = convert_str_to_list(predictions_text)

        eval_time = time.time() - t0
        print_time_info(eval_time, len(gold_labels), "Test results", file=None)

        f1, precision, recall = evaluate_llm(predictions, gold_labels)

        result_string, only_results = get_table_result_string(
            f'{self._language}\tTransformer test:{self.get_model_name()} {self._args}',
            f1=f1, prec=precision, rec=recall, train_test_time=eval_time
        )

        result_string = "\n-----------Test Results------------\n\t" + result_string

        logging.info("\n\n\n-----------Save results------------\n %s\n\n\n", str(only_results))

        with open(results_file, "w", encoding='utf-8') as f:
            f.write(only_results + "\n")

        logging.info(result_string)

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_prediction(self, text: str) -> str:
        pass

    @abstractmethod
    def build_basic_prompt(self):
        pass

    @abstractmethod
    def build_few_shot_prompt(self):
        pass

    def build_prompt(self):
        if self._args.few_shot:
            prompt = self.build_few_shot_prompt()
        else:
            prompt = self.build_basic_prompt()
        return prompt


def build_user_prompt_part():
    user_prompt = "The review:\n\n{text}"
    logging.info("User prompt: %s", str(user_prompt))

    return user_prompt


class ChatGPTChatClassifier(LargeLanguageModelClassifier):
    def __init__(
            self, args: argparse.Namespace,
            data_loader: DataLoader,
            prediction_file: str,
    ):
        super().__init__(
            data_loader=data_loader,
            prediction_file=prediction_file,
            language=args.language,
            args=args,
            no_wandb=args.no_wandb,
        )

        self._user_prompt = build_user_prompt_part()
        self._temperature = args.temperature
        self._max_tokens = args.max_tokens
        self._top_p = args.top_p

        logging.info("Loading credentials for ChatGPT")
        credentials_path = DEFAULT_CREDENTIALS_FILE_PATH_CHATGPT
        if self._args.credentials_file_path is not None:
            credentials_path = self._args.credentials_file_path

        api_key = load_credentials_open_ai(credentials_path)
        self._client = init_open_ai(api_key)

    def get_prediction(self, text: str) -> str:
        num_attempts = 0
        while True:
            try:
                num_attempts += 1
                result = classify_sentiment_chatgpt(
                    self._client,
                    self._args.model_version,
                    self._prompt,
                    self._user_prompt,
                    text,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    top_p=self._top_p,
                )
            except Exception as e:
                logging.info("Error: %s", str(e))
                logging.info("Sleep for %s secs...", str(self._sleep_time_retry_attempt_s))
                time.sleep(self._sleep_time_retry_attempt_s)
                result = None

            if result is not None or num_attempts > self._max_retry_attempts:
                break

        return result

    def build_basic_prompt(self):
        if self._args.task == Task.TASD:
            return BASIC_PROMPT_TASD
        elif self._args.task == Task.ACTE:
            return BASIC_PROMPT_ACTE
        elif self._args.task == Task.E2E:
            return BASIC_PROMPT_E2E
        elif self._args.task == Task.ACSA:
            return BASIC_PROMPT_ACSA
        else:
            raise Exception(f"Unknown task: {self._args.task}")

    def build_few_shot_prompt(self):
        if self._args.language not in FEW_SHOT_PROMPTS:
            raise ValueError(f"Language not supported: {self._args.language}")
        if self._args.task not in FEW_SHOT_PROMPTS[self._args.language]:
            raise ValueError(f"Task not supported: {self._args.task} for language: {self._args.language}")
        return FEW_SHOT_PROMPTS[self._args.language][self._args.task]

    def get_model_name(self) -> str:
        return "ChatGPT" + str(self._args.model_version)


def convert_str_to_list(str_labels: list[str]) -> list[list[tuple]]:
    """
    Convert string representation of list to list of tuples.

    :param str_labels: list of strings
    :return: list of list of tuples
    """
    labels = []
    for str_label in str_labels:
        if "Sentiment elements:" in str_label:
            str_label = str_label.replace("Sentiment elements:", "").strip()
        try:
            label = ast.literal_eval(str_label)
            if label is None:
                label = []
            labels.append(label)
        except Exception as e:
            # append empty list
            logging.error("Error during converting string %s to list: %s", str_label, str(e))
            labels.append([])
    return labels
