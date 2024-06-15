import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerFast

from src.data_utils.append_template import append_template_to_text
from src.utils.config import (SENTIMENT2OPINION, LOSS_IGNORE_INDEX, SENTIMENT2LABEL, SEPARATOR_SENTENCES,
                              TEXT_LABEL_SEPARATOR, SENTIMENT_ELEMENT_PARTS,
                              NULL_ASPECT_TERM, NULL_ASPECT_TERM_CONVERTED, CATEGORY_TO_NATURAL_PHRASE_MAPPING,
                              MULTI_TASK_TASKS)
from src.utils.tasks import Task


@dataclass
class Label:
    """Label dataclass."""
    label: str = ""
    start_offset: int = 0
    end_offset: int = 0

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return self.label == other.label


def _convert_category(category: str) -> str:
    """
    Convert category in format 'ENTITY#ATTRIBUTE' to '<entity> <attribute>'.

    :param category: category in format 'ENTITY#ATTRIBUTE'
    :return: category in format '<entity> <attribute>'
    """
    if category not in CATEGORY_TO_NATURAL_PHRASE_MAPPING:
        logging.error("category %s not in CATEGORY_TO_NATURAL_PHRASE_MAPPING", category)
        exit(1)
    return CATEGORY_TO_NATURAL_PHRASE_MAPPING[category]


class Seq2SeqDataset(Dataset):
    """Dataset for ABSA and Seq2Seq models."""

    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizerFast,
            max_seq_len_text: int,
            max_seq_len_label: int,
            task: Task,
            few_shot: int = 0,
    ) -> None:
        """
        Initialize dataset for Seq2Seq ABSA with given arguments.

        :param data_path: path to the data file
        :param tokenizer:  tokenizer
        :param max_seq_len_text: maximum length of the text sequence
        :param max_seq_len_label: maximum length of the label sequence
        :param task: task
        :param few_shot: number of samples to use for few shot learning, 0 means all samples
        """

        self._data_path = data_path
        self._max_seq_len_text = max_seq_len_text
        self._max_seq_len_label = max_seq_len_label
        self._tokenizer = tokenizer
        self._few_shot = few_shot
        self._task = task

        self._encoded_inputs = []
        self._encoded_labels = []
        self._labels = []
        self._tasks = []

        self._load_data()

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        :return: length of the dataset
        """
        return len(self._encoded_inputs)

    def __getitem__(self, index: int) -> dict:
        """
        Return the dictionary for item at the given index.
        Dictionary contains the following keys:
        - input_text_ids: token ids of the text sequence
        - input_attention_mask: attention mask of the text sequence
        - labels_ids: token ids of the label sequence
        - labels_attention_mask: attention mask of the label sequence
        - labels: label text
        - task: task

        :param index: index of the item
        :return: dictionary containing input text ids, input attention mask, label ids, label attention mask and label
        for item at the given index
        """
        source_ids = self._encoded_inputs[index].input_ids.squeeze()
        target_ids = self._encoded_labels[index].input_ids.squeeze()

        source_mask = self._encoded_inputs[index].attention_mask.squeeze()
        target_mask = self._encoded_labels[index].attention_mask.squeeze()

        return {
            "input_text_ids": source_ids,
            "input_attention_mask": source_mask,
            "labels_ids": target_ids,
            "labels_attention_mask": target_mask,
            "labels": self._labels[index],
            "task": self._tasks[index].value,
        }

    def _load_data(self) -> None:
        """
        Load data from the dataset file. Convert text and label into token IDs and attention masks.

        :return: None
        """
        tree = ET.parse(self._data_path)

        root = tree.getroot()

        sentence_elements = root.iter("sentence")

        for sentence_element in sentence_elements:
            text_element = sentence_element.find("text")
            if text_element is None:
                continue
            text = text_element.text
            if not text:
                continue
            labels = {}
            opinions = sentence_element.find("Opinions")
            if opinions is None or opinions.find("Opinion") is None or len(list(opinions.iter("Opinion"))) == 0:
                continue

            for opinion in opinions.iter("Opinion"):
                aspect_term = opinion.attrib.get("target", None)
                sentiment = opinion.attrib.get("polarity", None)
                category = opinion.attrib.get("category", None)
                start_offset = opinion.attrib.get("from", 0)
                end_offset = opinion.attrib.get("to", 0)
                if aspect_term is None or sentiment is None or category is None:
                    continue
                if sentiment not in SENTIMENT2LABEL:
                    continue

                if aspect_term == NULL_ASPECT_TERM:
                    aspect_term = NULL_ASPECT_TERM_CONVERTED

                if self._task != Task.MULTI_TASK:
                    converted_label = self._build_label(
                        aspect_term=aspect_term,
                        sentiment=sentiment,
                        category=category,
                        task=self._task,
                    )
                    if self._task not in labels:
                        labels[self._task] = []
                    label = Label(label=converted_label, start_offset=int(start_offset), end_offset=int(end_offset))
                    labels[self._task].append(label)
                else:
                    for task in MULTI_TASK_TASKS:
                        converted_label = self._build_label(
                            aspect_term=aspect_term,
                            sentiment=sentiment,
                            category=category,
                            task=task,
                        )
                        if task not in labels:
                            labels[task] = []
                        label = Label(label=converted_label, start_offset=int(start_offset), end_offset=int(end_offset))
                        labels[task].append(label)

            if not labels:
                continue

            if self._task != Task.MULTI_TASK:
                labels_for_task = set(labels.get(self._task, []))
                sorted_labels = [label.label for label in sorted(labels_for_task, key=sort_label_key)]
                labels_joined = f" {SEPARATOR_SENTENCES} ".join(sorted_labels)
                end_loop = self._add_sample(
                    text=text, label=labels_joined, task=self._task
                )

                if end_loop:
                    break
            else:
                for task in MULTI_TASK_TASKS:
                    labels_for_task = set(labels.get(task, []))
                    sorted_labels = [label.label for label in sorted(labels_for_task, key=sort_label_key)]
                    labels_joined = f" {SEPARATOR_SENTENCES} ".join(sorted_labels)
                    end_loop = self._add_sample(
                        text=text, label=labels_joined, task=task
                    )

                    if end_loop:
                        break

        # Print example of first sentence as tokens and converted back to text
        first_padding_idx = self._encoded_inputs[0].attention_mask.argmin()
        if first_padding_idx == 0:
            first_padding_idx = self._encoded_inputs[0].attention_mask.shape[1]
        first_example_input_ids = self._encoded_inputs[0].input_ids[0]
        logging.info("Example of first sentence token ids: %s", str(first_example_input_ids[:first_padding_idx]))
        logging.info(
            "Example of first sentence: %s",
            str(self._tokenizer.decode(first_example_input_ids[:first_padding_idx]))
        )

        first_padding_idx = self._encoded_labels[0].attention_mask.argmin()
        if first_padding_idx == 0:
            first_padding_idx = self._encoded_labels[0].attention_mask.shape[1]
        logging.info(
            "Example of first label token ids: %s", str(self._encoded_labels[0].input_ids[0][:first_padding_idx])
        )
        logging.info("Example of first label: %s", str(self._labels[0]))
        logging.info("Number of samples: %d", len(self._encoded_inputs))

    def _add_sample(self, text: str, label: str, task: Task) -> bool:
        """
        Append template for given task to text.
        Encode input text add it to the list.
        Append label to the list of labels.
        Encode label and append it to the list of encoded labels.
        Ensure that the label is not computed on pad tokens.
        Append task to the list of tasks.

        :param text: text to encode
        :param label: label
        :param task: task
        :return: bool value, True if few shot limit was reached, False otherwise
        """
        if task == Task.TASD:
            template = f"{TEXT_LABEL_SEPARATOR} {SENTIMENT_ELEMENT_PARTS['AT']} {SENTIMENT_ELEMENT_PARTS['AC']} {SENTIMENT_ELEMENT_PARTS['SP']}"
        elif task == Task.ACSA:
            template = f"{TEXT_LABEL_SEPARATOR} {SENTIMENT_ELEMENT_PARTS['AC']} {SENTIMENT_ELEMENT_PARTS['SP']}"
        elif task == Task.ACD:
            template = f"{TEXT_LABEL_SEPARATOR} {SENTIMENT_ELEMENT_PARTS['AC']}"
        elif task == Task.E2E:
            template = f"{TEXT_LABEL_SEPARATOR} {SENTIMENT_ELEMENT_PARTS['AT']} {SENTIMENT_ELEMENT_PARTS['SP']}"
        elif task == Task.ACTE:
            template = f"{TEXT_LABEL_SEPARATOR} {SENTIMENT_ELEMENT_PARTS['AT']} {SENTIMENT_ELEMENT_PARTS['AC']}"
        elif task == Task.ATE:
            template = f"{TEXT_LABEL_SEPARATOR} {SENTIMENT_ELEMENT_PARTS['AT']}"
        else:
            raise ValueError(f"Task {task} not supported for Seq2Seq models.")
        # text = f"{text} {TEXT_LABEL_SEPARATOR} {template}"
        encoded_input = self._tokenize_text(text, self._max_seq_len_text, template=template)
        self._encoded_inputs.append(encoded_input)

        self._labels.append(label)

        encoded_label = self._tokenize_text(label, self._max_seq_len_label)

        # Ignore computing loss on pad tokens
        encoded_label.input_ids[encoded_label.input_ids == self._tokenizer.pad_token_id] = LOSS_IGNORE_INDEX
        self._encoded_labels.append(encoded_label)

        self._tasks.append(task)

        if 0 < self._few_shot <= len(self._encoded_inputs):
            return True
        return False

    def _tokenize_text(self, text: str, max_length: int, template: str | None = None) -> BatchEncoding:
        """
        Tokenize text and return token ids and attention mask. Add template if provided.

        :param text: text to tokenize
        :param max_length: maximum length of the tokenized text
        :param template: template to add to the text, if None, no template is added
        :return: dictionary containing token IDs and attention mask
        """
        encoded_inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
        )
        if template is not None:
            encoded_template = self._tokenizer(
                template,
                return_tensors="pt",
                add_special_tokens=False,
                return_attention_mask=True,
            )
            encoded_inputs = append_template_to_text(
                tokenizer=self._tokenizer,
                encoded_input=encoded_inputs,
                encoded_template=encoded_template,
                max_length=max_length,
                last_token_id=self._tokenizer.eos_token_id,
            )
        return encoded_inputs

    def _build_label(
            self,
            aspect_term: str,
            sentiment: str,
            category: str,
            task: Task,
    ) -> str:
        """
        Build a label text for the given aspect term, sentiment and category.

        The label text is built as follows:
        - category is converted into '<entity> <attribute>'
        - final label is built as '[A] <aspect_term> [C] <category> [P] <sentiment polarity>' with
            missing parts based on the task

        :param aspect_term: aspect term
        :param sentiment: sentiment ('positive', 'negative', 'neutral')
        :param category: category in format 'ENTITY#ATTRIBUTE'
        :param task: task
        :return: label text
        """
        category = _convert_category(category)

        sentiment = SENTIMENT2OPINION[sentiment]

        if task == Task.TASD:
            label = f"{SENTIMENT_ELEMENT_PARTS['AT']} {aspect_term} {SENTIMENT_ELEMENT_PARTS['AC']} {category} {SENTIMENT_ELEMENT_PARTS['SP']} {sentiment}"
        elif task == Task.ACSA:
            label = f"{SENTIMENT_ELEMENT_PARTS['AC']} {category} {SENTIMENT_ELEMENT_PARTS['SP']} {sentiment}"
        elif task == Task.ACD:
            label = f"{SENTIMENT_ELEMENT_PARTS['AC']} {category}"
        elif task == Task.E2E:
            label = f"{SENTIMENT_ELEMENT_PARTS['AT']} {aspect_term} {SENTIMENT_ELEMENT_PARTS['SP']} {sentiment}"
        elif task == Task.ACTE:
            label = f"{SENTIMENT_ELEMENT_PARTS['AT']} {aspect_term} {SENTIMENT_ELEMENT_PARTS['AC']} {category}"
        elif task == Task.ATE:
            label = f"{SENTIMENT_ELEMENT_PARTS['AT']} {aspect_term}"
        else:
            raise ValueError(f"Task {task} not supported for Seq2Seq models.")

        return label


def sort_label_key(label: Label) -> tuple[int, int, int]:
    """
    Key for sorting labels.

    :param label: label
    :return: key for sorting
    """
    if label.start_offset == 0 and label.end_offset == 0:
        return 1, 0, 0
    return 0, label.start_offset, label.end_offset
