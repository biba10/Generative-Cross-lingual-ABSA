import logging
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset

from src.utils.config import SENTIMENT2LABEL, NULL_ASPECT_TERM, CATEGORY_TO_NATURAL_PHRASE_MAPPING
from src.utils.tasks import Task


def _build_label(
        aspect_term: str,
        sentiment: str,
        category: str,
        task: Task,
) -> str:
    if task == Task.TASD:
        label = (aspect_term, category, sentiment)
    elif task == Task.ACSA:
        label = (category, sentiment)
    elif task == Task.ACD:
        label = category
    elif task == Task.ACTE:
        label = (aspect_term, category)
    elif task == Task.ATE:
        label = aspect_term
    elif task == Task.E2E:
        label = (aspect_term, sentiment)
    else:
        raise ValueError(f"Task {task} not supported for Seq2Seq models.")

    return label


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


class LLMApiDataset(Dataset):
    """Dataset for prompting with LLMs like ChatGPT or LLama for API (no tokenizer needed)."""

    def __init__(
            self,
            data_path: str,
            task: Task,
            convert_label_to_czech: bool = False,
            max_test_data: int = 0,
    ):
        self._max_test_data = max_test_data
        self._data_path = data_path
        self._convert_label_to_czech = convert_label_to_czech
        self._task = task

        self._labels = []
        self._texts = []

        self._load_data()

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        :return: length of the dataset
        """
        return len(self._texts)

    def __getitem__(self, index: int) -> dict:
        return {
            "text": self._texts[index],
            "labels": self._labels[index]
        }

    def _load_data(self) -> None:
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
            opinions = sentence_element.find("Opinions")
            if opinions is None or opinions.find("Opinion") is None or len(list(opinions.iter("Opinion"))) == 0:
                continue

            labels_annotation = set()
            for i, opinion in enumerate(opinions.iter("Opinion")):
                aspect_term = opinion.attrib.get("target", None)
                sentiment = opinion.attrib.get("polarity", None)
                category = opinion.attrib.get("category", None)
                if aspect_term is None or sentiment is None or category is None:
                    continue
                if sentiment not in SENTIMENT2LABEL:
                    continue

                if aspect_term == NULL_ASPECT_TERM:
                    aspect_term = aspect_term.lower()

                category = _convert_category(category)

                label = _build_label(
                    aspect_term=aspect_term,
                    sentiment=sentiment,
                    category=category,
                    task=self._task,
                )
                labels_annotation.add(label)
            self._labels.append(list(labels_annotation))
            self._texts.append(text)

            if 0 < self._max_test_data <= len(self._texts):
                break

        logging.info("Example of first label: %s", str(self._labels[0]))
        logging.info("Example of first text: %s", str(self._texts[0]))
        logging.info("Number of samples: %d", len(self._texts))


def data_collate_llm_api_dataset(batch: list[dict]):
    """
    Collate function for DataLoader.

    :param batch: batch of data
    :return: batch of data
    """
    texts = []
    labels = []
    for sample in batch:
        texts.append(sample["text"])
        labels.append(sample["labels"])

    return {
        "text": texts,
        "labels": labels
    }
