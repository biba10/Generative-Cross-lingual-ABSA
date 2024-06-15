import logging
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data_utils.data_utils import get_encoded_prompt
from src.llm_prompting.templates.prompt_templates import get_instruction
from src.utils.config import SENTIMENT2LABEL, NULL_ASPECT_TERM, CATEGORY_TO_NATURAL_PHRASE_MAPPING
from src.utils.tasks import Task


def create_prompt(instruction: str, input: str, response: str) -> str:
    """
    Create a prompt with instruction, input and response.

    :param instruction: instruction
    :param input: input
    :param response: response
    :return: prompt
    """
    return f"""### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
"""


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


class LLMDataset(Dataset):
    """Dataset for prompting with LLMs like Orca or LlaMa."""

    def __init__(
            self,
            data_path: str,
            task: Task,
            tokenizer: PreTrainedTokenizerBase,
            language: str,
            few_shot_prompt: bool = False,
            max_data: int = 0,
            instruction_tuning: bool = False,
            testing: bool = False,
    ):
        if instruction_tuning and few_shot_prompt:
            raise ValueError("Few-shot prompt not supported for instruction tuning.")

        self._data_path = data_path
        self._task = task
        self._tokenizer = tokenizer
        self._max_data = max_data
        self._instruction = get_instruction(task=task, few_shot=few_shot_prompt, language=language)
        self._instruction_tuning = instruction_tuning
        self._testing = testing

        self._labels = []
        self._texts = []
        self._encoded_inputs = []

        self._load_data()

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        :return: length of the dataset
        """
        return len(self._texts)

    def __getitem__(self, index: int) -> dict:
        if self._instruction_tuning and not self._testing:
            return {
                "input_ids": self._encoded_inputs[index]["input_ids"].squeeze(),
                "attention_mask": self._encoded_inputs[index]["attention_mask"].squeeze(),
            }

        return {
            "input_ids": self._encoded_inputs[index]["input_ids"].squeeze(),
            "attention_mask": self._encoded_inputs[index]["attention_mask"].squeeze(),
            "text": self._texts[index],
            "labels": self._labels[index],
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

            labels = []
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

                label = self._build_label(
                    aspect_term=aspect_term,
                    sentiment=sentiment,
                    category=category,
                )
                labels.append(label)

            self._labels.append(labels)
            self._texts.append(text)

            encoded_input = get_encoded_prompt(
                labels=labels,
                text=text,
                tokenizer=self._tokenizer,
                instruction=self._instruction,
                instruction_tuning=self._instruction_tuning,
                testing=self._testing,
            )

            self._encoded_inputs.append(encoded_input)

            if 0 < self._max_data <= len(self._texts):
                break

        logging.info("Example of first label: %s", str(self._labels[0]))
        logging.info("Example of first text: %s", str(self._texts[0]))
        logging.info(
            "Example of first prompt: %s", str(self._tokenizer.decode(self._encoded_inputs[0]["input_ids"][0]))
        )
        logging.info("Number of samples: %d", len(self._texts))

    def _build_label(
            self,
            aspect_term: str,
            sentiment: str,
            category: str,
    ) -> str:
        if self._task == Task.TASD:
            label = (aspect_term, category, sentiment)
        elif self._task == Task.ACSA:
            label = (category, sentiment)
        elif self._task == Task.ACD:
            label = category
        elif self._task == Task.ACTE:
            label = (aspect_term, category)
        elif self._task == Task.ATE:
            label = aspect_term
        elif self._task == Task.E2E:
            label = (aspect_term, sentiment)
        else:
            raise ValueError(f"Task {self._task} not supported for Seq2Seq models.")

        return label
