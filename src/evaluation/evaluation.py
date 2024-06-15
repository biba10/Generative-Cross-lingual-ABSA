import logging

import numpy as np

from src.evaluation.f1_score_seq2seq import F1ScoreSeq2Seq
from src.evaluation.slots import Slots
from src.evaluation.total_f1_score import F1ScoreTotal
from src.utils.config import (NULL_ASPECT_TERM, SEPARATOR_SENTENCES, SENTIMENT_ELEMENT_PARTS,
                              NULL_ASPECT_TERM_CONVERTED)
from src.utils.tasks import Task


def _parse_sentence(sentence: str) -> tuple[str, str, str]:
    """
    Parse sentence to retrieve sentiment elements.

    :param sentence: sentence to parse
    :return: aspect category, aspect term and sentiment
    """
    for part in SENTIMENT_ELEMENT_PARTS.values():
        if part not in sentence:
            sentence += f"{part} null"
    indexes = [sentence.index(part) for part in SENTIMENT_ELEMENT_PARTS.values()]
    arg_index_list = list(np.argsort(indexes))

    result = []
    for i in range(len(indexes)):
        start = indexes[i] + len(list(SENTIMENT_ELEMENT_PARTS.values())[i])
        sort_index = arg_index_list.index(i)
        if sort_index < len(indexes) - 1:
            next_ = arg_index_list[sort_index + 1]
            re = sentence[start:indexes[next_]]
        else:
            re = sentence[start:]
        result.append(re.strip())

    aspect_term, aspect_category, sentiment = result

    return aspect_term, aspect_category, sentiment


def _retrieve_sentences(sample: str) -> list[str]:
    """
    Retrieve sentences from sample.

    :param sample: data sample
    :return: list of sentences
    """
    sentences = [sent.strip() for sent in sample.split(SEPARATOR_SENTENCES)]
    return sentences


def _parse_sample(
        sample: str,
        acd: set,
        ate: set,
        acte: set,
        tasd: set,
        e2e: set,
        acsa: set,
        task: Task
) -> None:
    """
    Parse sample to retrieve slots.

    :param sample: data sample
    :param acd: acd contains aspect categories
    :param ate: ate contains aspect terms
    :param acte: acte contains aspect categories and aspect terms
    :param tasd: tasd contains aspect categories, aspect terms and sentiment
    :param e2e: e2e contains aspect terms and sentiment
    :param acsa: acsa contains aspect categories and sentiment
    :param task: task for the sample
    :return: None
    """
    if sample.strip():
        sentences = _retrieve_sentences(sample)

        for sentence in sentences:
            try:
                aspect_term, aspect_category, sentiment = _parse_sentence(sentence)

            except Exception as e:
                logging.error("Exception: %s - %s", str(sentence), str(e))
                aspect_category, aspect_term, sentiment = "", "", ""

            if aspect_term == NULL_ASPECT_TERM_CONVERTED:
                aspect_term = NULL_ASPECT_TERM
            if task == Task.ACD:
                acd.add(aspect_category)
            elif task == Task.ATE:
                ate.add(aspect_term)
            elif task == Task.ACTE:
                # Add aspect category aspect_term tuple to ACTE
                acte.add((aspect_term, aspect_category))
            elif task == Task.TASD:
                # Add aspect category, aspect_term, sentiment tuple to TASD
                tasd.add((aspect_term, aspect_category, sentiment))
            elif task == Task.E2E:
                # Add aspect term, sentiment tuple to E2E
                e2e.add((aspect_term, sentiment))
            elif task == Task.ACSA:
                # Add aspect category, sentiment tuple to ACSA
                acsa.add((aspect_category, sentiment))


def retrieve_slots(sample: str, task: Task) -> Slots:
    """
    Retrieve slots from data sample.

    :param sample: data sample
    :param task: task for the sample
    :return: slots from sample
    """
    acd = set()
    ate = set()
    acte = set()
    tasd = set()
    e2e = set()
    acsa = set()

    # Get slots if sample is not empty
    try:
        _parse_sample(sample=sample, acd=acd, ate=ate, acte=acte, tasd=tasd, e2e=e2e, acsa=acsa, task=task)
    except IndexError as e:
        logging.error("ValueError: %s", str(e))
        return Slots(acd=acd, ate=ate, acte=acte, tasd=tasd, e2e=e2e, acsa=acsa)

    slots = Slots(acd=acd, ate=ate, acte=acte, tasd=tasd, e2e=e2e, acsa=acsa)

    return slots


def process_batch_for_evaluation(
        decoded_predictions: list[str],
        labels: list[str],
        acd_f1_score: F1ScoreSeq2Seq,
        ate_f1_score: F1ScoreSeq2Seq,
        acte_f1_score: F1ScoreSeq2Seq,
        tasd_f1_score: F1ScoreSeq2Seq,
        e2e_f1_score: F1ScoreSeq2Seq,
        acsa_f1_score: F1ScoreSeq2Seq,
        total_f1_score: F1ScoreTotal,
        tasks: list[Task],
) -> None:
    """
    Process batch for evaluation. Convert predictions and labels to slots and update metrics.

    :param decoded_predictions: decoded predictions
    :param labels: labels
    :param acd_f1_score: acd f1 score
    :param ate_f1_score: ate f1 score
    :param acte_f1_score: acte f1 score
    :param tasd_f1_score: tasd f1 score
    :param e2e_f1_score: e2e f1 score
    :param acsa_f1_score: acsa f1 score
    :param total_f1_score: total f1 score
    :param tasks: tasks
    :return: None
    """
    for decoded_prediction, label, task in zip(decoded_predictions, labels, tasks):
        logging.info("Example – task %s", str(task))
        logging.info("Decoded prediction: %s", str(decoded_prediction))
        logging.info("Label: %s", str(label))
        slots_predictions = retrieve_slots(decoded_prediction, task=task)
        slots_labels = retrieve_slots(label, task=task)
        acd_f1_score.update(predictions=slots_predictions.acd, labels=slots_labels.acd)
        ate_f1_score.update(predictions=slots_predictions.ate, labels=slots_labels.ate)
        acte_f1_score.update(predictions=slots_predictions.acte, labels=slots_labels.acte)
        tasd_f1_score.update(predictions=slots_predictions.tasd, labels=slots_labels.tasd)
        e2e_f1_score.update(predictions=slots_predictions.e2e, labels=slots_labels.e2e)
        acsa_f1_score.update(predictions=slots_predictions.acsa, labels=slots_labels.acsa)
        total_f1_score.update(
            [acd_f1_score, ate_f1_score, acte_f1_score, tasd_f1_score, e2e_f1_score, acsa_f1_score]
        )
