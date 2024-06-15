from functools import partial

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from transformers import Adafactor, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.evaluation.evaluation import process_batch_for_evaluation
from src.evaluation.f1_score_seq2seq import F1ScoreSeq2Seq
from src.evaluation.total_f1_score import F1ScoreTotal
from src.utils.config import (CATEGORY_TO_NATURAL_PHRASE_MAPPING, SENTIMENT2OPINION,
                              SPECIAL_TOKENS_CONSTRAINED_DECODING,
                              OTHER_TOKENS_CONSTRAINED_DECODING_WITH_PREFIX,
                              OTHER_TOKENS_CONSTRAINED_DECODING_WITHOUT_PREFIX)
from src.utils.tasks import Task, convert_value_to_task


class ABSAModelGenerative(pl.LightningModule):
    """Generative model for Aspect Based Sentiment Analysis."""

    def __init__(
            self,
            learning_rate: float,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            optimizer: str,
            beam_size: int,
            constrained_decoding: bool,
            max_length: int,
    ) -> None:
        """
        Initialize the model.

        :param learning_rate: learning rate
        :param model: pre-trained model (expecting a Seq2Seq model)
        :param tokenizer: pre-trained tokenizer
        :param optimizer: optimizer
        :param beam_size: beam size
        :param constrained_decoding: whether to use constrained decoding
        :param max_length: maximum sequence length for generation
        """
        super().__init__()

        self._learning_rate = learning_rate
        self._model = model
        self._tokenizer = tokenizer
        self._optimizer = optimizer
        self._beam_size = beam_size
        self._max_length = max_length
        self._constrained_decoding = constrained_decoding

        self._acd_f1_score = F1ScoreSeq2Seq()
        self._ate_f1_score = F1ScoreSeq2Seq()
        self._acte_f1_score = F1ScoreSeq2Seq()
        self._tasd_f1_score = F1ScoreSeq2Seq()
        self._e2e_f1_score = F1ScoreSeq2Seq()
        self._acsa_f1_score = F1ScoreSeq2Seq()
        self._total_f1_score = F1ScoreTotal()

        if self._constrained_decoding:
            self._force_tokens = create_force_tokens(self._tokenizer)

        self.save_hyperparameters(ignore=["model"])

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_attention_mask: torch.Tensor,
            labels: torch.Tensor,
    ) -> Seq2SeqLMOutput:
        """
        Perform forward pass through the model.

        :param input_ids: input ids
        :param attention_mask: attention mask
        :param decoder_attention_mask: decoder attention mask
        :param labels: labels
        :return: model output
        """
        output = self._model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return output

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute loss for a batch.

        :param batch: batch
        :return: loss
        """
        out = self(
            input_ids=batch["input_text_ids"],
            attention_mask=batch["input_attention_mask"],
            labels=batch["labels_ids"],
            decoder_attention_mask=batch["labels_attention_mask"],
        )
        loss = out.loss
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Perform training step for a single batch. Compute loss and log it.

        :param batch: batch
        :param batch_idx: batch index
        :return: loss
        """
        loss = self._compute_loss(batch)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _generate_output_and_update_metrics(self, batch: dict) -> torch.Tensor:
        """
        Generate output and predictions, calculate loss, and update metrics for a single batch.

        :param batch: batch
        :return: loss
        """
        loss = self._compute_loss(batch)

        generated_ids = self._model.generate(
            input_ids=batch["input_text_ids"],
            attention_mask=batch["input_attention_mask"],
            max_length=self._max_length,
            num_beams=self._beam_size,
            prefix_allowed_tokens_fn=partial(
                prefix_allowed_tokens_fn,
                self._tokenizer,
                self._force_tokens,
                batch["input_text_ids"],
                batch["task"],
            ) if self._constrained_decoding else None,
        )

        decoded_predictions = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        gold_labels = batch["labels"]

        tasks = [convert_value_to_task(task) for task in batch["task"]]

        process_batch_for_evaluation(
            decoded_predictions=decoded_predictions,
            labels=gold_labels,
            acd_f1_score=self._acd_f1_score,
            ate_f1_score=self._ate_f1_score,
            acte_f1_score=self._acte_f1_score,
            tasd_f1_score=self._tasd_f1_score,
            e2e_f1_score=self._e2e_f1_score,
            acsa_f1_score=self._acsa_f1_score,
            total_f1_score=self._total_f1_score,
            tasks=tasks,
        )

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform validation step for a single batch. Generate output and predictions, calculate loss, and update metrics.
        Log loss and metrics.

        :param batch: batch
        :param batch_idx: batch index
        :return: validation loss
        """
        loss = self._generate_output_and_update_metrics(batch)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("acd_f1", self._acd_f1_score, prog_bar=True, sync_dist=True)
        self.log("ate_f1", self._ate_f1_score, prog_bar=True, sync_dist=True)
        self.log("acte_f1", self._acte_f1_score, prog_bar=True, sync_dist=True)
        self.log("tasd_f1", self._tasd_f1_score, prog_bar=True, sync_dist=True)
        self.log("e2e_f1", self._e2e_f1_score, prog_bar=True, sync_dist=True)
        self.log("acsa_f1", self._acsa_f1_score, prog_bar=True, sync_dist=True)
        self.log("total_f1", self._total_f1_score, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform test step for a single batch. Generate output and predictions, calculate loss, and update metrics.
        Log loss and metrics.

        :param batch: batch
        :param batch_idx: batch index
        :return: test loss
        """
        loss = self._generate_output_and_update_metrics(batch)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acd_f1", self._acd_f1_score, prog_bar=True)
        self.log("test_ate_f1", self._ate_f1_score, prog_bar=True)
        self.log("test_acte_f1", self._acte_f1_score, prog_bar=True)
        self.log("test_tasd_f1", self._tasd_f1_score, prog_bar=True)
        self.log("test_e2e_f1", self._e2e_f1_score, prog_bar=True)
        self.log("test_acsa_f1", self._acsa_f1_score, prog_bar=True)
        self.log("test_total_f1", self._total_f1_score, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimizer.

        :return: optimizer
        """
        model = self._model
        if self._optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=self._learning_rate)
        elif self._optimizer == "adafactor":
            optimizer = Adafactor(
                model.parameters(),
                lr=self._learning_rate,
                scale_parameter=False,
                relative_step=False,
            )
        else:
            raise ValueError(f"Optimizer {self._optimizer} not implemented.")
        return optimizer


def create_force_tokens(tokenizer: PreTrainedTokenizerFast) -> dict[str, list[int] | dict[str, int]]:
    force_tokens = {}

    category_tokens = []
    for category in CATEGORY_TO_NATURAL_PHRASE_MAPPING.values():
        category_tokens.extend(tokenizer.encode(category, add_special_tokens=False))
    force_tokens["category_tokens"] = category_tokens

    sentiment_tokens = []
    for sentiment in SENTIMENT2OPINION.values():
        sentiment_tokens.extend(tokenizer.encode(sentiment, add_special_tokens=False))
    force_tokens["sentiment_tokens"] = sentiment_tokens

    special_tokens = []
    for special_token in SPECIAL_TOKENS_CONSTRAINED_DECODING:
        special_tokens.append(tokenizer.encode(special_token, add_special_tokens=False)[-1])
    # Special tokens are tokens that follow the left bracket, remove the left bracket which is at the beginning
    force_tokens["special_tokens"] = special_tokens

    other_tokens = {}
    for token in OTHER_TOKENS_CONSTRAINED_DECODING_WITH_PREFIX:
        other_tokens[token[-1]] = tokenizer.encode(token, add_special_tokens=False)[-1]

    for token in OTHER_TOKENS_CONSTRAINED_DECODING_WITHOUT_PREFIX:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        other_tokens[token] = encoded[0] if len(encoded) > 0 else None

    force_tokens["tokens_map"] = other_tokens

    return force_tokens


def prefix_allowed_tokens_fn(
        tokenizer: PreTrainedTokenizerFast,
        force_tokens: dict[str, list[int] | dict[str, int]],
        source_ids: torch.Tensor,
        tasks: list[str],
        batch_id: int,
        input_ids: torch.Tensor,
) -> list[int]:
    left_bracket_index = (input_ids == force_tokens["tokens_map"]["["]).nonzero()
    right_bracket_index = (input_ids == force_tokens["tokens_map"]["]"]).nonzero()
    left_brackets_count = len(left_bracket_index)
    right_brackets_count = len(right_bracket_index)
    last_left_bracket_pos = left_bracket_index[-1][0] if left_bracket_index.nelement() > 0 else -1
    task = convert_value_to_task(tasks[batch_id])

    # Get the ID of the last token as one number

    if last_left_bracket_pos == -1:
        # Start of the sequence, return "["
        return [force_tokens["tokens_map"]["["]]

    # There is at least one left bracket
    current_id = input_ids[-1].item()

    # Current term is the first ID after last left bracket
    current_term = None
    # If the last left bracket is also last token, take the token after the previous left bracket if there is any
    if last_left_bracket_pos == len(input_ids) - 1:
        # If it is not the first left bracket, then take the token after the previous left bracket
        if left_brackets_count > 1:
            current_term = input_ids[left_bracket_index[-2][0] + 1]
    else:
        # Current term is after the last left bracket
        current_term = input_ids[last_left_bracket_pos + 1]

    # Current term is known

    # Handle the case when the current ID is left bracket
    if current_id == force_tokens["tokens_map"]["["]:
        # If current term is None or current term is ";", then return "A" or "C", depending on the task
        if current_term is None or current_term == force_tokens["tokens_map"][";"]:
            if task == Task.ACD or task == Task.ACSA:
                return [force_tokens["tokens_map"]["C"]]
            else:
                return [force_tokens["tokens_map"]["A"]]
        # If current term is "A", then return "C", "P" or ";", depending on the task
        if current_term == force_tokens["tokens_map"]["A"]:
            if task == Task.E2E:
                return [force_tokens["tokens_map"]["P"]]
            elif task == Task.ATE:
                return [force_tokens["tokens_map"][";"]]
            else:
                return [force_tokens["tokens_map"]["C"]]
        # If current term is "C", then return "S" or ";", depending on the task
        if current_term == force_tokens["tokens_map"]["C"]:
            if task == Task.ACD or task == Task.ACTE:
                return [force_tokens["tokens_map"][";"]]
            else:
                return [force_tokens["tokens_map"]["P"]]
        # If current term is "P", then return ";"
        if current_term == force_tokens["tokens_map"]["P"]:
            return [force_tokens["tokens_map"][";"]]
        raise ValueError(f"Current term {current_term} not implemented.")

    # Handled the case when the current ID is left bracket

    # Handle the case when current ID is one after the last left bracket, which should mean that there are more left brackets than right brackets
    if right_brackets_count < left_brackets_count:
        if current_id in force_tokens["special_tokens"]:
            # If current term is one of the special tokens, then return "]"
            return [force_tokens["tokens_map"]["]"]]
        raise ValueError(f"Wrong current_id {current_id} after last left bracket.")

    # Brackets are handled, handle the return values for current term
    current_source_ids = source_ids[batch_id].tolist()
    if current_term == force_tokens["tokens_map"]["P"]:  # S
        ret = force_tokens["sentiment_tokens"]
    elif current_term == force_tokens["tokens_map"]["A"]:  # A
        force_list = current_source_ids
        force_list.append(force_tokens["tokens_map"]["it"])  # it
        ret = force_list
    elif current_term == force_tokens["tokens_map"]["C"]:  # C
        ret = force_tokens["category_tokens"]
    elif current_term == force_tokens["tokens_map"][";"]:  # ;
        ret = [force_tokens["tokens_map"]["["]]
    else:
        raise ValueError(f"Wrong current_term {current_term}.")

    ret = set(ret)
    # Same number of left and right brackets, do not allow right bracket
    ret.discard(force_tokens["tokens_map"]["]"])
    ret.discard(force_tokens["tokens_map"]["|"])
    # Also do not allow the special tokens that are inside the brackets if current term is "A"
    if current_term == force_tokens["tokens_map"]["A"]:
        # Find the position of "|" in the current source ids
        pipe_pos = current_source_ids.index(force_tokens["tokens_map"]["|"])
        if pipe_pos != -1:
            current_source_ids_set = set(current_source_ids[:pipe_pos])
            for special_token in force_tokens["special_tokens"]:
                if special_token not in current_source_ids_set:
                    ret.discard(special_token)

    # If the last token is the right bracket and the current term is not ";", do not allow left bracket and eos token
    # It is desired that something is generated after the right bracket
    ret.add(force_tokens["tokens_map"]["["])
    ret.add(tokenizer.eos_token_id)
    if current_id == force_tokens["tokens_map"]["]"]:
        # No EOS token after right bracket ever
        ret.discard(tokenizer.eos_token_id)
        if current_term != force_tokens["tokens_map"][";"]:
            # Left bracket can follow right bracket only if the current term is ";", i.e. new sentence is started
            ret.discard(force_tokens["tokens_map"]["["])
    if (current_id == force_tokens["tokens_map"][" "] and len(input_ids) > 1
            and input_ids[-2] == force_tokens["tokens_map"]["]"]):
        # No EOS token after right bracket ever
        ret.discard(tokenizer.eos_token_id)
    # Do not allow EOS token if last term is not the one corresponding to the task
    if task == Task.ATE:
        if current_term != force_tokens["tokens_map"]["A"]:
            ret.discard(tokenizer.eos_token_id)
    elif task == Task.ACD or task == Task.ACTE:
        if current_term != force_tokens["tokens_map"]["C"]:
            ret.discard(tokenizer.eos_token_id)
    else:
        if current_term != force_tokens["tokens_map"]["P"]:
            ret.discard(tokenizer.eos_token_id)
    ret = list(ret)
    return ret
