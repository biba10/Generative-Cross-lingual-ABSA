import logging
import os

from src.data_utils.data_loader import build_llm_api_prompt_data_loader
from src.llm_prompting.llm_api_classifier import ChatGPTChatClassifier
from src.utils.args_utils import init_args_prompting
from src.utils.config import RESULTS_DIR
from src.utils.logger_utils import generate_file_name, init_logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # For FastTokenizers


def main():
    init_logging()
    args = init_args_prompting()

    result_file = generate_file_name(vars(args))
    result_file = result_file + ".results"
    result_file = os.path.join(RESULTS_DIR, result_file)

    logging.info(f"Prediction file:{result_file}")

    logging.info("Creating Data loader")
    data_loader = build_llm_api_prompt_data_loader(
        language=args.language,
        batch_size=args.batch_size,
        task=args.task,
        max_test_data=args.max_test_data,
    )
    logging.info("Data loader created")

    classifier = ChatGPTChatClassifier(
        args=args,
        data_loader=data_loader,
        prediction_file=result_file,
    )

    classifier.perform_evaluation()
    logging.info("This is the end...")


if __name__ == '__main__':
    main()
