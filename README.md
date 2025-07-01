# Aspect-Based Sentiment Analysis #
This repository contains the code and data for the Generative Cross-lingual Sentiment Analysis.
## Requirements ##
Python 3.10 is required. The code was tested on Python 3.10.7.
The required Python packages are listed in the file `requirements.txt`. They can be installed using the command `pip install -r requirements.txt`.

## Structure ##
The repository contains the following files and folders:
* `main.py` – Main file of the application.
* `main_llm.py` – Main file of the application for open-source LLM prompting.
* `main_prompting.py` – Main file of the application for prompting with ChatGPT.
* `instruction_tuning.py` – Main file of the application for instruction tuning.
* `requirements.txt` – List of required Python packages.
* `README.md` – This file.
* `data` – Folder containing the data. See the section `Data` for more details.
* `src` – Folder containing the source code.
  * `data_utils` – Folder containing the code for data loading and preprocessing.
  * `evaluation` – Folder containing the code for evaluation.
  * `models` – Folder containing the code for models.
  * `utils` – Folder containing the code for other utilities.
  * `llm_prompting.py` – Code for LLM prompting.
  
## Data ##
There should be a folder `data` in the root of the repository. The folder should contain the following subfolders:
* `cs` – Folder containing the Czech data.
* `en` – Folder containing the English data.
* `es` – Folder containing the Spanish data.
* `fr` – Folder containing the French data.
* `nl` – Folder containing the Dutch data.
* `ru` – Folder containing the Russian data.
* `tr` – Folder containing the Turkish data.

Each of the language folders should contain the following files:
* `train.xlm` – File with training data.
* `dev.xlm` – File with validation data.
* `test.xlm` – File with test data.

## Usage ##
To run the code, use the command `python main.py`. The code can be configured using the following command-line arguments:
* `--model` – Name or path to pre-trained model. The default value is `t5-base`.
* `--batch_size` – Batch size. The default value is `64`.
* `--max_seq_length` – Maximum sequence length. The default value is `256`. Must be at least `32`.
* `--max_seq_length_label` – Maximum sequence length for labels. The default value is `256`. Must be at least `32`.
* `--lr` – Learning rate. The default value is `1e-4`.
* `--epochs` – Number of training epochs. The default value is `10`.
* `--target_language` – Language of the test data (target language). The default value is `en`. Options: `cs`, `en`, `es`, `fr`, `nl`, `ru`, `tr`.
* `--source_language` – Language of the training data (source language). The default value is `en`. Options: `cs`, `en`, `es`, `fr`, `nl`, `ru`, `tr`.
* `--optimizer` – Optimizer. The default value is `AdamW`. Options: `adafactor`, `AdamW`.
* `--mode` – Mode of the training. The default value is `dev`. Options:
  * `dev` – Evaluates model on validation data after each epoch. The validation set is used for selecting the best model (based on `checkpoint_monitor`), which is then evaluated on the test set of the target language.
  * `test` – Does not evaluate the model on the validation set. The model, fine-tuned exactly for the number of epochs, is evaluated on the test set.
* `--checkpoint_monitor` – Metric based on which the best model will be stored according to the performance on validation data in `dev` mode. The default value is `val_loss`. Options: `val_loss`, `acte_f1`, `tasd_f1`, `acsa_f1`, `e2e_f1`.
* `--accumulate_grad_batches` – Accumulates gradient batches. The default value is `1`. It is used when there is insufficient memory for training for the required effective batch size.
* `--beam_size` – Beam size for beam search decoding. The default value is `1`.
* `--task` – ABSA task. Options: `acte`, `acsa`, `acte`, `tasd`.
* `--target_language_few_shot` – Number of examples for training for target language. None means no examples, 0 means all examples.
* `--constrained_decoding` – Use constrained decoding. It has an effect only when used with sequence-to-sequence models.
* `--no_wandb` – Do not use Weights & Biases for logging.
* `--wandb_entity` – Weights & Biases entity.
* `--wandb_project_name` – Weights & Biases project name.
* `--tag` – Tag for the experiment. It is used for logging with Weights & Biases.
### Examples ###
* `python main.py --model google/mt5-large --batch_size 32 --max_seq_length 256 --max_seq_length_label 512 --lr 3e-4 --epochs 20 --target_language es --source_language en --optimizer adafactor --mode dev --checkpoint_monitor tasd_f1 --accumulate_grad_batches 2 --beam_size 1`
* `python main.py --model google/mt5-large --batch_size 32 --max_seq_length 256 --max_seq_length_label 512 --lr 3e-4 --epochs 20 --target_language_few_shot 10 --target_language es --source_language en --optimizer adafactor --mode dev --checkpoint_monitor tasd_f1 --accumulate_grad_batches 2 --beam_size 1`
* `python main.py --model google/mt5-large --batch_size 32 --max_seq_length 256 --max_seq_length_label 512 --lr 3e-4 --epochs 20 --target_language es --source_language es --optimizer adafactor --mode test --checkpoint_monitor acte_f1 --task acte --accumulate_grad_batches 2 --beam_size 1 --constrained_decoding`

### Open-source LLM Prompting ###
The code for open-source LLMs can be configured using the following command-line arguments:
* `language` – Language of the test dataset. The default value is `en`. Options: `cs`, `en`, `es`, `fr`, `nl`, `ru`, `tr`.
* `target_language` – Language of the test dataset. The default value is `None` and is used for monolingual experiments. Options: `cs`, `en`, `es`, `fr`, `nl`, `ru`, `tr`.
* `token` – Token for loading model.
* `load_in_8bits` – Use 8-bit precision.
* `model` – Path to pre-trained model or shortcut name. The default value is `microsoft/Orca-2-13b`.
* `task` – Task. Options: `tasd`, `acte`, `acsa`, `e2e`.
* `max_test_data` – Amount of data that will be used for testing. The default value is `0`, i.e. all.
* `max_train_data` – Amount of data that will be used for training. The default value is `0`, i.e. all.
* `max_dev_data` – Amount of data that will be used for validation. The default value is `0`, i.e. all.
* `epochs` – Number of training epochs. The default value is `10`.
* `batch_size` – Batch size. The default value is `16`.
* `gradient_accumulation_steps` – Gradient accumulation steps. The default value is `1`.
* `no_wandb` – Do not use WandB.
* `tag` – Tag for WandB.
* `use_cpu` – Use CPU even if GPU is available.
* `wandb_entity` – WandB entity name.
* `wandb_project_name` – WandB project name.
* `few_shot_prompt` – Use few shot.
* `instruction_tuning` – Use instruction tuning.

To run the code with zero-shot and few-shot prompting, use the command `python main_llm.py`.
To run the code with instruction tuning, use the command `python instruction_tuning.py`.
#### Examples ####
* `python main_llm.py --language en --model microsoft/Orca-2-13b --task tasd`
* `python main_llm.py --language cs --model microsoft/Orca-2-13b --task acte --few_shot_prompt`
* `python instruction_tuning.py --language en --model microsoft/Orca-2-13b --task tasd`
* `python instruction_tuning.py --language en --target_language cs --model microsoft/Orca-2-13b --task acte`

### ChatGPT Prompting ###
The code for ChatGPT prompting can be configured using the following command-line arguments:
* `--language` – Language of the test dataset. The default value is `en`. Options: `cs`, `en`, `es`, `fr`, `nl`, `ru`, `tr`.
* `--few_shot` – Use few shot.
* `--model_version` – Model version. The default value is `gpt-3.5-turbo`.
* `--credentials_file_path` – Path to the credentials file for ChatGPT. If not used, the default path `./private/credentials_chatgpt.txt` for ChatGPT.
* `--batch_size` – Batch size. The default value is `1`.
* `--max_test_data` – Amount of data that will be used for testing. The default value is `-1`, i.e. all.
* `--temperature` – Temperature of the model. The default value is `0.9`.
* `--top_p` – Top-p parameter of the model. The default value is `0.95`.
* `--max_tokens` – Max tokens parameter of the model. The default value is `1024`.
* `--task` – Task. Options: `tasd`, `acte`, `acsa`, `e2e`.
* `--reeval_file_path` – Path to file with results predictions. If the argument is passed, the predictions are loaded and evaluated again, otherwise the predictions are created.

To run the code with ChatGPT prompting, use the command `python main_prompting.py`.
#### Examples ####
* `python main_prompting.py --language en --model_version gpt-3.5-turbo --task tasd`
* `python main_prompting.py --language cs --model_version gpt-3.5-turbo --task acte --few_shot`

### Restrictions and details ###
* The user is responsible for selecting the correct `--checkpoint_monitor` (e.g. `acte_f1` is only measured with `acte` task).
* The program automatically detects whether it is possible to use GPU for training.
* If no WandB entity or project is provided, the program will not use WandB for logging.
* Using CPU for LLMs can cause a significant slowdown and unexpected behaviour.
* 
## Citation
If you find this repository helpful for your research, please cite our paper as follows:
```
@conference{icaart25,
 author={Jakub {\v{S}}m{\'i}d and Pavel P{\v{r}}ib{\'a}{\v{n}} and Pavel Kr{\'a}l},
 title={Advancing Cross-Lingual Aspect-Based Sentiment Analysis with LLMs and Constrained Decoding for Sequence-to-Sequence Models},
 booktitle={Proceedings of the 17th International Conference on Agents and Artificial Intelligence - Volume 2: ICAART},
 year={2025},
 pages={757-766},
 publisher={SciTePress},
 organization={INSTICC},
 doi={10.5220/0013349400003890},
 isbn={978-989-758-737-5},
 issn={2184-433X},
}
```
