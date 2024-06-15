from src.llm_prompting.templates.prompt_template_cs_few import (BASIC_PROMPT_TASD_CS_FEW, BASIC_PROMPT_ACSA_CS_FEW,
                                                                BASIC_PROMPT_E2E_CS_FEW, BASIC_PROMPT_ACTE_CS_FEW)
from src.llm_prompting.templates.prompt_template_en_few import (BASIC_PROMPT_TASD_EN_FEW, BASIC_PROMPT_ACSA_EN_FEW,
                                                                BASIC_PROMPT_E2E_EN_FEW, BASIC_PROMPT_ACTE_EN_FEW)
from src.llm_prompting.templates.prompt_template_tr_few import (BASIC_PROMPT_TASD_TR_FEW, BASIC_PROMPT_ACSA_TR_FEW,
                                                                BASIC_PROMPT_E2E_TR_FEW, BASIC_PROMPT_ACTE_TR_FEW)
from src.llm_prompting.templates.prompt_templates_es_few import (BASIC_PROMPT_TASD_ES_FEW, BASIC_PROMPT_ACSA_ES_FEW,
                                                                 BASIC_PROMPT_E2E_ES_FEW, BASIC_PROMPT_ACTE_ES_FEW)
from src.llm_prompting.templates.prompt_templates_fr_few import (BASIC_PROMPT_TASD_FR_FEW, BASIC_PROMPT_ACSA_FR_FEW,
                                                                 BASIC_PROMPT_E2E_FR_FEW, BASIC_PROMPT_ACTE_FR_FEW)
from src.llm_prompting.templates.prompt_templates_nl_few import (BASIC_PROMPT_TASD_NL_FEW, BASIC_PROMPT_ACSA_NL_FEW,
                                                                 BASIC_PROMPT_E2E_NL_FEW, BASIC_PROMPT_ACTE_NL_FEW)
from src.llm_prompting.templates.prompt_templates_ru_few import (BASIC_PROMPT_TASD_RU_FEW, BASIC_PROMPT_ACSA_RU_FEW,
                                                                 BASIC_PROMPT_E2E_RU_FEW, BASIC_PROMPT_ACTE_RU_FEW)
from src.utils.config import (LANG_ENGLISH, LANG_CZECH, LANG_SPANISH, LANG_FRENCH, LANG_DUTCH, LANG_RUSSIAN,
                              LANG_TURKISH)
from src.utils.tasks import Task

# TASD Prompt templates
BASIC_PROMPT_TASD = f"""According to the following sentiment elements definition:

- The "aspect term" refers to a specific feature, attribute, or aspect of a product or service on which a user can express an opinion. Explicit aspect terms appear explicitly as a substring of the given text. The aspect term might be "null" for the implicit aspect.

- The "aspect category" refers to the category that aspect belongs to, and the available categories include: "food general", "food quality", "food style_options", "food prices", "drinks prices", "drinks quality", "drinks style_options", "restaurant general", "restaurant miscellaneous", "restaurant prices", "service general", "ambience general", "location general", "restaurant style_options".

- The "sentiment polarity" refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: "positive", "negative" and "neutral". "neutral" means mildly positive or mildly negative. Triplets with objective sentiment polarity should be ignored.

Please carefully follow the instructions. Ensure that aspect terms are recognized as exact matches in the review or are "null" for implicit aspects. Ensure that aspect categories are from the available categories. Ensure that sentiment polarities are from the available polarities.

Recognize all sentiment elements with their corresponding aspect terms, aspect categories, and sentiment polarity in the given input text (review). Provide your response in the format of a Python list of tuples: 'Sentiment elements: [("aspect term", "aspect category", "sentiment polarity"), ...]'. Note that ", ..." indicates that there might be more tuples in the list if applicable and must not occur in the answer. Ensure there is no additional text in the response.

"""

# ACSA Prompt templates
BASIC_PROMPT_ACSA = f"""According to the following sentiment elements definition:

- The "aspect category" refers to the category that aspect belongs to, and the available categories include: "food general", "food quality", "food style_options", "food prices", "drinks prices", "drinks quality", "drinks style_options", "restaurant general", "restaurant miscellaneous", "restaurant prices", "service general", "ambience general", "location general", "restaurant style_options".

- The "sentiment polarity" refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: "positive", "negative" and "neutral". "neutral" means mildly positive or mildly negative. Tuples with objective sentiment polarity should be ignored.

Please carefully follow the instructions. Ensure that aspect categories are from the available categories. Ensure that sentiment polarities are from the available polarities.

Recognize all sentiment elements with their corresponding aspect categories and sentiment polarity in the given input text (review). Provide your response in the format of a Python list of tuples: 'Sentiment elements: [("aspect category", "sentiment polarity"), ...]'. Note that ", ..." indicates that there might be more tuples in the list if applicable and must not occur in the answer. Ensure there is no additional text in the response.

"""

# E2E Prompt templates
BASIC_PROMPT_E2E = f"""According to the following sentiment elements definition:

- The "aspect term" refers to a specific feature, attribute, or aspect of a product or service on which a user can express an opinion. Explicit aspect terms appear explicitly as a substring of the given text. The aspect term might be "null" for the implicit aspect.

- The "sentiment polarity" refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: "positive", "negative" and "neutral". "neutral" means mildly positive or mildly negative. Tuples with objective sentiment polarity should be ignored.

Please carefully follow the instructions. Ensure that aspect terms are recognized as exact matches in the review or are "null" for implicit aspects. Ensure that sentiment polarities are from the available polarities.

Recognize all sentiment elements with their corresponding aspect terms and sentiment polarity in the given input text (review). Provide your response in the format of a Python list of tuples: 'Sentiment elements: [("aspect term", "sentiment polarity"), ...]'. Note that ", ..." indicates that there might be more tuples in the list if applicable and must not occur in the answer. Ensure there is no additional text in the response.

"""

# ACTE Prompt templates
BASIC_PROMPT_ACTE = f"""According to the following sentiment elements definition:

- The "aspect term" refers to a specific feature, attribute, or aspect of a product or service on which a user can express an opinion. Explicit aspect terms appear explicitly as a substring of the given text. The aspect term might be "null" for the implicit aspect.

- The "aspect category" refers to the category that aspect belongs to, and the available categories include: "food general", "food quality", "food style_options", "food prices", "drinks prices", "drinks quality", "drinks style_options", "restaurant general", "restaurant miscellaneous", "restaurant prices", "service general", "ambience general", "location general", "restaurant style_options".

Please carefully follow the instructions. Ensure that aspect terms are recognized as exact matches in the review or are "null" for implicit aspects.

Recognize all sentiment elements with their corresponding aspect terms and aspect categories in the given input text (review). Provide your response in the format of a Python list of tuples: 'Sentiment elements: [("aspect term", "aspect category"), ...]'. Note that ", ..." indicates that there might be more tuples in the list if applicable and must not occur in the answer. Ensure there is no additional text in the response.

"""

FEW_SHOT_PROMPTS = {
    LANG_CZECH: {
        Task.TASD: BASIC_PROMPT_TASD + BASIC_PROMPT_TASD_CS_FEW,
        Task.ACSA: BASIC_PROMPT_ACSA + BASIC_PROMPT_ACSA_CS_FEW,
        Task.E2E: BASIC_PROMPT_E2E + BASIC_PROMPT_E2E_CS_FEW,
        Task.ACTE: BASIC_PROMPT_ACTE + BASIC_PROMPT_ACTE_CS_FEW,
    },
    LANG_ENGLISH: {
        Task.TASD: BASIC_PROMPT_TASD + BASIC_PROMPT_TASD_EN_FEW,
        Task.ACSA: BASIC_PROMPT_ACSA + BASIC_PROMPT_ACSA_EN_FEW,
        Task.E2E: BASIC_PROMPT_E2E + BASIC_PROMPT_E2E_EN_FEW,
        Task.ACTE: BASIC_PROMPT_ACTE + BASIC_PROMPT_ACTE_EN_FEW,
    },
    LANG_SPANISH: {
        Task.TASD: BASIC_PROMPT_TASD + BASIC_PROMPT_TASD_ES_FEW,
        Task.ACSA: BASIC_PROMPT_ACSA + BASIC_PROMPT_ACSA_ES_FEW,
        Task.E2E: BASIC_PROMPT_E2E + BASIC_PROMPT_E2E_ES_FEW,
        Task.ACTE: BASIC_PROMPT_ACTE + BASIC_PROMPT_ACTE_ES_FEW,
    },
    LANG_FRENCH: {
        Task.TASD: BASIC_PROMPT_TASD + BASIC_PROMPT_TASD_FR_FEW,
        Task.ACSA: BASIC_PROMPT_ACSA + BASIC_PROMPT_ACSA_FR_FEW,
        Task.E2E: BASIC_PROMPT_E2E + BASIC_PROMPT_E2E_FR_FEW,
        Task.ACTE: BASIC_PROMPT_ACTE + BASIC_PROMPT_ACTE_FR_FEW,
    },
    LANG_DUTCH: {
        Task.TASD: BASIC_PROMPT_TASD + BASIC_PROMPT_TASD_NL_FEW,
        Task.ACSA: BASIC_PROMPT_ACSA + BASIC_PROMPT_ACSA_NL_FEW,
        Task.E2E: BASIC_PROMPT_E2E + BASIC_PROMPT_E2E_NL_FEW,
        Task.ACTE: BASIC_PROMPT_ACTE + BASIC_PROMPT_ACTE_NL_FEW,
    },
    LANG_RUSSIAN: {
        Task.TASD: BASIC_PROMPT_TASD + BASIC_PROMPT_TASD_RU_FEW,
        Task.ACSA: BASIC_PROMPT_ACSA + BASIC_PROMPT_ACSA_RU_FEW,
        Task.E2E: BASIC_PROMPT_E2E + BASIC_PROMPT_E2E_RU_FEW,
        Task.ACTE: BASIC_PROMPT_ACTE + BASIC_PROMPT_ACTE_RU_FEW,
    },
    LANG_TURKISH: {
        Task.TASD: BASIC_PROMPT_TASD + BASIC_PROMPT_TASD_TR_FEW,
        Task.ACSA: BASIC_PROMPT_ACSA + BASIC_PROMPT_ACSA_TR_FEW,
        Task.E2E: BASIC_PROMPT_E2E + BASIC_PROMPT_E2E_TR_FEW,
        Task.ACTE: BASIC_PROMPT_ACTE + BASIC_PROMPT_ACTE_TR_FEW,
    },
}


def get_instruction(task: Task, few_shot: bool = False, language: str = LANG_ENGLISH) -> str:
    """
    Get instruction for the task.

    :param task: task
    :param few_shot: few-shot
    :param language: language
    :return: instruction
    """
    if few_shot:
        if language not in FEW_SHOT_PROMPTS:
            raise ValueError(f"Language not supported: {language}")
        if task not in FEW_SHOT_PROMPTS[language]:
            raise ValueError(f"Task {task} not supported for language: {language}")
        return FEW_SHOT_PROMPTS[language][task]

    if task == Task.TASD:
        return BASIC_PROMPT_TASD
    elif task == Task.ACSA:
        return BASIC_PROMPT_ACSA
    elif task == Task.E2E:
        return BASIC_PROMPT_E2E
    elif task == Task.ACTE:
        return BASIC_PROMPT_ACTE
    else:
        raise ValueError(f"Task {task} not supported.")
