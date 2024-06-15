# TASD Prompt templates
BASIC_PROMPT_TASD_FR_FEW = f"""
Please carefully follow the instructions. Ensure that aspect terms are recognized as exact matches in the review or are "null" for implicit aspects. Ensure that aspect categories are from the available categories. Ensure that sentiment polarities are from the available polarities.

Recognize all sentiment elements with their corresponding aspect terms, aspect categories, and sentiment polarity. Return the answer as a Python list of tuple, with each tuple in the format ("aspect term", "aspect category", "sentiment polarity"), without any additional content.

Input: \"\"\"L'ambiance est agréable dans ce petit restaurant à la déco sobre et élégante.\"\"\"
Sentiment elements: [("ambiance", "ambience general", "positive"), ("déco", "ambience general", "positive")]

Input: \"\"\"Ni brasserie ni restaurant.\"\"\"
Sentiment elements: [("null", "restaurant general", "neutral")]

Input: \"\"\"Pour les autres plat, aucun légume vert n'est proposé en accompagnement il faut se contenter de frites, purée de pommes de terre risotto ou les légumes du pot au feu....Le service inexistant, à la question qu'y a-t-il dans le café gourmand la réponse fut ça dépend...\"\"\"
Sentiment elements: [("plat", "food style_options", "negative"), ("service", "service general", "negative")]

Input: \"\"\"Prix élevé pour la prestation et surtout la netteté laisse à désirer dans ce restaurant.\"\"\"
Sentiment elements: [("null", "restaurant prices", "negative"), ("restaurant", "restaurant miscellaneous", "negative")]

Input: \"\"\"Quand aux plats, n'y allez pas pour faire une découverte originale, que du "classique"....\"\"\"
Sentiment elements: [("plats", "food style_options", "negative")]

Input: \"\"\"Demande de changement de place car nous étions près de la porte mais nous sommes restés sans réponse.\"\"\"
Sentiment elements: [("null", "service general", "neutral")]

Input: \"\"\"La carte pourtant est bien garnie et alléchante.\"\"\"
Sentiment elements: [("carte", "food style_options", "positive")]

Input: \"\"\"Des produits frais, bien assaisonnés et préparés avec respect de la cuisine française.\"\"\"
Sentiment elements: [("produits", "food quality", "positive"))]

Input: \"\"\"Un menu enfant avec un seul choix de plat et pas possible d'avoir autre chose, nous avons dû commander la pintade beaucoup trop copieuse pour un enfant de 8 ans qui nou sa été compté en plat adulte à 25 euros !\"\"\"
Sentiment elements: [("menu enfant", "food style_options", "negative"), ("pintade", "food style_options", "negative"), ("pintade", "food prices", "negative")]

Input: \"\"\"La synchronisation (moment pour servir les différents plats) était tout à fait correcte.\"\"\"
Sentiment elements: [("synchronisation", "service general", "positive")]

"""

# ACSA Prompt templates
BASIC_PROMPT_ACSA_FR_FEW = f"""
Input: \"\"\"L'ambiance est agréable dans ce petit restaurant à la déco sobre et élégante.\"\"\"
Sentiment elements: [("ambience general", "positive"), ("ambience general", "positive")]

Input: \"\"\"Ni brasserie ni restaurant.\"\"\"
Sentiment elements: [("restaurant general", "neutral")]

Input: \"\"\"Pour les autres plat, aucun légume vert n'est proposé en accompagnement il faut se contenter de frites, purée de pommes de terre risotto ou les légumes du pot au feu....Le service inexistant, à la question qu'y a-t-il dans le café gourmand la réponse fut ça dépend...\"\"\"
Sentiment elements: [("food style_options", "negative"), ("service general", "negative")]

Input: \"\"\"Prix élevé pour la prestation et surtout la netteté laisse à désirer dans ce restaurant.\"\"\"
Sentiment elements: [("restaurant prices", "negative"), ("restaurant miscellaneous", "negative")]

Input: \"\"\"Quand aux plats, n'y allez pas pour faire une découverte originale, que du "classique"....\"\"\"
Sentiment elements: [("food style_options", "negative")]

Input: \"\"\"Demande de changement de place car nous étions près de la porte mais nous sommes restés sans réponse.\"\"\"
Sentiment elements: [("service general", "neutral")]

Input: \"\"\"La carte pourtant est bien garnie et alléchante.\"\"\"
Sentiment elements: [("food style_options", "positive")]

Input: \"\"\"Des produits frais, bien assaisonnés et préparés avec respect de la cuisine française.\"\"\"
Sentiment elements: [("food quality", "positive"))]

Input: \"\"\"Un menu enfant avec un seul choix de plat et pas possible d'avoir autre chose, nous avons dû commander la pintade beaucoup trop copieuse pour un enfant de 8 ans qui nou sa été compté en plat adulte à 25 euros !\"\"\"
Sentiment elements: [("food style_options", "negative"), ("food prices", "negative")]

Input: \"\"\"La synchronisation (moment pour servir les différents plats) était tout à fait correcte.\"\"\"
Sentiment elements: [("service general", "positive")]

"""

# E2E Prompt templates
BASIC_PROMPT_E2E_FR_FEW = f"""
Input: \"\"\"L'ambiance est agréable dans ce petit restaurant à la déco sobre et élégante.\"\"\"
Sentiment elements: [("ambiance", "positive"), ("déco", "positive")]

Input: \"\"\"Ni brasserie ni restaurant.\"\"\"
Sentiment elements: [("null", "neutral")]

Input: \"\"\"Pour les autres plat, aucun légume vert n'est proposé en accompagnement il faut se contenter de frites, purée de pommes de terre risotto ou les légumes du pot au feu....Le service inexistant, à la question qu'y a-t-il dans le café gourmand la réponse fut ça dépend...\"\"\"
Sentiment elements: [("plat", "negative"), ("service", "negative")]

Input: \"\"\"Prix élevé pour la prestation et surtout la netteté laisse à désirer dans ce restaurant.\"\"\"
Sentiment elements: [("null", "negative"), ("restaurant", "negative")]

Input: \"\"\"Quand aux plats, n'y allez pas pour faire une découverte originale, que du "classique"....\"\"\"
Sentiment elements: [("plats", "negative")]

Input: \"\"\"Demande de changement de place car nous étions près de la porte mais nous sommes restés sans réponse.\"\"\"
Sentiment elements: [("null", "neutral")]

Input: \"\"\"La carte pourtant est bien garnie et alléchante.\"\"\"
Sentiment elements: [("carte", "positive")]

Input: \"\"\"Des produits frais, bien assaisonnés et préparés avec respect de la cuisine française.\"\"\"
Sentiment elements: [("produits", "positive"))]

Input: \"\"\"Un menu enfant avec un seul choix de plat et pas possible d'avoir autre chose, nous avons dû commander la pintade beaucoup trop copieuse pour un enfant de 8 ans qui nou sa été compté en plat adulte à 25 euros !\"\"\"
Sentiment elements: [("menu enfant", "negative"), ("pintade", "negative")]

Input: \"\"\"La synchronisation (moment pour servir les différents plats) était tout à fait correcte.\"\"\"
Sentiment elements: [("synchronisation", "positive")]

"""

# ACTE Prompt templates
BASIC_PROMPT_ACTE_FR_FEW = f"""
Input: \"\"\"L'ambiance est agréable dans ce petit restaurant à la déco sobre et élégante.\"\"\"
Sentiment elements: [("ambiance", "ambience general"), ("déco", "ambience general")]

Input: \"\"\"Ni brasserie ni restaurant.\"\"\"
Sentiment elements: [("null", "restaurant general")]

Input: \"\"\"Pour les autres plat, aucun légume vert n'est proposé en accompagnement il faut se contenter de frites, purée de pommes de terre risotto ou les légumes du pot au feu....Le service inexistant, à la question qu'y a-t-il dans le café gourmand la réponse fut ça dépend...\"\"\"
Sentiment elements: [("plat", "food style_options"), ("service", "service general")]

Input: \"\"\"Prix élevé pour la prestation et surtout la netteté laisse à désirer dans ce restaurant.\"\"\"
Sentiment elements: [("null", "restaurant prices"), ("restaurant", "restaurant miscellaneous")]

Input: \"\"\"Quand aux plats, n'y allez pas pour faire une découverte originale, que du "classique"....\"\"\"
Sentiment elements: [("plats", "food style_options")]

Input: \"\"\"Demande de changement de place car nous étions près de la porte mais nous sommes restés sans réponse.\"\"\"
Sentiment elements: [("null", "service general")]

Input: \"\"\"La carte pourtant est bien garnie et alléchante.\"\"\"
Sentiment elements: [("carte", "food style_options")]

Input: \"\"\"Des produits frais, bien assaisonnés et préparés avec respect de la cuisine française.\"\"\"
Sentiment elements: [("produits", "food quality"))]

Input: \"\"\"Un menu enfant avec un seul choix de plat et pas possible d'avoir autre chose, nous avons dû commander la pintade beaucoup trop copieuse pour un enfant de 8 ans qui nou sa été compté en plat adulte à 25 euros !\"\"\"
Sentiment elements: [("menu enfant", "food style_options"), ("pintade", "food style_options"), ("pintade", "food prices")]

Input: \"\"\"La synchronisation (moment pour servir les différents plats) était tout à fait correcte.\"\"\"
Sentiment elements: [("synchronisation", "service general")]

"""
