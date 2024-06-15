# TASD Prompt templates
BASIC_PROMPT_TASD_EN_FEW = f"""
Input: \"\"\"My boyfriend had Prime Rib it was good .\"\"\"
Sentiment elements: [("Prime Rib", "food quality", "positive")]

Input: \"\"\"They should have called it mascarpone with chocolate chips-good but a far cry from what the name implies.\"\"\"
Sentiment elements: [("null", "food quality", "positive")]

Input: \"\"\"If you are looking for a good quality, cheap eats - this is the place.\"\"\"
Sentiment elements: [("eats", "food quality", "positive"), ("eats", "food prices", "positive")]

Input: \"\"\"The decor is night tho...but they REALLY need to clean that vent in the ceiling...its quite un-appetizing, and kills your effort to make this place look sleek and modern.\"\"\"
Sentiment elements: [("place", "ambience general", "negative"), ("decor", "ambience general", "positive"), ("vent", "ambience general", "negative")]

Input: \"\"\"My girlfriend, being slightly more aggressive, and having been equally disgusted causing her to throw out the remainder of her barely eaten meal, called back only to be informed that I was probably wrong and that it was most likely an oyster, and that we were also blacklisted from their restaurant.\"\"\"
Sentiment elements: [("meal", "food quality", "negative"), ("null", "service general", "negative")]

Input: \"\"\"You are not eating haut cuisine with subtle hints of whatever but: Cassuolet, Steake Fritte, Tripe Stew, etc; simple stuff.\"\"\"
Sentiment elements: [("null", "food style_options", "positive"), ("null", "food quality", "positive")]

Input: \"\"\"IT WAS HORRIBLE.\"\"\"
Sentiment elements: [("null", "restaurant general", "negative")]

Input: \"\"\"The food is wonderful, tasty and filling, and the service is professional and friendly.\"\"\"
Sentiment elements: [("food", "food quality", "positive"), ("food", "food style_options", "positive"), ("service", "service general", "positive")]

Input: \"\"\"Still, any quibbles about the bill were off-set by the pour-your-own measures of liquers which were courtesey of the house...\"\"\"
Sentiment elements: [("null", "restaurant prices", "neutral"), ("measures of liquers", "drinks style_options", "positive")]

Input: \"\"\"Even after they overcharged me the last time I was there.\"\"\"
Sentiment elements: [("null", "service general", "negative")]

"""

# ACSA Prompt templates
BASIC_PROMPT_ACSA_EN_FEW = f"""
Input: \"\"\"My boyfriend had Prime Rib it was good .\"\"\"
Sentiment elements: [("food quality", "positive")]

Input: \"\"\"They should have called it mascarpone with chocolate chips-good but a far cry from what the name implies.\"\"\"
Sentiment elements: [("food quality", "positive")]

Input: \"\"\"If you are looking for a good quality, cheap eats - this is the place.\"\"\"
Sentiment elements: [("food quality", "positive"), ("food prices", "positive")]

Input: \"\"\"The decor is night tho...but they REALLY need to clean that vent in the ceiling...its quite un-appetizing, and kills your effort to make this place look sleek and modern.\"\"\"
Sentiment elements: [("ambience general", "negative"), ("ambience general", "positive")]

Input: \"\"\"My girlfriend, being slightly more aggressive, and having been equally disgusted causing her to throw out the remainder of her barely eaten meal, called back only to be informed that I was probably wrong and that it was most likely an oyster, and that we were also blacklisted from their restaurant.\"\"\"
Sentiment elements: [("food quality", "negative"), ("service general", "negative")]

Input: \"\"\"You are not eating haut cuisine with subtle hints of whatever but: Cassuolet, Steake Fritte, Tripe Stew, etc; simple stuff.\"\"\"
Sentiment elements: [("food style_options", "positive"), ("food quality", "positive")]

Input: \"\"\"IT WAS HORRIBLE.\"\"\"
Sentiment elements: [("restaurant general", "negative")]

Input: \"\"\"The food is wonderful, tasty and filling, and the service is professional and friendly.\"\"\"
Sentiment elements: [("food quality", "positive"), ("food style_options", "positive"), ("service general", "positive")]

Input: \"\"\"Still, any quibbles about the bill were off-set by the pour-your-own measures of liquers which were courtesey of the house...\"\"\"
Sentiment elements: [("restaurant prices", "neutral"), ("drinks style_options", "positive")]

Input: \"\"\"Even after they overcharged me the last time I was there.\"\"\"
Sentiment elements: [("service general", "negative")]

"""

# E2E Prompt templates
BASIC_PROMPT_E2E_EN_FEW = f"""
Input: \"\"\"My boyfriend had Prime Rib it was good .\"\"\"
Sentiment elements: [("Prime Rib", "positive")]

Input: \"\"\"They should have called it mascarpone with chocolate chips-good but a far cry from what the name implies.\"\"\"
Sentiment elements: [("null", "positive")]

Input: \"\"\"If you are looking for a good quality, cheap eats - this is the place.\"\"\"
Sentiment elements: [("eats", "positive"), ("eats", "positive")]

Input: \"\"\"The decor is night tho...but they REALLY need to clean that vent in the ceiling...its quite un-appetizing, and kills your effort to make this place look sleek and modern.\"\"\"
Sentiment elements: [("place", "negative"), ("decor", "positive"), ("vent", "negative")]

Input: \"\"\"My girlfriend, being slightly more aggressive, and having been equally disgusted causing her to throw out the remainder of her barely eaten meal, called back only to be informed that I was probably wrong and that it was most likely an oyster, and that we were also blacklisted from their restaurant.\"\"\"
Sentiment elements: [("meal", "negative"), ("null", "negative")]

Input: \"\"\"You are not eating haut cuisine with subtle hints of whatever but: Cassuolet, Steake Fritte, Tripe Stew, etc; simple stuff.\"\"\"
Sentiment elements: [("null", "positive")]

Input: \"\"\"IT WAS HORRIBLE.\"\"\"
Sentiment elements: [("null",  "negative")]

Input: \"\"\"The food is wonderful, tasty and filling, and the service is professional and friendly.\"\"\"
Sentiment elements: [("food", "positive"), ("service", "positive")]

Input: \"\"\"Still, any quibbles about the bill were off-set by the pour-your-own measures of liquers which were courtesey of the house...\"\"\"
Sentiment elements: [("null", "neutral"), ("measures of liquers", "positive")]

Input: \"\"\"Even after they overcharged me the last time I was there.\"\"\"
Sentiment elements: [("null", "negative")]

"""

# ACTE Prompt templates
BASIC_PROMPT_ACTE_EN_FEW = f"""
Input: \"\"\"My boyfriend had Prime Rib it was good .\"\"\"
Sentiment elements: [("Prime Rib", "food quality")]

Input: \"\"\"They should have called it mascarpone with chocolate chips-good but a far cry from what the name implies.\"\"\"
Sentiment elements: [("null", "food quality")]

Input: \"\"\"If you are looking for a good quality, cheap eats - this is the place.\"\"\"
Sentiment elements: [("eats", "food quality"), ("eats", "food prices")]

Input: \"\"\"The decor is night tho...but they REALLY need to clean that vent in the ceiling...its quite un-appetizing, and kills your effort to make this place look sleek and modern.\"\"\"
Sentiment elements: [("place", "ambience general"), ("decor", "ambience general"), ("vent", "ambience general")]

Input: \"\"\"My girlfriend, being slightly more aggressive, and having been equally disgusted causing her to throw out the remainder of her barely eaten meal, called back only to be informed that I was probably wrong and that it was most likely an oyster, and that we were also blacklisted from their restaurant.\"\"\"
Sentiment elements: [("meal", "food quality"), ("null", "service general")]

Input: \"\"\"You are not eating haut cuisine with subtle hints of whatever but: Cassuolet, Steake Fritte, Tripe Stew, etc; simple stuff.\"\"\"
Sentiment elements: [("null", "food style_options"), ("null", "food quality")]

Input: \"\"\"IT WAS HORRIBLE.\"\"\"
Sentiment elements: [("null", "restaurant general")]

Input: \"\"\"The food is wonderful, tasty and filling, and the service is professional and friendly.\"\"\"
Sentiment elements: [("food", "food quality"), ("food", "food style_options"), ("service", "service general")]

Input: \"\"\"Still, any quibbles about the bill were off-set by the pour-your-own measures of liquers which were courtesey of the house...\"\"\"
Sentiment elements: [("null", "restaurant prices"), ("measures of liquers", "drinks style_options")]

Input: \"\"\"Even after they overcharged me the last time I was there.\"\"\"
Sentiment elements: [("null", "service general")]

"""
