# TASD Prompt templates
BASIC_PROMPT_TASD_CS_FEW = f"""
Input: \"\"\"Rumpsteak rozhodne nebyl medium, spis well done az done too much\"\"\"
Sentiment elements: [("Rumpsteak", "food quality", "negative")]

Input: \"\"\"Pěkné ubytování, dobrá snídaně a milý personal.\"\"\"
Sentiment elements: [("snídaně", "food quality", "positive"), ("personal", "service general", "positive")]

Input: \"\"\"z medium steaku se vyklubal z 1/3 syrovy doplneny o omacku pry s peprem\"\"\"
Sentiment elements: [("steaku", "food quality", "negative"), ("omacku", "food quality", "negative")]

Input: \"\"\"Mrzí mě to ale nic moc,nedoporucuji.\"\"\"
Sentiment elements: [("null", "restaurant general", "negative")]

Input: \"\"\"Snad i restaurace Na Scestí se jednou dostane na SPRÁVNOU CESTU ! ! !\"\"\"
Sentiment elements: [("restaurace Na Scestí", "restaurant general", "positive")]

Input: \"\"\"jedná se o zteplalou vodu (nejspíše z kohoutku) ve které se louhuje kolečko pomeranče, limetky a pár lístků máty\"\"\"
Sentiment elements: [("vodu", "drinks quality", "negative")]

Input: \"\"\"zhoreny gulas\"\"\"
Sentiment elements: [("gulas", "food quality", "negative")]

Input: \"\"\"Je potřeba na ně pozvat finanční úřad a hygienu a další\"\"\"
Sentiment elements: [("null", "restaurant miscellaneous", "negative")]

Input: \"\"\"Oslava byla super zásluhou dobrého jídla a milého personálu\"\"\"
Sentiment elements: [("jídla", "food quality", "positive"), ("personálu", "service general", "positive")]

Input: \"\"\"Restauraci Dynamo navštěvuji pravidelně a jsem naprosto spokojena\"\"\"
Sentiment elements: [("Restauraci Dynamo", "restaurant general", "positive")]

"""

# ACSA Prompt templates
BASIC_PROMPT_ACSA_CS_FEW = f"""
Input: \"\"\"Rumpsteak rozhodne nebyl medium, spis well done az done too much\"\"\"
Sentiment elements: [("food quality", "negative")]

Input: \"\"\"Pěkné ubytování, dobrá snídaně a milý personal.\"\"\"
Sentiment elements: [("food quality", "positive"), ("service general", "positive")]

Input: \"\"\"z medium steaku se vyklubal z 1/3 syrovy doplneny o omacku pry s peprem\"\"\"
Sentiment elements: [("food quality", "negative"), ("food quality", "negative")]

Input: \"\"\"Mrzí mě to ale nic moc,nedoporucuji.\"\"\"
Sentiment elements: [("restaurant general", "negative")]

Input: \"\"\"Snad i restaurace Na Scestí se jednou dostane na SPRÁVNOU CESTU ! ! !\"\"\"
Sentiment elements: [("restaurant general", "positive")]

Input: \"\"\"jedná se o zteplalou vodu (nejspíše z kohoutku) ve které se louhuje kolečko pomeranče, limetky a pár lístků máty\"\"\"
Sentiment elements: [("drinks quality", "negative")]

Input: \"\"\"zhoreny gulas\"\"\"
Sentiment elements: [("food quality", "negative")]

Input: \"\"\"Je potřeba na ně pozvat finanční úřad a hygienu a další\"\"\"
Sentiment elements: [("restaurant miscellaneous", "negative")]

Input: \"\"\"Oslava byla super zásluhou dobrého jídla a milého personálu\"\"\"
Sentiment elements: [("food quality", "positive"), ("service general", "positive")]

Input: \"\"\"Restauraci Dynamo navštěvuji pravidelně a jsem naprosto spokojena\"\"\"
Sentiment elements: [("restaurant general", "positive")]

"""

# E2E Prompt templates
BASIC_PROMPT_E2E_CS_FEW = f"""
Input: \"\"\"Rumpsteak rozhodne nebyl medium, spis well done az done too much\"\"\"
Sentiment elements: [("Rumpsteak", "negative")]

Input: \"\"\"Pěkné ubytování, dobrá snídaně a milý personal.\"\"\"
Sentiment elements: [("snídaně", "positive"), ("personal", "positive")]

Input: \"\"\"z medium steaku se vyklubal z 1/3 syrovy doplneny o omacku pry s peprem\"\"\"
Sentiment elements: [("steaku", "negative"), ("omacku", "negative")]

Input: \"\"\"Mrzí mě to ale nic moc,nedoporucuji.\"\"\"
Sentiment elements: [("null", "negative")]

Input: \"\"\"Snad i restaurace Na Scestí se jednou dostane na SPRÁVNOU CESTU ! ! !\"\"\"
Sentiment elements: [("restaurace Na Scestí", "positive")]

Input: \"\"\"jedná se o zteplalou vodu (nejspíše z kohoutku) ve které se louhuje kolečko pomeranče, limetky a pár lístků máty\"\"\"
Sentiment elements: [("vodu", "negative")]

Input: \"\"\"zhoreny gulas\"\"\"
Sentiment elements: [("gulas", "negative")]

Input: \"\"\"Je potřeba na ně pozvat finanční úřad a hygienu a další\"\"\"
Sentiment elements: [("null", "negative")]

Input: \"\"\"Oslava byla super zásluhou dobrého jídla a milého personálu\"\"\"
Sentiment elements: [("jídla", "positive"), ("personálu", "positive")]

Input: \"\"\"Restauraci Dynamo navštěvuji pravidelně a jsem naprosto spokojena\"\"\"
Sentiment elements: [("Restauraci Dynamo", "positive")]

"""

# ACTE Prompt templates
BASIC_PROMPT_ACTE_CS_FEW = f"""
Input: \"\"\"Rumpsteak rozhodne nebyl medium, spis well done az done too much\"\"\"
Sentiment elements: [("Rumpsteak", "food quality")]

Input: \"\"\"Pěkné ubytování, dobrá snídaně a milý personal.\"\"\"
Sentiment elements: [("snídaně", "food quality"), ("personal", "service general")]

Input: \"\"\"z medium steaku se vyklubal z 1/3 syrovy doplneny o omacku pry s peprem\"\"\"
Sentiment elements: [("steaku", "food quality"), ("omacku", "food quality")]

Input: \"\"\"Mrzí mě to ale nic moc,nedoporucuji.\"\"\"
Sentiment elements: [("null", "restaurant general")]

Input: \"\"\"Snad i restaurace Na Scestí se jednou dostane na SPRÁVNOU CESTU ! ! !\"\"\"
Sentiment elements: [("restaurace Na Scestí", "restaurant general")]

Input: \"\"\"jedná se o zteplalou vodu (nejspíše z kohoutku) ve které se louhuje kolečko pomeranče, limetky a pár lístků máty\"\"\"
Sentiment elements: [("vodu", "drinks quality")]

Input: \"\"\"zhoreny gulas\"\"\"
Sentiment elements: [("gulas", "food quality")]

Input: \"\"\"Je potřeba na ně pozvat finanční úřad a hygienu a další\"\"\"
Sentiment elements: [("null", "restaurant miscellaneous")]

Input: \"\"\"Oslava byla super zásluhou dobrého jídla a milého personálu\"\"\"
Sentiment elements: [("jídla", "food quality"), ("personálu", "service general")]

Input: \"\"\"Restauraci Dynamo navštěvuji pravidelně a jsem naprosto spokojena\"\"\"
Sentiment elements: [("Restauraci Dynamo", "restaurant general")]

"""
