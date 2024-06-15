# TASD Prompt templates
BASIC_PROMPT_TASD_NL_FEW = f"""
Input: \"\"\"En het was zeker niet slecht als je opteert voor een café-restaurant ergens in het centrum van een stad.\"\"\"
Sentiment elements: [("null", "food quality", "neutral")]

Input: \"\"\"Qua wijn hebben ze enkel de duurdere flessen.\"\"\"
Sentiment elements: [("wijn", "drinks prices", "negative")]

Input: \"\"\"Het enige waarvan we de prijs niet hadden gevraagd (5 euro voor een glas wijn konden we nog aanvaarden) was een stukje focaccia van 5x5 cm oftwel 3 happen.\"\"\"
Sentiment elements: [("focaccia", "food style_options", "negative"), ("wijn", "drinks prices", "neutral")]

Input: \"\"\"Personeel kijkt je niet aan, kwakken je eten op je tafel en nemen niet eens de moeite om iets tegen je te zeggen.\"\"\"
Sentiment elements: [("Personeel", "service general", "negative")]

Input: \"\"\"Maar het Indische eten is heel lekker en de bediening ongelooflijk vriendelijk.\"\"\"
Sentiment elements: [("bediening", "service general", "positive"), ("Indische eten", "food quality", "positive")]

Input: \"\"\"Mij zien ze daar niet meer, veel te duur voor zo'n slechte kwaliteit!\"\"\"
Sentiment elements: [("null", "restaurant general", "negative"), ("null", "restaurant prices", "negative")]

Input: \"\"\"Eerlijke keuken en best geschikt voor in (ruim) gezelfschap maar nu niet direct de ideale gelegenheid voor een romantische tête à tête met je geliefde; een beetje no-nonsense en best leuk, bovendien een ruime kaart en zeker meer dan gemiddeld lekker\"\"\"
Sentiment elements: [("null", "food quality", "positive"), ("kaart", "food style_options", "positive"), ("keuken", "food quality", "positive")]

Input: \"\"\"Na verschillende malen lang gewacht te hebben op ons eten (+45min!!!), bestelden we de laatste maal videe en stoofvlees.\"\"\"
Sentiment elements: [("null", "service general", "negative")]

Input: \"\"\"Als tip zou bij het dessert ook een dessertwijn geschonken kunnen worden, waar alles was prima voor elkaar.\"\"\"
Sentiment elements: [("null", restaurant general, "positive")]

Input: \"\"\"De wijn was lekker, duurste fles nog aan modeste prijs van 30 eur.....\"\"\"
Sentiment elements: [("wijn", "drinks quality", "positive"), ("wijn", "drinks prices", "positive")]

"""

# ACSA Prompt templates
BASIC_PROMPT_ACSA_NL_FEW = f"""
Input: \"\"\"En het was zeker niet slecht als je opteert voor een café-restaurant ergens in het centrum van een stad.\"\"\"
Sentiment elements: [("food quality", "neutral")]

Input: \"\"\"Qua wijn hebben ze enkel de duurdere flessen.\"\"\"
Sentiment elements: [("drinks prices", "negative")]

Input: \"\"\"Het enige waarvan we de prijs niet hadden gevraagd (5 euro voor een glas wijn konden we nog aanvaarden) was een stukje focaccia van 5x5 cm oftwel 3 happen.\"\"\"
Sentiment elements: [("food style_options", "negative"), ("drinks prices", "neutral")]

Input: \"\"\"Personeel kijkt je niet aan, kwakken je eten op je tafel en nemen niet eens de moeite om iets tegen je te zeggen.\"\"\"
Sentiment elements: [("service general", "negative")]

Input: \"\"\"Maar het Indische eten is heel lekker en de bediening ongelooflijk vriendelijk.\"\"\"
Sentiment elements: [("service general", "positive"), ("food quality", "positive")]

Input: \"\"\"Mij zien ze daar niet meer, veel te duur voor zo'n slechte kwaliteit!\"\"\"
Sentiment elements: [("restaurant general", "negative"), ("restaurant prices", "negative")]

Input: \"\"\"Eerlijke keuken en best geschikt voor in (ruim) gezelfschap maar nu niet direct de ideale gelegenheid voor een romantische tête à tête met je geliefde; een beetje no-nonsense en best leuk, bovendien een ruime kaart en zeker meer dan gemiddeld lekker\"\"\"
Sentiment elements: [("food quality", "positive"), ("food style_options", "positive")]

Input: \"\"\"Na verschillende malen lang gewacht te hebben op ons eten (+45min!!!), bestelden we de laatste maal videe en stoofvlees.\"\"\"
Sentiment elements: [("service general", "negative")]

Input: \"\"\"Als tip zou bij het dessert ook een dessertwijn geschonken kunnen worden, waar alles was prima voor elkaar.\"\"\"
Sentiment elements: [(restaurant general, "positive")]

Input: \"\"\"De wijn was lekker, duurste fles nog aan modeste prijs van 30 eur.....\"\"\"
Sentiment elements: [("drinks quality", "positive"), ("drinks prices", "positive")]

"""

# E2E Prompt templates
BASIC_PROMPT_E2E_NL_FEW = f"""
Input: \"\"\"En het was zeker niet slecht als je opteert voor een café-restaurant ergens in het centrum van een stad.\"\"\"
Sentiment elements: [("null", "neutral")]

Input: \"\"\"Qua wijn hebben ze enkel de duurdere flessen.\"\"\"
Sentiment elements: [("wijn", "negative")]

Input: \"\"\"Het enige waarvan we de prijs niet hadden gevraagd (5 euro voor een glas wijn konden we nog aanvaarden) was een stukje focaccia van 5x5 cm oftwel 3 happen.\"\"\"
Sentiment elements: [("focaccia", "negative"), ("wijn", "neutral")]

Input: \"\"\"Personeel kijkt je niet aan, kwakken je eten op je tafel en nemen niet eens de moeite om iets tegen je te zeggen.\"\"\"
Sentiment elements: [("Personeel", "negative")]

Input: \"\"\"Maar het Indische eten is heel lekker en de bediening ongelooflijk vriendelijk.\"\"\"
Sentiment elements: [("bediening", "positive"), ("Indische eten", "positive")]

Input: \"\"\"Mij zien ze daar niet meer, veel te duur voor zo'n slechte kwaliteit!\"\"\"
Sentiment elements: [("null", "negative"), ("null", "negative")]

Input: \"\"\"Eerlijke keuken en best geschikt voor in (ruim) gezelfschap maar nu niet direct de ideale gelegenheid voor een romantische tête à tête met je geliefde; een beetje no-nonsense en best leuk, bovendien een ruime kaart en zeker meer dan gemiddeld lekker\"\"\"
Sentiment elements: [("null", "positive"), ("kaart", "positive"), ("keuken", "positive")]

Input: \"\"\"Na verschillende malen lang gewacht te hebben op ons eten (+45min!!!), bestelden we de laatste maal videe en stoofvlees.\"\"\"
Sentiment elements: [("null", "negative")]

Input: \"\"\"Als tip zou bij het dessert ook een dessertwijn geschonken kunnen worden, waar alles was prima voor elkaar.\"\"\"
Sentiment elements: [("null", "positive")]

Input: \"\"\"De wijn was lekker, duurste fles nog aan modeste prijs van 30 eur.....\"\"\"
Sentiment elements: [("wijn", "positive")]

"""

# ACTE Prompt templates
BASIC_PROMPT_ACTE_NL_FEW = f"""
Input: \"\"\"En het was zeker niet slecht als je opteert voor een café-restaurant ergens in het centrum van een stad.\"\"\"
Sentiment elements: [("null", "food quality")]

Input: \"\"\"Qua wijn hebben ze enkel de duurdere flessen.\"\"\"
Sentiment elements: [("wijn", "drinks prices")]

Input: \"\"\"Het enige waarvan we de prijs niet hadden gevraagd (5 euro voor een glas wijn konden we nog aanvaarden) was een stukje focaccia van 5x5 cm oftwel 3 happen.\"\"\"
Sentiment elements: [("focaccia", "food style_options"), ("wijn", "drinks prices")]

Input: \"\"\"Personeel kijkt je niet aan, kwakken je eten op je tafel en nemen niet eens de moeite om iets tegen je te zeggen.\"\"\"
Sentiment elements: [("Personeel", "service general")]

Input: \"\"\"Maar het Indische eten is heel lekker en de bediening ongelooflijk vriendelijk.\"\"\"
Sentiment elements: [("bediening", "service general"), ("Indische eten", "food quality")]

Input: \"\"\"Mij zien ze daar niet meer, veel te duur voor zo'n slechte kwaliteit!\"\"\"
Sentiment elements: [("null", "restaurant general"), ("null", "restaurant prices")]

Input: \"\"\"Eerlijke keuken en best geschikt voor in (ruim) gezelfschap maar nu niet direct de ideale gelegenheid voor een romantische tête à tête met je geliefde; een beetje no-nonsense en best leuk, bovendien een ruime kaart en zeker meer dan gemiddeld lekker\"\"\"
Sentiment elements: [("null", "food quality"), ("kaart", "food style_options"), ("keuken", "food quality")]

Input: \"\"\"Na verschillende malen lang gewacht te hebben op ons eten (+45min!!!), bestelden we de laatste maal videe en stoofvlees.\"\"\"
Sentiment elements: [("null", "service general")]

Input: \"\"\"Als tip zou bij het dessert ook een dessertwijn geschonken kunnen worden, waar alles was prima voor elkaar.\"\"\"
Sentiment elements: [("null", restaurant general)]

Input: \"\"\"De wijn was lekker, duurste fles nog aan modeste prijs van 30 eur.....\"\"\"
Sentiment elements: [("wijn", "drinks quality"), ("wijn", "drinks prices")]

"""
