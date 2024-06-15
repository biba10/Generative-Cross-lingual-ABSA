# TASD Prompt templates
BASIC_PROMPT_TASD_RU_FEW = f"""
Input: \"\"\"Девушки официантки все здороваются и улыбаются, интерьер потрясающий.\"\"\"
Sentiment elements: [("Девушки официантки", "service general", "positive"), ("интерьер", "ambience general", "positive")]

Input: \"\"\"В общем не в восторге.\"\"\"
Sentiment elements: [("null", "restaurant general", "negative")]

Input: \"\"\"Придем еще обязательно!\"\"\"
Sentiment elements: [("null", "restaurant general", "positive")]

Input: \"\"\"В общем уходить быстро не хочется...\"\"\"
Sentiment elements: [("null", "restaurant general", "positive")]

Input: \"\"\"В принципе праздник удался и недочеты не испортили общего впечатления.\"\"\"
Sentiment elements: [("null", "restaurant general", "positive")]

Input: \"\"\"Но даже с отличной кухней, я бы не вернулся туда, если бы общение официанта с посетителем не было таким душевным.\"\"\"
Sentiment elements: [("кухней", "food quality", "positive"), ("официанта", "service general", "positive")]

Input: \"\"\"Порции большие - то, что осталось упаковывают с собой.\"\"\"
Sentiment elements: [("Порции", "food quality", "positive"), ("null", "service general", "positive")]

Input: \"\"\"А еще, мы не стали заказывать десерты, т. к. наш ужин получился довольно сытным, но, приятно удивились, когда нас угостили вкусняшками в маленьких рюмочках, (по-моему панакота).\"\"\"
Sentiment elements: [("null", "service general", "positive"), ("панакота", "food quality", "positive")]

Input: \"\"\"Салаты вообще оказались вкуснейшими.\"\"\"
Sentiment elements: [("Салаты", "food quality", "positive")]

Input: \"\"\"Когда мы вошли нас очень мило встретили и предложили расположиться на первом этаже, не долго думая сели почти у входа рядом со стойкой бара.\"\"\"
Sentiment elements: [("null", "service general", "positive")]

"""

# ACSA Prompt templates
BASIC_PROMPT_ACSA_RU_FEW = f"""
Input: \"\"\"Девушки официантки все здороваются и улыбаются, интерьер потрясающий.\"\"\"
Sentiment elements: [("service general", "positive"), ("ambience general", "positive")]

Input: \"\"\"В общем не в восторге.\"\"\"
Sentiment elements: [("restaurant general", "negative")]

Input: \"\"\"Придем еще обязательно!\"\"\"
Sentiment elements: [("restaurant general", "positive")]

Input: \"\"\"В общем уходить быстро не хочется...\"\"\"
Sentiment elements: [("restaurant general", "positive")]

Input: \"\"\"В принципе праздник удался и недочеты не испортили общего впечатления.\"\"\"
Sentiment elements: [("restaurant general", "positive")]

Input: \"\"\"Но даже с отличной кухней, я бы не вернулся туда, если бы общение официанта с посетителем не было таким душевным.\"\"\"
Sentiment elements: [("food quality", "positive"), ("service general", "positive")]

Input: \"\"\"Порции большие - то, что осталось упаковывают с собой.\"\"\"
Sentiment elements: [("food quality", "positive"), ("service general", "positive")]

Input: \"\"\"А еще, мы не стали заказывать десерты, т. к. наш ужин получился довольно сытным, но, приятно удивились, когда нас угостили вкусняшками в маленьких рюмочках, (по-моему панакота).\"\"\"
Sentiment elements: [("service general", "positive"), ("food quality", "positive")]

Input: \"\"\"Салаты вообще оказались вкуснейшими.\"\"\"
Sentiment elements: [("food quality", "positive")]

Input: \"\"\"Когда мы вошли нас очень мило встретили и предложили расположиться на первом этаже, не долго думая сели почти у входа рядом со стойкой бара.\"\"\"
Sentiment elements: [("service general", "positive")]

"""

# E2E Prompt templates
BASIC_PROMPT_E2E_RU_FEW = f"""
Input: \"\"\"Девушки официантки все здороваются и улыбаются, интерьер потрясающий.\"\"\"
Sentiment elements: [("Девушки официантки", "positive"), ("интерьер", "positive")]

Input: \"\"\"В общем не в восторге.\"\"\"
Sentiment elements: [("null", "negative")]

Input: \"\"\"Придем еще обязательно!\"\"\"
Sentiment elements: [("null", "positive")]

Input: \"\"\"В общем уходить быстро не хочется...\"\"\"
Sentiment elements: [("null", "positive")]

Input: \"\"\"В принципе праздник удался и недочеты не испортили общего впечатления.\"\"\"
Sentiment elements: [("null", "positive")]

Input: \"\"\"Но даже с отличной кухней, я бы не вернулся туда, если бы общение официанта с посетителем не было таким душевным.\"\"\"
Sentiment elements: [("кухней", "positive"), ("официанта", "positive")]

Input: \"\"\"Порции большие - то, что осталось упаковывают с собой.\"\"\"
Sentiment elements: [("Порции", "positive"), ("null", "positive")]

Input: \"\"\"А еще, мы не стали заказывать десерты, т. к. наш ужин получился довольно сытным, но, приятно удивились, когда нас угостили вкусняшками в маленьких рюмочках, (по-моему панакота).\"\"\"
Sentiment elements: [("null", "positive"), ("панакота", "positive")]

Input: \"\"\"Салаты вообще оказались вкуснейшими.\"\"\"
Sentiment elements: [("Салаты", "positive")]

Input: \"\"\"Когда мы вошли нас очень мило встретили и предложили расположиться на первом этаже, не долго думая сели почти у входа рядом со стойкой бара.\"\"\"
Sentiment elements: [("null", "positive")]

"""

# ACTE Prompt templates
BASIC_PROMPT_ACTE_RU_FEW = f"""
Input: \"\"\"Девушки официантки все здороваются и улыбаются, интерьер потрясающий.\"\"\"
Sentiment elements: [("Девушки официантки", "service general"), ("интерьер", "ambience general")]

Input: \"\"\"В общем не в восторге.\"\"\"
Sentiment elements: [("null", "restaurant general")]

Input: \"\"\"Придем еще обязательно!\"\"\"
Sentiment elements: [("null", "restaurant general")]

Input: \"\"\"В общем уходить быстро не хочется...\"\"\"
Sentiment elements: [("null", "restaurant general")]

Input: \"\"\"В принципе праздник удался и недочеты не испортили общего впечатления.\"\"\"
Sentiment elements: [("null", "restaurant general")]

Input: \"\"\"Но даже с отличной кухней, я бы не вернулся туда, если бы общение официанта с посетителем не было таким душевным.\"\"\"
Sentiment elements: [("кухней", "food quality"), ("официанта", "service general")]

Input: \"\"\"Порции большие - то, что осталось упаковывают с собой.\"\"\"
Sentiment elements: [("Порции", "food quality"), ("null", "service general")]

Input: \"\"\"А еще, мы не стали заказывать десерты, т. к. наш ужин получился довольно сытным, но, приятно удивились, когда нас угостили вкусняшками в маленьких рюмочках, (по-моему панакота).\"\"\"
Sentiment elements: [("null", "service general"), ("панакота", "food quality")]

Input: \"\"\"Салаты вообще оказались вкуснейшими.\"\"\"
Sentiment elements: [("Салаты", "food quality")]

Input: \"\"\"Когда мы вошли нас очень мило встретили и предложили расположиться на первом этаже, не долго думая сели почти у входа рядом со стойкой бара.\"\"\"
Sentiment elements: [("null", "service general")]

"""
