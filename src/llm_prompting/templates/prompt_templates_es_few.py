# TASD Prompt templates
BASIC_PROMPT_TASD_ES_FEW = f"""
Input: \"\"\"Fabuloso comienzo de año con una comida estupenda y una estancia genial tanto por el ambiente como por el agradable trato recibido, como en anteriores veces, repetiremos.\"\"\"
Sentiment elements: [("comida", "food quality", "positive"), ("ambiente", "ambience general", "positive"), ("trato", "service general", "positive")]

Input: \"\"\"Generoso en cantidad, pero la calidad de la carne dejaba un poco que desear.\"\"\"
Sentiment elements: [("carne", "food quality", "negative")]

Input: \"\"\"El tomate, como entrante,  es algo especial a destacar.\"\"\"
Sentiment elements: [("tomate", "food quality", "positive")]

Input: \"\"\"Muy recomendable.\"\"\"
Sentiment elements: [("null", "restaurant general", "positive")]

Input: \"\"\"Por lo que cuesta el menu, la comida no esta nada mal.\"\"\"
Sentiment elements: [("comida", "food quality", "positive")]

Input: \"\"\"Puedes encontrar desde un chuletón bien trabajado, pasando por unos pescados en su punto justo o irte a menús mas elaborados, llenos de sabor y contrastes sin adulteraciones en el sabor.\"\"\"
Sentiment elements: [("chuletón", "food quality", "positive"), ("pescados", "food quality", "positive"), ("menús", "food style_options", "positive")]

Input: \"\"\"El local muy bien, la calidad de la comida muy buena, servicio  bueno, pero la presentación de los platos, ya que se lo cobra, deja mucho que desear.\"\"\"
Sentiment elements: [("local", "ambience general", "positive"), ("comida", "food quality", "positive"), ("servicio", "service general", "neutral")]

Input: \"\"\"Para finalizar con los previos antes de entrar con la comida un comentario: un poquito de musica de ambiente daria intimidad, se oia todo lo que se decia en las otras 5 mesas que estaban ocupadas (todas ellas con parejas con cupones de descuento como nosotros).\"\"\"
Sentiment elements: [("null", "ambience general", "negative")]

Input: \"\"\"Comida abundante, buena relación calidad-precio si pides entrante+segundo se puede cenar por unos 12 euros\"\"\"
Sentiment elements: [("Comida", "food style_options", "positive"), ("null", "food prices", "positive")]

Input: \"\"\"Personalmente este restaurante, ya por el servicio y la actitud de los camareros, del maitrê es excelente, creo que este restaurante se merece una estrella michelin, ya que se come muy bien, son unas grandes personas.\"\"\"
Sentiment elements: [("camareros", "service general", "positive"), ("maitrê", "service general", "positive"), ("restaurante", "restaurant general", "positive"), ("restaurante", "food quality", "positive"), ("personas", "service general", "positive")]

"""

# ACSA Prompt templates
BASIC_PROMPT_ACSA_ES_FEW = f"""
Input: \"\"\"Fabuloso comienzo de año con una comida estupenda y una estancia genial tanto por el ambiente como por el agradable trato recibido, como en anteriores veces, repetiremos.\"\"\"
Sentiment elements: [("food quality", "positive"), ("ambience general", "positive"), ("service general", "positive")]

Input: \"\"\"Generoso en cantidad, pero la calidad de la carne dejaba un poco que desear.\"\"\"
Sentiment elements: [("food quality", "negative")]

Input: \"\"\"El tomate, como entrante,  es algo especial a destacar.\"\"\"
Sentiment elements: [("food quality", "positive")]

Input: \"\"\"Muy recomendable.\"\"\"
Sentiment elements: [("restaurant general", "positive")]

Input: \"\"\"Por lo que cuesta el menu, la comida no esta nada mal.\"\"\"
Sentiment elements: [("food quality", "positive")]

Input: \"\"\"Puedes encontrar desde un chuletón bien trabajado, pasando por unos pescados en su punto justo o irte a menús mas elaborados, llenos de sabor y contrastes sin adulteraciones en el sabor.\"\"\"
Sentiment elements: [("food quality", "positive"), ("food quality", "positive"), ("food style_options", "positive")]

Input: \"\"\"El local muy bien, la calidad de la comida muy buena, servicio  bueno, pero la presentación de los platos, ya que se lo cobra, deja mucho que desear.\"\"\"
Sentiment elements: [("ambience general", "positive"), ("food quality", "positive"), ("service general", "neutral")]

Input: \"\"\"Para finalizar con los previos antes de entrar con la comida un comentario: un poquito de musica de ambiente daria intimidad, se oia todo lo que se decia en las otras 5 mesas que estaban ocupadas (todas ellas con parejas con cupones de descuento como nosotros).\"\"\"
Sentiment elements: [("ambience general", "negative")]

Input: \"\"\"Comida abundante, buena relación calidad-precio si pides entrante+segundo se puede cenar por unos 12 euros\"\"\"
Sentiment elements: [("food style_options", "positive"), ("food prices", "positive")]

Input: \"\"\"Personalmente este restaurante, ya por el servicio y la actitud de los camareros, del maitrê es excelente, creo que este restaurante se merece una estrella michelin, ya que se come muy bien, son unas grandes personas.\"\"\"
Sentiment elements: [("service general", "positive"), ("restaurant general", "positive"), ("food quality", "positive")]

"""

# E2E Prompt templates
BASIC_PROMPT_E2E_ES_FEW = f"""
Input: \"\"\"Fabuloso comienzo de año con una comida estupenda y una estancia genial tanto por el ambiente como por el agradable trato recibido, como en anteriores veces, repetiremos.\"\"\"
Sentiment elements: [("comida", "positive"), ("ambiente", "positive"), ("trato", "positive")]

Input: \"\"\"Generoso en cantidad, pero la calidad de la carne dejaba un poco que desear.\"\"\"
Sentiment elements: [("carne", "negative")]

Input: \"\"\"El tomate, como entrante,  es algo especial a destacar.\"\"\"
Sentiment elements: [("tomate", "positive")]

Input: \"\"\"Muy recomendable.\"\"\"
Sentiment elements: [("null", "positive")]

Input: \"\"\"Por lo que cuesta el menu, la comida no esta nada mal.\"\"\"
Sentiment elements: [("comida", "positive")]

Input: \"\"\"Puedes encontrar desde un chuletón bien trabajado, pasando por unos pescados en su punto justo o irte a menús mas elaborados, llenos de sabor y contrastes sin adulteraciones en el sabor.\"\"\"
Sentiment elements: [("chuletón", "positive"), ("pescados", "positive"), ("menús", "positive")]

Input: \"\"\"El local muy bien, la calidad de la comida muy buena, servicio  bueno, pero la presentación de los platos, ya que se lo cobra, deja mucho que desear.\"\"\"
Sentiment elements: [("local", "positive"), ("comida", "positive"), ("servicio", "neutral")]

Input: \"\"\"Para finalizar con los previos antes de entrar con la comida un comentario: un poquito de musica de ambiente daria intimidad, se oia todo lo que se decia en las otras 5 mesas que estaban ocupadas (todas ellas con parejas con cupones de descuento como nosotros).\"\"\"
Sentiment elements: [("null", "negative")]

Input: \"\"\"Comida abundante, buena relación calidad-precio si pides entrante+segundo se puede cenar por unos 12 euros\"\"\"
Sentiment elements: [("Comida", "positive"), ("null", "positive")]

Input: \"\"\"Personalmente este restaurante, ya por el servicio y la actitud de los camareros, del maitrê es excelente, creo que este restaurante se merece una estrella michelin, ya que se come muy bien, son unas grandes personas.\"\"\"
Sentiment elements: [("camareros", "positive"), ("maitrê", "positive"), ("restaurante", "positive"), ("personas", "positive")]

"""

# ACTE Prompt templates
BASIC_PROMPT_ACTE_ES_FEW = f"""
Input: \"\"\"Fabuloso comienzo de año con una comida estupenda y una estancia genial tanto por el ambiente como por el agradable trato recibido, como en anteriores veces, repetiremos.\"\"\"
Sentiment elements: [("comida", "food quality"), ("ambiente", "ambience general"), ("trato", "service general")]

Input: \"\"\"Generoso en cantidad, pero la calidad de la carne dejaba un poco que desear.\"\"\"
Sentiment elements: [("carne", "food quality")]

Input: \"\"\"El tomate, como entrante,  es algo especial a destacar.\"\"\"
Sentiment elements: [("tomate", "food quality")]

Input: \"\"\"Muy recomendable.\"\"\"
Sentiment elements: [("null", "restaurant general")]

Input: \"\"\"Por lo que cuesta el menu, la comida no esta nada mal.\"\"\"
Sentiment elements: [("comida", "food quality")]

Input: \"\"\"Puedes encontrar desde un chuletón bien trabajado, pasando por unos pescados en su punto justo o irte a menús mas elaborados, llenos de sabor y contrastes sin adulteraciones en el sabor.\"\"\"
Sentiment elements: [("chuletón", "food quality"), ("pescados", "food quality"), ("menús", "food style_options")]

Input: \"\"\"El local muy bien, la calidad de la comida muy buena, servicio  bueno, pero la presentación de los platos, ya que se lo cobra, deja mucho que desear.\"\"\"
Sentiment elements: [("local", "ambience general", "positive"), ("comida", "food quality"), ("servicio", "service general")]

Input: \"\"\"Para finalizar con los previos antes de entrar con la comida un comentario: un poquito de musica de ambiente daria intimidad, se oia todo lo que se decia en las otras 5 mesas que estaban ocupadas (todas ellas con parejas con cupones de descuento como nosotros).\"\"\"
Sentiment elements: [("null", "ambience general")]

Input: \"\"\"Comida abundante, buena relación calidad-precio si pides entrante+segundo se puede cenar por unos 12 euros\"\"\"
Sentiment elements: [("Comida", "food style_options"), ("null", "food prices")]

Input: \"\"\"Personalmente este restaurante, ya por el servicio y la actitud de los camareros, del maitrê es excelente, creo que este restaurante se merece una estrella michelin, ya que se come muy bien, son unas grandes personas.\"\"\"
Sentiment elements: [("camareros", "service general"), ("maitrê", "service general"), ("restaurante", "restaurant general"), ("restaurante", "food quality"), ("personas", "service general")]

"""
