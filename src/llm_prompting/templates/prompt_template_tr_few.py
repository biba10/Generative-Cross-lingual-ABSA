# TASD Prompt templates
BASIC_PROMPT_TASD_TR_FEW = f"""
Input: \"\"\"Kalabalik zamanlarda servis fazlasiyla aksiyor.\"\"\"
Sentiment elements: [("servis", "service general", "negative")]

Input: \"\"\"Içki servisi kaldirilmis olmasi garibime gitti açikcasi.\"\"\"
Sentiment elements: [("Içki servisi", "drinks style_options", "negative")]

Input: \"\"\"beşiktaşa otobüsle zincirlikuyudan geldiğinizde gözünüze direk çarpan bir mekan.\"\"\"
Sentiment elements: [("mekan", "location general", "positive")]

Input: \"\"\"Gelen tavuk Wrapten 5 dakika sonra biz kofte durum istememize ragmen tavuk sis dürümu geldi durumu soyledik ve degistirdiler.\"\"\"
Sentiment elements: [("kofte durum", "service general", "negative")]

Input: \"\"\"Jumbo combo ve kola istedim ama börek ve tavuk parçaları dışındaki tatlar bana göre değildi.\"\"\"
Sentiment elements: [("börek ve tavuk parçaları", "food quality", "positive"), ("Jumbo combo ve kola", "food quality", "neutral")]

Input: \"\"\"Cerkez yemekleri sevenler icin guzel bir tercih .\"\"\"
Sentiment elements: [("Cerkez yemekleri", "food quality", "positive")]

Input: \"\"\"Cok guzel sirin bir mekan yemekler kokteyliler super manzara cok iyi, tabiki bu guzel seyler icin sira bekleyebilirsiniz.\"\"\"
Sentiment elements: [("manzara", "ambience general", "positive"), ("yemekler", "food quality", "positive"), ("kokteyliler", "drinks quality", "positive"), ("mekan", "ambience general", "positive")]

Input: \"\"\"Sunumu beğendik.\"\"\"
Sentiment elements: [("Sunumu", "service general", "positive")]

Input: \"\"\"servisi gayet iyi.\"\"\"
Sentiment elements: [("servisi", "service general", "positive")]

Input: \"\"\"Mekan küçük ama şirin, aşk-ı memnu'yu daha önce hiç bitiremedim ama gerçekten çok güzel bir tadı var.\"\"\"
Sentiment elements: [("aşk-ı memnu", "food quality", "positive"), ("Mekan", "ambience general", "positive")]

"""

# ACSA Prompt templates
BASIC_PROMPT_ACSA_TR_FEW = f"""
Input: \"\"\"Kalabalik zamanlarda servis fazlasiyla aksiyor.\"\"\"
Sentiment elements: [("service general", "negative")]

Input: \"\"\"Içki servisi kaldirilmis olmasi garibime gitti açikcasi.\"\"\"
Sentiment elements: [("drinks style_options", "negative")]

Input: \"\"\"beşiktaşa otobüsle zincirlikuyudan geldiğinizde gözünüze direk çarpan bir mekan.\"\"\"
Sentiment elements: [("location general", "positive")]

Input: \"\"\"Gelen tavuk Wrapten 5 dakika sonra biz kofte durum istememize ragmen tavuk sis dürümu geldi durumu soyledik ve degistirdiler.\"\"\"
Sentiment elements: [("service general", "negative")]

Input: \"\"\"Jumbo combo ve kola istedim ama börek ve tavuk parçaları dışındaki tatlar bana göre değildi.\"\"\"
Sentiment elements: [("food quality", "positive"), ("food quality", "neutral")]

Input: \"\"\"Cerkez yemekleri sevenler icin guzel bir tercih .\"\"\"
Sentiment elements: [("food quality", "positive")]

Input: \"\"\"Cok guzel sirin bir mekan yemekler kokteyliler super manzara cok iyi, tabiki bu guzel seyler icin sira bekleyebilirsiniz.\"\"\"
Sentiment elements: [("ambience general", "positive"), ("food quality", "positive"), ("drinks quality", "positive"), ("ambience general", "positive")]

Input: \"\"\"Sunumu beğendik.\"\"\"
Sentiment elements: [("service general", "positive")]

Input: \"\"\"servisi gayet iyi.\"\"\"
Sentiment elements: [("service general", "positive")]

Input: \"\"\"Mekan küçük ama şirin, aşk-ı memnu'yu daha önce hiç bitiremedim ama gerçekten çok güzel bir tadı var.\"\"\"
Sentiment elements: [("food quality", "positive"), ("ambience general", "positive")]

"""

# E2E Prompt templates
BASIC_PROMPT_E2E_TR_FEW = f"""
Input: \"\"\"Kalabalik zamanlarda servis fazlasiyla aksiyor.\"\"\"
Sentiment elements: [("servis", "negative")]

Input: \"\"\"Içki servisi kaldirilmis olmasi garibime gitti açikcasi.\"\"\"
Sentiment elements: [("Içki servisi", "negative")]

Input: \"\"\"beşiktaşa otobüsle zincirlikuyudan geldiğinizde gözünüze direk çarpan bir mekan.\"\"\"
Sentiment elements: [("mekan", "positive")]

Input: \"\"\"Gelen tavuk Wrapten 5 dakika sonra biz kofte durum istememize ragmen tavuk sis dürümu geldi durumu soyledik ve degistirdiler.\"\"\"
Sentiment elements: [("kofte durum", "negative")]

Input: \"\"\"Jumbo combo ve kola istedim ama börek ve tavuk parçaları dışındaki tatlar bana göre değildi.\"\"\"
Sentiment elements: [("börek ve tavuk parçaları", "positive"), ("Jumbo combo ve kola", "neutral")]

Input: \"\"\"Cerkez yemekleri sevenler icin guzel bir tercih .\"\"\"
Sentiment elements: [("Cerkez yemekleri", "positive")]

Input: \"\"\"Cok guzel sirin bir mekan yemekler kokteyliler super manzara cok iyi, tabiki bu guzel seyler icin sira bekleyebilirsiniz.\"\"\"
Sentiment elements: [("manzara", "positive"), ("yemekler", "positive"), ("kokteyliler", "positive"), ("mekan", "positive")]

Input: \"\"\"Sunumu beğendik.\"\"\"
Sentiment elements: [("Sunumu", "positive")]

Input: \"\"\"servisi gayet iyi.\"\"\"
Sentiment elements: [("servisi", "positive")]

Input: \"\"\"Mekan küçük ama şirin, aşk-ı memnu'yu daha önce hiç bitiremedim ama gerçekten çok güzel bir tadı var.\"\"\"
Sentiment elements: [("aşk-ı memnu", "positive"), ("Mekan", "positive")]

"""

# ACTE Prompt templates
BASIC_PROMPT_ACTE_TR_FEW = f"""
Input: \"\"\"Kalabalik zamanlarda servis fazlasiyla aksiyor.\"\"\"
Sentiment elements: [("servis", "service general")]

Input: \"\"\"Içki servisi kaldirilmis olmasi garibime gitti açikcasi.\"\"\"
Sentiment elements: [("Içki servisi", "drinks style_options")]

Input: \"\"\"beşiktaşa otobüsle zincirlikuyudan geldiğinizde gözünüze direk çarpan bir mekan.\"\"\"
Sentiment elements: [("mekan", "location general")]

Input: \"\"\"Gelen tavuk Wrapten 5 dakika sonra biz kofte durum istememize ragmen tavuk sis dürümu geldi durumu soyledik ve degistirdiler.\"\"\"
Sentiment elements: [("kofte durum", "service general")]

Input: \"\"\"Jumbo combo ve kola istedim ama börek ve tavuk parçaları dışındaki tatlar bana göre değildi.\"\"\"
Sentiment elements: [("börek ve tavuk parçaları", "food quality"), ("Jumbo combo ve kola", "food quality")]

Input: \"\"\"Cerkez yemekleri sevenler icin guzel bir tercih .\"\"\"
Sentiment elements: [("Cerkez yemekleri", "food quality")]

Input: \"\"\"Cok guzel sirin bir mekan yemekler kokteyliler super manzara cok iyi, tabiki bu guzel seyler icin sira bekleyebilirsiniz.\"\"\"
Sentiment elements: [("manzara", "ambience general"), ("yemekler", "food quality"), ("kokteyliler", "drinks quality"), ("mekan", "ambience general")]

Input: \"\"\"Sunumu beğendik.\"\"\"
Sentiment elements: [("Sunumu", "service general")]

Input: \"\"\"servisi gayet iyi.\"\"\"
Sentiment elements: [("servisi", "service general")]

Input: \"\"\"Mekan küçük ama şirin, aşk-ı memnu'yu daha önce hiç bitiremedim ama gerçekten çok güzel bir tadı var.\"\"\"
Sentiment elements: [("aşk-ı memnu", "food quality"), ("Mekan", "ambience general")]

"""
