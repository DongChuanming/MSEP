from preannotation_rules import preannotate_taba
from CamemBERT_medical_status_extractors import medical_status_annotation
from preprocessing import sentence_segmentation

id2label = {0: "indifferent", 1: "absence", 2:"presence", 3:"former",}


#print(preannotate_taba("pas d'intoxication tabagique chez le patient"))

sentence_segmentation("test.txt", "fr_dep_news_trf", "output.txt")
#print(type(medical_status_annotation("../../../../../msep_packages/medical_status_extractors/taba/CamemBERT_large_based_extractor/model_taba_cmbert", "Pas d'intoxication tabagique chez le patient")))

#donnees/home/dong/improved/taba_models_cross_validation/stratifie_taba_transformer_teste_sur_part_0/checkpoint-1420











    
    
    
