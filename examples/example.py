from preannotation_rules import preannotate_taba
from CamemBERT_medical_status_extractors import medical_status_annotation
from preprocessing import sentence_segmentation

id2label = {0: "indifferent", 1: "absence", 2:"presence", 3:"former",}


#print(preannotate_taba("pas d'intoxication tabagique chez le patient"))

sentence_segmentation("test.txt", "fr_dep_news_trf", "output.txt")
#print(type(medical_status_annotation("../../../../../msep_packages/medical_status_extractors/taba/CamemBERT_large_based_extractor/model_taba_cmbert", "Pas d'intoxication tabagique chez le patient")))

#donnees/home/dong/improved/taba_models_cross_validation/stratifie_taba_transformer_teste_sur_part_0/checkpoint-1420


from data_selection import show_sample_indiff, sample_sentence_selection, show_status_prop, sample_selection_by_number 

pre_annotated_sentences = { 

    "texts": [ 

        "The patient has chronic kidney disease.", 
        
        "The patient has chronic kidney disease.", 

        "No signs of chronic kidney disease detected.", 

        "CKD has been present since 2018.", 

        "No kidney problem reported.", 

        "Normal kidney function.", 

        "Confirmed case of chronic kidney disease.", 

        "Patient has no history of CKD.", 

        "CKD not found at this time.", 

        "No renal abnormalities detected.", 

        "Severe chronic kidney disease diagnosed." 

    ], 

    "labels": [ 

        "2",  # presence 
        
        "2",  # presence 

        "1",  # absence 

        "2", 

        "1", 

        "0",  # indifferent 

        "2", 

        "1", 

        "1", 

        "0", 

        "2" 

    ], 

    "sent_ids": list(range(1, 12)) 

} 

 

# Display the proportion of sample vs. indifferent sentences 

show_sample_indiff(pre_annotated_sentences) 

show_status_prop(pre_annotated_sentences, duplication=False)


new_list=sample_sentence_selection(pre_annotated_sentences, {"0" : 8, "1" : 4, "2":4}, selecting_function=sample_selection_by_number)
print(new_list)
    
    
