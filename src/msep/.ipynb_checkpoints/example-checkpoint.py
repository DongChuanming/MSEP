from preannotation_rules import preannotate_taba
from CamemBERT_medical_status_extractors import medical_status_annotation

id2label = {0: "indifferent", 1: "absence", 2:"presence", 3:"former",}


print(type(preannotate_taba("pas d'intoxication tabagique chez le patient")))


print(type(medical_status_annotation("../../../../../msep_packages/medical_status_extractors/taba/CamemBERT_large_based_extractor/model_taba_cmbert", "Pas d'intoxication tabagique chez le patient")))

#donnees/home/dong/improved/taba_models_cross_validation/stratifie_taba_transformer_teste_sur_part_0/checkpoint-1420




"""prodigy ID_of_recipe name_of_your_database_for_annotation ./input_file.jsonl -F ./prodigy_recipe_example.py"""




balanced_weights=produce_weights(all_rec)


class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)

        logits = outputs.get('logits')
        # compute custom loss
        #loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([weight1,weight2,weight3]))#,weight4]))
        loss_fct = nn.BCELoss(weight=torch.tensor(balanced_weights))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    
    
    
