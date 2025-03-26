import os
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer

#id2label = {0: "indifferent", 1: "absence", 2:"presence", 3:"former",} #this set of label and ids was used during model training, do not change it or the label cannot be translated from the ID returned by the model with expected value

def medical_status_annotation(path_to_model, sent): # this fonction calls the pre-trained model and use it to annotate a sentence
    tokenizer = AutoTokenizer.from_pretrained(path_to_model) # calls the model's tokenizer
    model_taba=AutoModelForSequenceClassification.from_pretrained(path_to_model) # calls the model for sentence annotation task (the only task it was trained for, do not attempt other tasks)
    inputs = tokenizer(sent, max_length=500, truncation=True,return_tensors="pt") # tokenize the sentence 
    with torch.no_grad():
        staba = model_taba(**inputs).logits.argmax().item() # the model annotates the sentence, and we save the label ID to staba
    return staba # this returns the ID of the labels (0,1,2 or3), not the label itself


#label=medical_status_annotation("./model_taba_cmbert", "Pas d'intoxication tabagisme.") # apply the donction that uses the model to annotate a sentence

#print(id2label[label])


def medical_status_annotation_iter(path_to_model, list_of_sentences): # this fonction calls the pre-trained model and use it to annotate a list of sentences
    tokenizer = AutoTokenizer.from_pretrained(path_to_model) # calls the model's tokenizer
    model_taba=AutoModelForSequenceClassification.from_pretrained(path_to_model) # calls the model for sentence annotation task (the only task it was trained for, do not attempt other tasks)
    labels=list() # list to save the annotations
    for sent in list_of_sentences:
        inputs = tokenizer(sent, max_length=500, truncation=True,return_tensors="pt") # tokenize the sentence 
        with torch.no_grad():
            staba = model_taba(**inputs).logits.argmax().item() # the model annotates the sentence, and we save the label ID to staba
        labels.append(staba) # this returns the ID of the labels (0,1,2 or3), not the label itself
    return labels

# apply the donction that uses the model to annotate a list of sentences
#labels=medical_status_annotation_iter("./model_taba_cmbert", ["Pas d'intoxication tabagisme.", "cigarettes : 3 paquets/jour", "sevrage tabagisme depuis 2 ans", "diabetes type 2 diagnostiquées."]) 
#print([id2label[e] for e in labels])