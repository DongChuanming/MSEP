import random
import pickle
import torch
from torch import nn
import evaluate
import datasets
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from transformers import Trainer
from transformers import AutoModel, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback, IntervalStrategy

def data_grouping_by_status(lr):
    """
    lr = output file from prodigy containing annotated sentences 
    """
    pr=list()
    ab=list()
    an=list()
    un=list()
    for d in lr: #all_taba_complet:
        if not d["accept"]:
            un.append(d["text"])
        if "1" in d["accept"]:
            ab.append(d["text"])
        if "2" in d["accept"]:
            pr.append(d["text"])
        if "3" in d["accept"]:
            an.append(d["text"])

    return [un, ab, pr, an]

data=data_grouping_by_status(lr)




def stratified_data_split(n, data):
    """
    split the data into n parts with equal size
    
    """
    all_rec=list()
    for i in range(len(data)):
        rec=[[] for e in range(n)]
        dt=data[i]
        random.shuffle(dt)
        #print(len(dt))
        sp=round(len(dt)/n)
        if len(dt)/n<1:
            print("not enough samples for cross validation on status "+str(i)+", therefore it is excluded from the process")
            all_rec.append(rec)
            continue
        for e in range(n):
            if (e+1)*sp < len(dt):
                rec[e]=dt[e*sp:(e+1)*sp]
            else:
                rec[e]=dt[e*sp:]
        all_rec.append(rec)
    return all_rec



def produce_weights(all_rec):
    
    total=sum([len(a[0]) for a in all_rec])
    
    return [total/a[0] for a in all_rec]


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
    


res=stratified_data_split(n, data)
    
def save_training_validating_datasets(all_rec, saving_location):
    """
    create and save training and validating datasets
    
    """

    dic_train={"text":[], "label":[]}
    dic_test={"text":[], "label":[]}
    for i in range(len(res[0])):
        for e in range(len(res)):
            for i2 in range(len(res[0])):
                if i2!=i:
                    dic_train["text"]+=res[e][i2]
                    dic_train["label"]+=[e]*len(res[e][i2])
                else:
                    dic_test["text"]+=res[e][i2]
                    dic_test["label"]+=[e]*len(res[e][i2])
                    
        with open(saving_location+"training_set_"+str(i)+".pickle", 'wb') as tst:
            pickle.dump(dic_train, tst)
        with open(saving_location+"validating_set_"+str(i)+".pickle", 'wb') as tst:
            pickle.dump(dic_test, tst)
    


with open(saving_location+"training_set_0.pickle", 'rb') as tst:
    dic_train=pickle.load(tst)  
with open(saving_location+"validating_set_0.pickle", 'rb') as tst:
    dic_test=pickle.load(tst)


#tokenizer = AutoTokenizer.from_pretrained(path_to_pretrained_model)
    
def preprocess_function(examples):
    """
    customize tokenizer
    
    """
    return tokenizer(examples["text"], max_length=500, truncation=True)


def dataset_formatting_for_finetuning(dic_train, dic_test, path_to_pretrained_model):
    """
    prepare your data for fine-tuning a transformer based pre-trained language model
    
    """
    
    #from datasets import Dataset
    train_patient = Dataset.from_dict(dic_train)
    test_patient = Dataset.from_dict(dic_test)
    
    dict_patient=datasets.DatasetDict({"train":train_patient,"test":test_patient})
    
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path_to_pretrained_model)
    
    tokenized_patient = dict_patient.map(preprocess_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return tokenized_patient, data_collator            
            

tokenized_patient, data_collator =  dataset_formatting_for_finetuning(dic_train, dic_test, path_to_pretrained_model)        

            
            
            
def f1(l1,l2,a):
    """
    calculate the f-score of a single medical status (a) comparing the labels given by model (l2) with labels annotated manually (l1)
    return a list containing [f-score, precision, recall]
    
    """
    vp=0
    fp=0
    fn=0
    for i in range(len(l1)):
        if l1[i]==a:
            if l2[i]==a:
                vp+=1
            else:
                fn+=1
        else:
            if l2[i]==a:
                fp+=1
    if (vp+fp)!=0:
        prec=vp/(vp+fp)
    else:
        prec=0
    if (vp+fn)!=0:
        rap=vp/(vp+fn)
    else:
        rap=0
    if (prec+rap)!=0:
        f1=2*(prec*rap)/(prec+rap)
    else:
        f1=0
    return [f1, prec, rap]
    #for e in l2:
    #    if e==a:

    
id2label = {0: "indifferent", 1: "absence", 2:"presence", 3:"former",}
label2id = {"indifferent": 0, "absence": 1, "presence":2, "former":3}  

def compute_metrics(eval_pred):
    """
    evaluate the output of trained extractor with f-score, precision, recall and specificity for each medical status, 
    and macro f-score, balalced accuracy for all status 
    
    """
    
    predictions, labels = eval_pred
    #with open("output_pred.txt", "w", encoding="utf-8") as o:
    #    o.write(str(predictions))
    #    o.write(str(labels))
    predictions = np.argmax(predictions, axis=1)
    
    concat=0
    
    dico_mesure=dict()
    
    for k in id2label:
        fk=f1(labels, predictions, k)
        lb=id2label[k]
        dico_mesure[lb+"_f1"]=fk[0]
        dico_mesure[lb+"_prec"]=fk[1]
        dico_mesure[lb+"_rap"]=fk[2]
        temp_label=[]
        temp_predict=[]
        for x in labels:
            if x!=k:
                temp_label.append("t")
            else:
                temp_label.append(str(k))
        for x in predictions:
            if x!=k:
                temp_predict.append("t")
            else:
                temp_predict.append(str(k))
        
        dico_mesure[lb+"_specificity"]=recall_score(temp_label, temp_predict, pos_label="t")
        
        concat+=fk[0]
    macro=concat/len(id2label.keys())
    #macro=concat/3
    dico_mesure["macro_f1"]=macro
    dico_mesure["balanced_accuracy"]=balanced_accuracy_score(labels, predictions)
    
    return dico_mesure #{"macro_f1":macro}
        



#id2label = {0: "indifferent", 1: "absence", 2:"presence" }
#label2id = {"indifferent": 0, "absence": 1, "presence":2}  

#id2label = {0: "indifferent", 2:"presence", }
#label2id = {"indifferent": 0, "presence":2, }  

## begin fine-tuning
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
#from transformers import RobertaForSequenceClassification, TrainingArguments, Trainer
#model = CamembertForSequenceClassification.from_pretrained(
#    "/donnees/share/models/camembert-large",num_labels=4, id2label=id2label, label2id=label2id
#)
model = AutoModelForSequenceClassification.from_pretrained(
    path_to_pretrained_model, num_labels=4, id2label=id2label, label2id=label2id
)


training_args = TrainingArguments(
    output_dir=model_saving_location,
    learning_rate=2e-5,#0.1,#2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=15, #20
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model = 'macro_f1',
    load_best_model_at_end=True,
    push_to_hub=False,
    save_total_limit = 1,
)

training_args = training_args.set_dataloader(num_workers=2, prefetch_factor=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_patient["train"],
    eval_dataset=tokenized_patient["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

trainer.train()





def mean_std(scores):
    """
    calculate the average and standard deviation of a type of score for a medical status
    
    e.g., for absence taba, the arg "scores" should contain the f-score of each extractor fine-tuned during the sessions of cross validation 
    for detecting this medical status 
    
    return a tuple of the average and stantard deviation
    """
    smean=np.mean(scores)
    sstd=np.std(scores)
    return (smean, sstd)










    
    
    
    
    