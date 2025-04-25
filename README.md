# MSEP : A pipeline for medical status extraction

This package contains functions used to construct the MSEP pipeline for extracting medical status. Image pipeline_structure.jpg in this depository shows the steps in the pipeline, which you can realise using the modules in "./src/msep"

## Installation
Place you in the directory of your choice

Then install the package from the wheel file (in the folder ./dist)
```
pip install msep-0.0.1-py3-none-any.whl
```
The package and his dependencies are installed.


## Usage

Here is an example of how the pipeline is constructed step by step.

### 1. Preprocessing (P2)

The goal is mainly segmente the text in medical documents into sentences. Use model of your choice to realise this step. Our team also developped a package to realise the text preprocessing based on the quality of our text resources : https://gitlab.com/ltsi-dms/data-science/recherche-developpement/segmentation-de-phrases.git

### 2. Pre-annotation (D1)

MSEP provides a set of rule-based functions for annotating sentences as confirmation/negation of the presence of one of these medical conditions : smoking, diabetes, hypertension, heart failure, COPD and family history of cancer. 
The functions return a number as the label of status: 
- "0": indifference, meaning the sentence does not give information about the medical condition's presence/absence; 
- "1": absence, meaning the sentence confirms the absence of the condition within the patient; 
- "2": presence, meaning the sentence confirms the presence of the condition within the patient; 
- "3": (only for smoking) former, meaning the sentence confirms that the patient used to have the condition, but has stopped now; 

The pre-annotation functions for each medical condition are:
- preannotate_taba (smoking)
- preannotate_diab (diabetes)
- preannotate_hyper (hypertension)
- preannotate_cardia (heart_failure)
- preannotate_copd (COPD)
- preannotate_fam (family history of cancer)

To use a pre-annotation function :
```
from msep.preannotation_rules import preannotate_taba

id2label = {"0": "indifferent", "1": "absence_taba", "2":"presence_taba", "3":"former_taba",}

sentence = "Pas d'intoxication tabagique."

label= preannotate_taba(sentence)

print(id2label[label])
```

Additionally, you can extract status (only absence and presence) of any medical condition by construct your own pre-annotate function, if you have a list of keywords about the targeted medical condition. 

```
from msep.preannotation_rules import preannotate_any

id2label = {"0": "indifferent", "1": "absence_taba", "2":"presence_taba"}

label = preannotate_any(your_sentence, your_list_of_keywords)

print(id2label[label])
```
The keywords should be in lowercase, and without diacritics (é=e, à=a etc.). 
You can also customize your own negation patterns by indicating your negative prefix (such as "pas de ", "absence de ") and suffix lists (" non trouvé", " absent")

```
label = preannotate_any(your_sentence, your_list_of_keywords, list_neg=your_list_of_neg_prefix, list_neg_post=your_list_of_neg_suffix)
```
All prefix end with a space " ", and all suffix begin with a space " ".


### 3.Sentence slection (D2)

First, you need to learn the sample density in your data to decide wether it is necessary to filter out non-sample sentences to increase sample density.

To do that, input pre-annotated sentences (in form of a dictionary that contains a list respectively for sentences, their annotation, and their id defined by you for easy data source trace-back {"texts": [sentence1, sentence2, ... sentenceN], "labels": [label1, label2, ... labelN], "sent_ids":[id1,id2,...idN]}) to the function "show_sample_indiff"

```
from msep.data_selection import show_sample_indiff

show_sample_indiff(pre-annotated_sentences)

```
This will print the percentage of samples (by default it concerns the presence/absence/former status of a medical condition) and non samples (by default it is indifference)
You can also define your own criteria for samples by indicating the label of status that you consider as samples

```
from msep.data_selection import show_sample_indiff

show_sample_indiff(pre-annotated_sentences, samples=["2","3"])

```

If you need to filter out sentences of certain status (indifference for example) to balance the sample density among all status, use the function "sample_sentence_selection", and indicate the type of status you want to select and the ratio for its selection in "category_selection_ratio" argument (ratio < 1 means keep a "ratio" of the sentences correspending to the status and filter out the rest, ratio >1 means multiply the sentences of this status according to the ratio). For example, category_selection_ratio = {"0" : 0.1, "1" : 2} means select 10% of indifferent sentences (labeled with "0"), and double the samples of absence of medical status (labeled with "1")

```
from msep.data_selection import sample_sentence_selection

seletced_sentences=sample_sentence_selection(pre-annotated_sentences, category_selection_ratio)

```

By default, the function selects sentences based on proportion, it can also select by number if you indicate the selecting function as "sample_selection_by_number"

```
from msep.data_selection import sample_sentence_selection

seletced_sentences=sample_sentence_selection(pre-annotated_sentences, category_selection_ratio, selecting_function=sample_selection_by_number)

```
In this cas, in argument "category_selection_ratio" you should indicate the exact number of sentences to select for a medical status instead of proportion.

### 4.Manual annotation (M1)

Format your selected pre-annotated sentences according to the format of your chosen manual annotation tool. 

We used prodigy annotation interface, which takes a jsonl file as input. 

The function prodigy_input_generator can generate a jsonl file using the pre-annotated sentences as input (in forms of {"texts": [sentence1, sentence2, ... sentenceN], "labels": [label1, label2, ... labelN], "sent_ids":[id1,id2,...idN]})

```
from msep.manual_annotation_utils import prodigy_input_generator

prodigy_input_generator(dict_txt_label, path_to_output_file)

```
Using prodigy interface, the annotators assess one sentence at a time, and classify it into corresponding status. To initiate a Prodigy interface for sentence classification, you need to define a recipy that configures the page of annotation. You'll find in folder "src/msep" a file named "prodigy_recipe_example.py", which is a template for a prodigy sentenc classification interface for medical status annotation. Here is an example of the command line that initiate the prodigy interface that calls the recipy:

```
prodigy ID_of_recipe name_of_your_database_for_saving_annotation ./input_file.jsonl -F ./prodigy_recipe_example.py

```
And here is an example of the command line that downloads the annotated sentence database:

```
prodigy db-out name_of_your_database_for_saving_annotation > output_file_path_and_name

```
For more information about the Prodigy annotation interface, please visite https://prodi.gy/

Once manual annotation is finished, the annotation result can be read by the function "read_annotated_result"

```
from msep.manual_annotation_utils import read_annotated_result

annotation=read_annotated_result(path_to_annotated_file)

```
### 5.Automatic conflictual annotation detection (M2)
You can calculate the inter-annotator agreements using "cohen_kappa_annotator_agreement" if you have two annotators who annotated the same sentences

```
from msep.manual_annotation_utils import cohen_kappa_annotator_agreement

agreement_score = cohen_kappa_annotator_agreement(annotation_file1, annotation_file2)

```

it returns the score, and a list containing the differences of the annotation results

### 6.Semi-automatic harmonisation (M4)
If you want to correct the result by adding sentences with key words that shoud have been considered as medical status samples (provide a list of key words "ajout"), or delete sentences with false key words that should have not be considered as samples (provide a list of key words "supprime"), use the function "correction"

```
from msep.manual_annotation_utils import correction

corrected_annotation = correction(annotation_to_be_corrected, ajout, supprime)

```


### 7.Training & validating datasets partition (C1,C2)
To generate the training & validating datasets for cross validation, first, the annotated sentences need to be grouped by status

```
from msep.cross_validation import data_grouping_by_status

data=data_grouping_by_status(corrected_annotation)

```
it takes the annotation result (read from jsonl file) and returns a list containing a list of sentences for each status. In the out list, the medical status' sentence lists are listed in this order : [indifference, absence, presence, former]

Then, we split, for each status, the sentences into n equal sets, n = the number of sessions of your cross validation process

```
from msep.cross_validation import stratified_data_split

all_rec=stratified_data_split(n, data)

```

Use the function "save_training_validating_datasets" to assamble these sets into training & validating datasets for each session of cross validation, and save them into an indicated folder for later use 

```
from msep.cross_validation import save_training_validating_datasets

save_training_validating_datasets(all_rec, saving_location)

```
For the n sets of sentences split by function "stratified_data_split",  "save_training_validating_datasets" select one set each time to save it as the validating set, and combine the rest n-1 sets as the training set. It repeat this operation for n times, and each time the training & validating sets are saved files whose name indicate the session during which they are generated. For example, the first pair of training & validating sets are registered in files "training_set_0.pickle" and "validating_set_0.pickle", and the second pair are in files "training_set_1.pickle" and "validating_set_1.pickle".

You can read the saved datasets like this :

```
with open(saving_location+"training_set_0.pickle", 'rb') as tst:
    dic_train=pickle.load(tst)  
with open(saving_location+"validating_set_0.pickle", 'rb') as tst:
    dic_test=pickle.load(tst)
```

Use function "dataset_formatting_for_finetuning" to format the datasets as compatible input for finetuning a pre-trained language model. Indicate the location of the pre-trained language model to be used for tokenization (this should be the same model you want to fine-tune) in the argument "path_to_pretrained_language_model"

```
from msep.cross_validation import dataset_formatting_for_finetuning

tokenized_data, data_collator = dataset_formatting_for_finetuning(dic_train, dic_test, path_to_pretrained_language_model)

```
this will generate the "tokenized_data" and "data_collator" that are called by the language model Trainer

### 8.Validating (C3)

First, define two variables that will be called when you initiate an instance of the pre-trained model : "id2label" and "label2id", which correspond respectively the label id to label name, and the name to the id. 
For example, for training our smoking extractors, we set the variables as :
```
id2label = {0: "indifferent", 1: "absence", 2:"presence", 3:"former",}
label2id = {"indifferent": 0, "absence": 1, "presence":2, "former":3}  
```

Our customized matrics for validating the extractors are defined in the function "compute_metrics". It takes a couple of lists containing respectively the labels predicted by the extractors, and the correct labels given by annotators. It returns as result the f-score, precision, recall and specificity for each medical status, 
and macro f-score, balalced accuracy for all status.


```
from msep.cross_validation import compute_metrics

id2label = {0: "indifferent", 1: "absence", 2:"presence", 3:"former",}
label2id = {"indifferent": 0, "absence": 1, "presence":2, "former":3}  

validating_score = compute_metrics((automatically_annotated_labels, manually_annotated_labels))
```

You can also weight your loss function if you find the annotated data have unbalanced samples for different status.

First, use "produce_weights" to produce a list of weights 

```
from msep.cross_validation import data_grouping_by_status, stratified_data_split, produce_weights

data=data_grouping_by_status(corrected_annotation)
all_rec=stratified_data_split(n, data)
balanced_weights=produce_weights(all_rec)
```
It gives each status a weight = 1/(its proportion among all status). The order of the weight is the same as the order of status in "all_rec", and should be the same as the order of status in id2label and label2id.

Once the list of weights are generated, you can customize your trainer this way :
```
class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.BCELoss(weight=torch.tensor(balanced_weights))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

Once you've generated the "tokenized_data", "data_collator", defined "compute_metrics", you can call a language model (that can perform sequence classification task) and configure a trainer for fine-tuning it (define a " model_saving_location" to save the trained model):

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    path_to_pretrained_model, num_labels=4, id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(path_to_pretrained_model)


training_args = TrainingArguments(
    output_dir = model_saving_location,
    learning_rate=2e-5,#0.1,#2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=15, 
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
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

trainer.train()
```


The trainer will train and validate the models, and print the metric scores.
Collect scores from all sessions of cross validation of the same metric, then you can use function "mean_std" to calculate the average score and standard deviation for this metric.

```
from msep.cross_validation import mean_std

average, standard_deviation = mean_std(scores)

```

To use the fine-tuned extractor for annotating a sentence :

```
from msep.CamemBERT_medical_status_extractors import medical_status_annotation

label = medical_status_annotation(path_to_the_saved_model, your_sentence)

```
this returns a single label id corresponding to the status ("0","1","2" or"3", representing "indifference", "absence", "presence", "former")

To use the fine-tuned extractor for annotating a list of sentences :

```
from CamemBERT_medical_status_extractors import medical_status_annotation

labels = medical_status_annotation_iter(path_to_the_saved_model, list_of_sentences)

```


### 9.Model assessement (C4)

As comparison, we've also designed LLM prompts for medical status extraction:
- prompt_template_taba, for extracting smoking status
- prompt_template_diab, for extracting diabetes status
- prompt_template_hyper, for extracting hypertension status
- prompt_template_cardia, for extracting heart failure status
- prompt_template_copd, for extracting COPD status
- prompt_template_fam, for extracting family history of cancer status

To use one of them to annotate a list of sentences :
```
from msep.concurrent_LLM_extractors import llm_annotate

sentence_labels=llm_annotate(list_of_sentences, prompt_template_of_your_choice, path_to_llm)

```

You can also configure the temperature, repetition_penalty, return_full_text and max_new_tokens of the LLM pipeline:

```
from msep.concurrent_LLM_extractors import llm_annotate

sentence_labels=llm_annotate(list_of_sentences, prompt_template_of_your_choice, path_to_llm, temperature=0.2, repetition_penalty=1.1, return_full_text=True, max_new_tokens=1000)

```

The function returns a list containing lists of a sentence and its label, e.g. [[sentence1, label1], [sentence2, label2]...[sentenceN, labelN]]














