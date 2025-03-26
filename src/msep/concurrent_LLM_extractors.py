# Import libraries 
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = "cpu"







#LLM prompt for extracting smoking status
prompt_template_taba = """
### Instruction: Act as a LARGE Language Models Expert. I will give you an SENTENCE from a medical report in French, 
and you will answer with and only with one of these 4 labels that indicates the smoking status of the patient implied by the sentence : 
"0", "1", "2", "3". 
Here, "3" means that the SENTENCE implies that the patient used to smoke, but has stopped;
"2" means that the SENTENCE implies that the patient is an active smoker, or that the patient smokes and has no sign of stopping;
"1" means that the SENTENCE implies that the patient doesn't smoke, or has no trace of tobacco poisoning;
"0" means that the SENTENCE donesn't imply any of those cases above.
If the smoking status implied by the SENTENCE doesn't concern the patient (but rather concerns his/her parents for example), 
then you should give "0" as answer regardless of the staus; 
By default, the SENTENCE describes information about a patient, unless it indicates clearly the subject or the object is about other people, 
such as his/her parents, in which case the sentence should receive the label "0"; 
By default, the French word "sevrage" (or "sévérage") indicates a stop of smoking, which means the SENTENCE should receive the label "3", 
unless the SENTENCE express clearly that the "sevrage" is about something else other than smoking, such as a cessation (sevrage) of alcohol, 
in which case the SENTENCE should receive the label 0; Variations of words (abbreviation, words deprived of French diacritics, 
spelling mistakes etc) should also be taken into account while labeling the sentence, specially those frequently used by doctors, for example, 
"severage" equals "sévérage", "taba" equals "tabagisme".



### SENTENCE:
{sentence}

### Answer:
 """

#LLM prompt for extracting diabetes status
prompt_template_diab = """
### Instruction: Act as a LARGE Language Models Expert. I will give you an SENTENCE from a medical report in French, 
and you will answer with and only with one of these 3 labels that indicates the diabetes status of the patient implied by the sentence : "0", "1", "2". 
Here, "2" means that the SENTENCE implies that the patient currently has diabetes or is under treatment of diabetes, regardless of the type of diabetes;
"1" means that the SENTENCE implies that the patient doesn't have diabetes currently, regarless of the type of diabetes;
"0" means that the SENTENCE donesn't imply any of those cases above.
Keep in mind : if the diabetes status implied by the SENTENCE doesn't concern the patient (but rather concerns his/her parents for example), 
then you should give "0" as answer regardless of the staus; By default, the SENTENCE describes information about a patient, unless it
indicates clearly the subject or the object is about other people, such as his/her parents, in which case the sentence should receive the label "0"; 
Please take the variations of words (abbreviation, words deprived of French diacritics, spelling mistakes etc) into account while labeling the sentence, specially those frequently used by doctors, for example, "diabetes" equals "diabètes", "dt2" equals "diabètes type 2", etc..
Finally, if not otherwise explicited, we consider the mention of diabetes in the text as a confirmation of its presence, and you should response with "2" in that cas. 


### SENTENCE:
{sentence}

### Answer:
 """


#hypertension
prompt_template_hyper = """
### Instruction: Act as a LARGE Language Models Expert. I will give you an SENTENCE from a medical report in French, 
and you will answer with and only with one of these 3 labels that indicates the blood pressure status of the patient implied by the sentence : "0", "1", "2". 
Here, "2" means that the SENTENCE implies that the patient has high blood pressure;
"1" means that the SENTENCE implies that the patient doesn't have high blood pressure;
"0" means that the SENTENCE donesn't imply any of those cases above.
Keep in mind : if the blood pressure status implied by the SENTENCE doesn't concern the patient (but rather concerns his/her parents for example), 
then you should give "0" as answer regardless of the staus; By default, the SENTENCE describes information about a patient, unless it
indicates clearly the subject or the object is about other people, such as his/her parents, in which case the sentence should receive the label "0"; 
Please take the variations of words (abbreviation, words deprived of French diacritics, spelling mistakes etc) into account while labeling the sentence, specially those frequently used by doctors, for example, "HTA" equals "hypertension artérielle", which means high blood pressure.
Finally, if not otherwise explicited, we consider the mention of high pressure (in French "hypertension") in the text as a confirmation of the presence of high blood pressure, and you should response with "2" in that cas. 


### SENTENCE:
{sentence}

### Answer:
 """


#heart failure
prompt_template_cardia = """
### Instruction: Act as a LARGE Language Models Expert. I will give you an SENTENCE from a medical report in French, 
and you will answer with and only with one of these 3 labels that indicates the heart failure status of the patient implied by the sentence : "0", "1", "2". 
Here, "2" means that the SENTENCE implies that the patient has heart failure condition ("insuffisance cardiaque" in french);
"1" means that the SENTENCE implies that the patient doesn't have heart failure;
"0" means that the SENTENCE donesn't imply any of those cases above.
Keep in mind : if the heart failure status implied by the SENTENCE doesn't concern the patient (but rather concerns his/her parents for example), 
then you should give "0" as answer regardless of the staus; By default, the SENTENCE describes information about a patient, unless it
indicates clearly the subject or the object is about other people, such as his/her parents, in which case the sentence should receive the label "0"; 
Please take the variations of words (abbreviation, words deprived of French diacritics, spelling mistakes etc) into account while labeling the sentence, specially those frequently used by doctors, for example, "IC" equals "insuffisance cardique", which means heart failure.
All types of heart failure are taken into account, that means expressions such as "ICD", abbreviation of "insuffisance cardique droite", is an indicator of heart failure.
But do not consider causes or consequences of heart failure as the confirmation of its presence, that means the expression such as "insuffisance mitrale", a common cause for heart failure, is not an indicator of the presence of heart failure.
Finally, if not otherwise explicited, we consider the mention of heart failure (in French "insuffisance cardiaque" or "décompensation cardiaque" or their abbreviations and variants) in the text as a confirmation of the presence of heart failure, and you should response with "2" in that cas. 


### SENTENCE:
{sentence}

### Answer:
 """



#COPD
prompt_template_copd = """
### Instruction: Act as a LARGE Language Models Expert. I will give you an SENTENCE from a medical report in French, 
and you will answer with and only with one of these 3 labels that indicates the chronic obstructive pulmonary disease (COPD, or BPCO in French) status of the patient implied by the sentence : "0", "1", "2". 
Here, "2" means that the SENTENCE implies that the patient has COPD ( la bronchopneumopathie chronique obstructive (BPCO) in french);
"1" means that the SENTENCE implies that the patient doesn't have COPD (BPCO);
"0" means that the SENTENCE donesn't imply any of those cases above.
Keep in mind : if the COPD (BPCO) status implied by the SENTENCE doesn't concern the patient (but rather concerns his/her parents for example), 
then you should give "0" as answer regardless of the COPD (BPCO) staus; By default, the SENTENCE describes information about a patient, unless it
indicates clearly the subject or the object is about other people, such as his/her parents, in which case the sentence should receive the label "0"; 
Please take the variations of words (abbreviation, words deprived of French diacritics, spelling mistakes etc) into consideration while labeling the sentence, specially those frequently used by doctors, for example, "BPCO" equals "la bronchopneumopathie chronique obstructive", "bcpo" equals "pbco" equals "BPCO".
All types of COPD are taken into account, but do not consider causes or consequences of COPD as the confirmation of its presence.
Finally, if not otherwise explicited, we consider the mention of COPD (in French "BPCO" or "bronchopneumopathie chronique obstructive" or "bronchite chronique" or their abbreviations and variants) in the text as a confirmation of the presence of heart failure, and you should response with "2" in that cas. 


### SENTENCE:
{sentence}

### Answer:
 """



#family history of cancer
prompt_template_fam = """
### Instruction: Act as a LARGE Language Models Expert. I will give you an SENTENCE from a medical report in French, 
and you will answer with and only with one of these 3 labels that indicates the family history of cancer of the patient implied by the sentence : "0", "1", "2". 
Here, "2" means that the SENTENCE implies that the patient has family history of cancer;
"1" means that the SENTENCE implies that the patient doesn't have family history of cancer;
"0" means that the SENTENCE donesn't imply any of those cases above.
Keep in mind : if the cancer implied by the SENTENCE doesn't concern the patient's family member (but rather concerns him/herself), 
then you should give "0" as answer; By default, the SENTENCE describes information about a patient, unless it
indicates clearly the subject or the object is about his/her family member;
All family members related by blood (father, mother or cousin etc) and all types of cancer counts.
Please take the variations of words (abbreviation, words deprived of French diacritics, spelling mistakes etc) into consideration while labeling the sentence, specially those frequently used by doctors, for example, "fam" equals "family", "atcd fam K" equals "antécédent familial de cancer" equals "family history of cancer".
Finally, if not otherwise explicited, we consider the mention of family history of cancer (in french "antécédent familial de cancer") in the text as a confirmation of its presence, and you should response with "2" in that cas. 


### SENTENCE:
{sentence}

### Answer:
 """




def llm_annotate(ls, prompt_template_of_your_choice, path_to_llm, temperature=0.2, repetition_penalty=1.1, return_full_text=True, max_new_tokens=1000):
    """
    using a chosen LLM prompt to annotate a list of sentences with pre-defined medical status
    
    e.g. llm_annotate(ls, prompt_template_taba, "./Mixtral-8x7B-Instruct-v0.1") to call Mixtral to annotate the smoking status in texts
    
    """
    
    
    model = AutoModelForCausalLM.from_pretrained(path_to_llm, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(path_to_llm)


    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        return_full_text=repetition_penalty,
        max_new_tokens=max_new_tokens,

    )
    
    
    
    
    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["sentence"],
        template=prompt_template_of_your_choice,
    )
    # Create llm chain 
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)



    for s in ls:

        input_text = {
            "sentence": s,
        }

        # Generate and print LLM response 

        response = llm_chain(input_text)



        lb=response['text'].split("### Réponse:")[-1].strip("-").strip()

        lres.append([s,lb])

    return lres




