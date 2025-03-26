import json
import re
import unidecode


options = [{'id': '1', 'text': 'absence'},
 {'id': '2', 'text': 'presence'},
 {'id': '3', 'text': 'former'}]

label_ids={"indifference":"0", "absence": "1", "presence":"2", "former":"3"}

# generate the jsonl files used by prodigy to initiate annotation interface
def prodigy_input_generator(dict_txt_label, options=options, output_file):
    """
    Genereate input jasonl file for Prodigy manual annotation from a list of sentences and a list of their pre-annotated labels compiled in a dictionary 
    
    Args:
    
        dict_txt_label: the dictionary containing the sentences and their labels in this format : 
        {"texts": [sent1, sent2, ... sentN], "labels": [label1, label2, ... labelN], "sent_ids":[id1,id2,...idN]};
        
        options : the label choices (id and name) that you want to shouw on the Prodigy annotation interface 
        
        
        output_file: path to the output jsonl file e.g. "./prodigy_preannotation_smoking.jsonl" 
    """

    texts=dict_txt_label["text"]
    labels=dict_txt_label["label"]
    sent_ids=dict_txt_label["sent_ids"]
    assert len(texts)==len(labels), "number of sentences and labels doesn't match"
    assert len(texts)==len(sentence_ids), "number of sentences and their ids doesn't match"
    
    with open(output_file, "w", encoding="utf-8") as w:

        for i in range(len(texts)):
            
            line_count=str(i)
            sent_id=sent_ids[i]

            txt=texts[i]
            val=str(labels[i])

            dic={"text":"", "meta":"", "options": options, "_input_hash":"", "_task_hash":"", "_view_id":"", "accept":[val], "answer":"", "_timestamp":""}
            dic["text"]=txt
            dic["meta"]={"ID_sent": sent_id, "annotator": 1, "annotator2": 2}
            dic["_input_hash"]=-line_count
            dic["_task_hash"]=-line_count
            dic["_view_id"]="choice"
            dic["answer"]="accept"
            dic["_timestamp"]=line_count
            w.write(json.dumps(dic, ensure_ascii=False).strip()+"\n")
            
            
# after the sentence are annotated, a jsonl is generated with prodigy
def read_annotated_result(path_to_file):
    """
    read a jsonl file from path_to_file and return the content in a list of dictionaries lr1
    """
    lr=list()
    with open(path_to_file, "r", encoding="utf-8") as t1:
        listdic = t1.readlines()
    for i in listdic:
        lr.append(json.loads(i))
    return lr
        

def cohen_kappa_annotator_agreement(file1, file2):
    
    """
    file1: path to the file of the sentences annotated by the 1st annotator
    file2: path to the file of the sentences annotated by the 2nd annotator
    """
    
    lr1=read_annotated_result(file1)
    lr2=read_annotated_result(file2)
    
    lb1=[]
    lb2=[]
    
    difference=list()

    for d1 in lr1:
        for d2 in lr2:
            if d1["text"]==d2["text"]:
                if d1["ID_sent"]==d2["ID_sent"]:
                    if d1["text"].strip() not in rec_phrase:
                        
                        if d1["accept"]:
                            lb1.append(d1["accept"][0])
                        else:
                            lb1.append("0")

                        if d2["accept"]:
                            lb2.append(d2["accept"][0])
                        else:
                            lb2.append("0")
                            
                        if d1['accept']!=d2["accept"]:
                            
                            difference.append((d1["text"], d1["accept"], d2["accept"]))
                            
    
    
    agreemet=cohen_kappa_score(lb1, lb2)
    
    return agreement, difference
                        

    
list_neg=["pas ", "pas de ", "pas d'","absence ", "absence de ", "absence d'", 
          "non ", "sans ", "sans signe de ", "sans signe d'","ni de ", "ni ", "aucun ", 
          "aucun signes d'", "aucun signes de ", "pas de signe d'","pas de signe de ", "pas de signes de ", 
          "pas de signes d'", "pas d'autre signe d'", "pas d'autre signe de ", "pas de signes en faveur d'une "]

list_neg_post=[" absent", " absente", " non trouve", " non trouvee", 
               " non detecte",  " non detectee", " traite",  " traitee",
               " deja traite", " deja traitee"," pas trouve", " pas trouvee"," pas detecte"," pas detectee", " arrete", " arretee"]
    
# correct the annotation result (lr) by including more instances (ajout) or deleting certain instances (supprime) for a certain medical status

def correction(lr, list_neg=list_neg, list_neg_post=list_neg_post, label_ids=label_ids, ajout, supprime):
    """
    a function that automatically correct the manual annotation result (jsonl file read into a list of dictionaries "lr" using read_annotated_result function). 
    
    It does two things : unannotate sentences that were wrongfully considered as related to the medical condition 
    (use the list "supprime" which contains expressions that were wrongfully considered as an indicator of the medical condition) 
    
    and annotate the sentences that were wrongfully missed during the manual annotation
    (use the list "ajout" which contains the expressions that were wrongfully considered as not an indicator of the medical condition)
    into "presence" of the medical condition, and "absence" of the medical condition 
    (using list_neg and list_neg_post to add negation prefix and suffix to the key expressions in ajout to form a list of negation for the medical condition)
    
    return a corrected version of "lr"
    
    """
    corrected=list()
    a_decider=list()
    
    
    for d1 in lr:
        nsent=" "+unidecode.unidecode(d1["text"].strip().lower())+" "
        if d1["accept"]:
            #c1=0
            for e1 in supprime :
                pat1.compile("\W+"+e1+"s?\W+")
                if re.search(pat1,nsent):
                    #c1+=1
                    
                    c2=0
                    
                    for e2 in ajout:
                        pat2=re.compile("\W+"+e2+"s?\W+")
                        if re.search(pat2,nsent):
                            #corrected.append(d1)
                            c2+=1
                            break
                    if not c2:
                        d1["accept"]=[]
                        #corrected.append(d1)
                    #else:
                        #corrected.append(d1)
                    #break
            
            #if not c1:
            corrected.append(d1)
        
        else:
            #c3=0
            for e3 in ajout:
                pat3=re.compile("\W+"+e3+"s?\W+")
                if re.search(pat3,nsent):
                    #c3+=1
                    patterns_neg = []
                    for b in list_neg:
                        patterns_neg.append(b+e3)
                    for c in list_neg_post:
                        patterns_neg.append(e3+c)
                    for e4 in patterns_neg:
                        pat4=re.compile("\W+"+e4+"s?\W+")
                        if re.search(pat4,nsent):
                            d1["accept"]=[label_ids["absence"]]
                            break
                        d1["accept"]=[label_ids["presence"]]

                        
            
            #if not c3:
            corrected.append(d1)
    
    return corrected







