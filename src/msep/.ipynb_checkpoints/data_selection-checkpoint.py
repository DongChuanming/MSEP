import random



def sample_selection_by_proportion(s, ratio):
    """
    select a proportion of content from a list of sentences s
    
    ratio = proportion of sentences to be extracted, e.g. ratio=0.5 means extract a half of the sentences; ratio=2 means duplicating the sentences by 2
    
    sentences are ordered by their length, when iterating the list, for every n sentences one is selected
    
    return a list of selected sentence
    
    
    """
    n=1/ratio
    ns=list()
    #ls=sorted(s, key=lambda a:len(a), reverse=True)
    random.shuffle(s)
    total=len(s)
    if n<total:
        if n>=1:
            times=round(total/n)
            for i in range(times):
                try:
                    ns.append(s[i*n])
                except:
                    return ns
        else:
                
            return s*round(ratio)
        
    else:
        return [s[round(total/2)]]
    return ns


def sample_selection_by_number(s, n):
    """
    select n samples from a list of sentences s
    
    sentences are ordered by their length, when iterating the list, for every s/n sentences one is selected
    
    return a list of selected sentence
    
    """
    ns=list()
    #ls=sorted(s, key=lambda a:len(a), reverse=True)
    random.shuffle(s)
    total=len(s)
    if n<total:
        if n>1:
            times=round(total/n)
            for i in range(n):
                try:
                    ns.append(s[i*times])
                except:
                    return ns
        else:
            return [s[round(total/2)]]
    else:
        return s*round(n/total)
    return ns



def show_sample_indiff(dict_txt_label, samples=["1","2","3"]):
    """
    calculate the proportion of samples and non-samples of medical status in the list of annotated sentences dict_text_label:
    {"texts": [sent1, sent2, ... sentN], "labels": [label1, label2, ... labelN]};
    
    arguments:
    
    dict_txt_label: the dictionary containing the sentences and their labels in at least this format : 
    {"texts": [sent1, sent2, ... sentN], "labels": [label1, label2, ... labelN]};
    
    samples: the list of labels that represent medical status samples. By default, "0" marks the non-sample sentences. you can define your own list of labels of 
    samples defined in your project
    
    """
    
    texts=dict_txt_label["texts"]
    labels=dict_txt_label["labels"]
    
    assert len(texts)==len(labels), "number of sentences and labels doesn't match"
    
    samples=dict()
    indiff=dict()
    
    for i in range(len(texts)):
        if labels[i] not in samples:
            indiff["texts"]=indiff.get("texts", [])+[texts[i]]
            indiff["labels"]=indiff.get("labels", [])+[labels[i]]
            
        else:
            samples["texts"]=samples.get("texts", [])+[texts[i]]
            samples["labels"]=samples.get("labels", [])+[labels[i]]
        
    sample_ratio=round(len(samples["texts"])/len(texts),2)*100
    indiff_ratio=round(len(indiff["texts"])/len(texts),2)*100
    
    print("The "+str(len(texts))+" sentences have "+str(len(samples["texts"]))+" samples, i.e. "+str(sample_ratio)+"%")
    print("The "+str(len(texts))+" sentences have "+str(len(indiff["texts"]))+" non-samples, i.e. "+str(indiff_ratio)+"%")
    
    sample_indiff_pair=(samples, indiff)
    
    return sample_indiff_pair
            





def sample_sentence_selection(dict_txt_label, category_selection_ratio, selecting_function=sample_selection_by_proportion):
    
    """
    reduce certain number/ratio of certain category from annotated sentences
    
    arguments:
    
    dict_txt_label: the dictionary containing the sentences, their medical status labels and their ids in at least this format : 
    {"texts": [sent1, sent2, ... sentN], "labels": [label1, label2, ... labelN], "sent_ids":[id1,id2,...id3]};
    sentence_ids : list of ID for each sentence defined by yourself (it is recommanded to include document id in the sentence id allow data source trace-back. 
        For example, if the sentence is the 3rd sentence selected from the document "EHOP12345" to be annotated, then the sentence id sould be "EHOP12345_3"). 
        
    
    category_selection_ratio: dictionary that indicate the label of status and the proportion/number you want to select in format {label : proportion/number}, 
    e.g. {"0" : 0.1, "1" : 2} means select 10% of indifferent sentences (labeled with "0"), and double the samples of absence of medical status (labeled with "1") 
    
    selecting_function: selection function of your choice. Our package propose 2 function, for selecting by proportion use sample_selection_proportion, 
    for selecting by number use sample_selection. By default, sample_selection_proportion is applied here
    
    """
    
    
    texts=dict_txt_label["texts"]
    labels=dict_txt_label["labels"]
    sent_ids=dict_txt_label["sent_ids"]
    
    assert len(texts)==len(labels), "number of sentences and labels doesn't match"
    assert len(texts)==len(sentence_ids), "number of sentences and ids doesn't match"
    
    selected={"texts":[], "labels":[], "sent_ids":[]}
    rec=dict()
    
    for i in range(len(texts)):
        lb=labels[i]
        txt=texts[i]
        sid=sent_ids[i]
        if lb in category_selection_ratio:
            rec[lb]=rec.get(lb,[])+[(txt,sid)]
        else:
            selected["texts"].append(txt)
            selected["labels"].append(lb)
            selected["sent_ids"].append(sid)
    
    for lb, ltxt in rec.items():
        rate=category_selection_ratio[lb]
        selected_sentences=selecting_function(ltxt, rate)
        selected["text"]+=[e[0] for e in selected_sentences]
        selected["label"]+=[lb]*len(selected_sentences)
        selected["sent_ids"]+=[e[1] for e in selected_sentences]
    
    return selected
        
        
    
    