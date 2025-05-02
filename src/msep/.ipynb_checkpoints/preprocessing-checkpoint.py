import spacy
import os
import unicodedata
import unidecode

def uniform_sentence(sent):
    return unidecode.unidecode(sent.strip().lower())

def sentence_segmentation(path, segmenter_model, output=False, uniform=False):
    nlp = spacy.load(segmenter_model)
    sentences=[]
    if os.path.isfile(path):
        txt=open(path, "r", encoding="utf-8").read()
        doc=nlp(txt)
        if uniform==True:
            sentences=[uniform_sentence(e.text) for e in doc.sents if e ]
        else:  
            sentences=[e.text.strip() for e in doc.sents if e ]
        if output:
            with open(output, "w", encoding="utf-8") as w:
                w.write("\n".join(sentences))
        
    elif os.path.isdir(path):
        for file in os.listdir(path):
            temp=list()
            if os.path.isfile(path.rstrip("/")+"/"+file):
                txt=open(path.rstrip("/")+"/"+file, "r", encoding="utf-8").read()
                doc=nlp(txt)
                if uniform==True:
                    temp=[uniform_sentence(e.text) for e in doc.sents if e ]
                else:
                    temp=[e.text.strip() for e in doc.sents if e ]
                sentences+=temp
                if output:
                    if not os.path.exists(output):
                        os.mkdir(output)
                        print(f"Folder '{output}' created.")
                    else:
                        print(f"Folder '{output}' already exists.")
                    with open(output.rstrip("/")+"/"+file, "w", encoding="utf-8") as o:
                        o.write("\n".join(temp))

    else:
        raise Exception("can't recognize the input as a file or a folder, check your input path")
    
    return sentences








