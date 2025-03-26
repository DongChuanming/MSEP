import edsnlp
import spacy
import unicodedata
import unidecode
import re


# output labels explination : 0=indifferent, 1=absence, 2=presence, 3=former

## Extraction of smoking status

def preannotate_taba(text):
    text=unidecode.unidecode(text.strip().lower())
    smoking_label = '0'
    smoking_status = 'indifferent_sentence'
    nlp_explain = None
    
    # spacy modelling
    nlp_smoking = spacy.blank("fr")
    nlp_smoking.add_pipe("eds.sentences")
    nlp_smoking.add_pipe("eds.normalizer")

    terms = dict(
        former_smoker = ["tabagisme sevre", "ex fumeur", "ex fumeuse", "ancien fumeur", "ancienne fumeuse", 'tabagisme ancien',
                         'tabagisme sevre', 'tabagisme : sevre', 'tabagisme passe', "sevre depuis", "sevrage tabagique oui",
                         'tabac : sevrage oui', 'ex-fumeur', 'ex-fumeuse', 'sevrage tabagique ? oui', "ancien tabagique sevre", 
                         "actuellement abstinent", "sevre il y a", "intoxication tabagique sevree"], 
        
        active_smoker = ['tabagisme', 'tabagisme actif', 'tabagisme estime','tabagisme 10cig/jour','tabagique bpco', 'tabagisme oui',
                         'tabagisme ancien en cours de sevrage',  'tabagiqque 70pa', 'tabagisme poursuivi', 'tabagisme chronique', 
                         'tabagique active', "sevrage tabagique non", "sevrage non", 'sevrage tabagique ? non', 'tabagisme non sevre', 
                         "intoxication tabagique", "intoxication alcoolotabagique", "intoxication alcoolo-tabagique", 
                         "consommation ethylotabagique","consommation ethylo-tabagique", "tabac non sevre"],
        
        never_smoker = ['non fumeur', 'jamais fume', 'tabagisme : 0', 'tabac 0', 'tabagisme non',  'tabac non', 
                       "pas d'intoxication tabagique", "pas d'intoxication alcoolotabagique", "pas d'intoxication alcoolo-tabagique", 
                       "pas de consommation ethylotabagique", "pas de consommation ethylo-tabagique", "pas de tabagisme", 
                       "tabagisme : present ou passe non", "pas d'antecedent de tabagisme"]
    )

    regex = dict(
        former_smoker = r"(tabagisme \d[\s]?cig/j pdt[\s]?\d* ans)|((arreter|arrete|sevre) (\w*\d* ){0,3}(fumer|tabac|tabagique|tabagisme))|(fumer|tabac|tabagique|tabagisme)(\w*\d* ){0,3}(arreter|arrete|sevre)",
        active_smoker = r"((tabac[\s]?\d*)|(tabac|tabagisme) a?[\s]?\d*[\s]?pa)|(tabagisme \+\+)|(fumeur|fumeuse|fumer|fume|tabagisme (\w* ){0,3}actif)|(tabac|tabagisme) (\w*\d* ){0,3}[^0\d*] pa|tabagisme (\w*\d* ){0,3}[^0\d*] cig*. |intoxication tabagique|(tabagisme stoppe pendant \d+ an et repris)",
        never_smoker = r"(non_fumeur|non fumeur|non-fumeur|(non|ni|pas) (\w* ){0,2}(fumer|fume|tabac))|tabac[\s]?[=]?[:]?[\s+]?0|(tabagisme: aucun)"
    )
    
    nlp_smoking.add_pipe("eds.matcher",
                         config = dict(
                             terms = terms,
                             regex = regex,
                             attr = "LOWER",
                             term_matcher = "exact",
                             term_matcher_config = {},),)
    nlp_smoking.add_pipe("eds.negation")
    nlp_smoking.add_pipe("eds.family")

    # Add qualifiers
    doc = nlp_smoking(text)
    
    entities = []
    family=False
    for ent in doc.ents:
        smoking_status = ent.label_
        family=ent._.family
        nlp_explain = dict(
        lexical_variant = ent.text,
        label = ent.label_,
        negation = ent._.negation,
        family = ent._.family,
        )
        entities.append(nlp_explain)
    if family :
        return smoking_label
    
    if smoking_status == "never_smoker":
        smoking_label="1"
    elif smoking_status == "active_smoker":
        smoking_label="2"
    elif smoking_status == "former_smoker":
        #if not family:
        smoking_label="3"
    return smoking_label






# preannotate family history of cancer

def preannotate_fam(text):
    text=unidecode.unidecode(text.strip().lower())
    tumor_family = None
    cancer_history="0"
    family=False
    negation=False
    nlp_explain = None
    
    # spacy modelling
    nlp_atcd_tum = spacy.blank("fr")

    nlp_atcd_tum.add_pipe("eds.sentences")
    nlp_atcd_tum.add_pipe("eds.normalizer")
    # à ranger 
    terms = dict(
        atcd_vessie = ["cancer de la vessie chez le pere",
                       "cancer de la vessie chez la mere",
                       "pere : cancer vessie",
                       "mere : cancer vessie",
                       "tumeur de la vessie chez le pere",
                       "tumeur de la vessie chez la mere",
                       "pere : tumeur vessie",
                       "mere : tumeur vessie",
                       "frere decede d'un cancer de prostate",
                       "mere : lithiase vesiculaire",
                       "frere DCD  cancer de la vessie",
                       "frere decede d'un cancer de la vessie",
                       "le pere a eu un cancer prostatique",
                       "antecedent familial de cancer prostatique chez le pere",
                       "K vessie chez sa mere",
                       "Z8000",
                       "cancer",
                       "pere 2 IDM, le premier a 53 ans, neo vessie et rein",
                       "Antecedent familial de cancer de prostate chez le pere",
                       "antecedent familial de cancer de prostate", 
                       "Pere decede d'un cancer de vessie", 
                       "Pere : cancer estomac, cordes vocales et prostate ", 
                       "mere : lithiase vesiculaire",
                       "Paternels:  cancer gorge, cancer prostate.",
                       "Paternels:  K de la vessie",
                       "Paternels:  Multiples cancers (leucemie, colon, etc...) des deux cotes de la famille",
                       "2 antecedents familiaux de cancer de prostate",
                       "Cancer de prostate chez son pere",
                       "histoire familiale de cancer +++++++",
                       "Antecedents familiaux : ses deux parents sont morts d'un cancer d'origine inconnue", 
                       "Mere: decede d'un cancer generalise", 
                       "pere decede d'un cancer (ne sait pas de quoi)",
                       "atcd familiaux de K",
                       "Pere decede d'un cancer de prostate",
                       "Antecedent familial de cancer de prostate chez oncle",
                       "pere a eut cancer prostate",
                       "polype de la prostate",
                       "tumeur de vessie chez le pere et l'oncle",
                       "Cancer de la prostate chez le pere",
                       "familial de cancer de prostate chez le pere et 3 freres",
                       "pere : prostate",
                       "nombreux cancers prostatiques dans la branche paternel",
                       "Pere : cancer de prostate",
                       "cancer de prostate chez son oncle",
                       "antecedent  familial  de  cancer  de  vessie",
                       "pere decede du cancer de la prostate",
                       "tenu d antecedents familiaux de polypes",
                       "cancer prostatique chez son pere",
                       "ATCD familial chez le pere d'un deces par adenocarcinome prostatique",
                       "Il a un antecedent familial de cancer prostatique"
                      ],
        
        atcd_colon = ["cancer du colon chez sa mere",
                      "K COLON pere",
                      "pere: K du colon",
                      "mere decedee d'un cancer du colon",
                      "pere decede d'un cancer du colon",
                      "Terrain : Polypes coliques (deces de sa mere a 81 ans d'un cancer colique)",
                      "contexte d'antecedents familiaux de CCR",
                      "CCR chez la mere",
                      "CCR chez le pere",
                      "antecedent familial de CCR au 1er degre",
                      "un cancer du colon chez le pere", 
                      "antecedents familiaux de neoplasie colique",
                      "antecedent familial de cancer colorectal",
                      "antecedents familiaux de cancer colorectal",
                      "Maternels:  cancer colon",
                      "antecedents familiaux de CCR",
                      "CCR chez la mere",
                      "Cancer du rectum chez la mere",
                      "Cancer du colon chez un oncle maternel",
                      "Cancer du colon chez la cousine germaine de la mere",
                      "Pere+ 2 soeurs  cancer colon",
                      "cancer du colon chez sa fille",
                      "frere cancer rectal",
                      "antecedents familiaux de polypes coliques",
                      "Grand-pere maternel decede d?un cancer du colon.",
                      "Antecedents familiaux cancer du colon",
                      "Antecedents familiaux cancer colique",
                      "un antecedent familial de cancer du colon au 1er degre chez son pere",
                      "antecedent familial de cancer colique au premier degre.",
                      "avec des antecedents familiaux de cancer du spectre",
                      "pere et mere traites endoscopiquement pour polype",
                      "polypes mere",
                      "polypes chez sa mere",
                      "polypes pere",
                      "pere : cancer colo rectal",
                      "Cancer du colon chez la mere",
                      "Il a un antecedent familial de cancer du colon chez son pere",
                      "Chez le pere, cancer rectal",
                      "Cancer colorectal chez son pere", 
                      "Cancer colorectal chez sa mere",
                      "pere: cancer du colon",
                      "frere polype colon",
                      "(antecedent familial de cancer du colon)"
                     ], 
        
        atcd_rein = ["insuffisance renale chronique chez la mere",
                     "mere porteuse d'une insuffisance renale",
                     "mere : cancer du rein",
                     "pere: \"pb de rein\"",
                     "Pere : cancer de la plevre et du rein.",
                     "Pere decede d\'un cancer du rein, cancer chez la mere (tumeur primitive indeterminee)",
                     "polykystose renale familiale",
                     "Fille : Lithiase renale",
                     "notion de cancer du rein chez son pere",
                     "une mere qui a presente un neo du rein opere",
                     "4 freres et soeurs  dcd cancer poumon et cancer rein",
                     "Pere cancer rein"
                    ]
    
    )
                       
    
    nlp_atcd_tum.add_pipe("eds.matcher", 
                 config = dict(
            terms=terms,
            attr="NORM",
            term_matcher="exact",#"simstring",
            term_matcher_config={},
        ))

    
    # Add qualifiers
    nlp_atcd_tum.add_pipe("eds.negation")
    nlp_atcd_tum.add_pipe("eds.family")
    
    doc = nlp_atcd_tum(text)

    entities = []
    for ent in doc.ents:
        tumor_family = ent.label_
        negation = ent._.negation
        family = ent._.family
        nlp_explain = dict(
        lexical_variant = ent.text,
        label = ent.label_,
        negation = ent._.negation,
        family = ent._.family
        )
        entities.append(nlp_explain)
        
    
    # Ici, on exclure les confirmations de cancer chez le patient lui-même, concentre seulement sur les membre de famille
    #if tumor_family == "atcd_vessie" and not negation and family:
    if tumor_family in ["atcd_vessie","atcd_colon", "atcd_rein"] and not negation: 
        if family:
            cancer_history  = "2"
        #return tumor_family
    elif tumor_family is not None:
        if family:
            cancer_history = "1"
        #return tumor_family

    #with open("log_family_history_extraction.txt", 'a') as file:
    #    file.write("%s, %s : %s ---> label %s \n" % (id_pat, id_doc, text, tumor_family)) 
    
    return cancer_history #, nlp_explain












# liste de negation pour les règles d'annotations des symptômes (hypertension, diabètes etc.)
list_neg=["pas ", "pas de ", "pas d'","absence ", "absence de ", "absence d'", 
          "non ", "sans ", "sans signe de ", "sans signe d'","ni de ", "ni ", "aucun ", 
          "aucun signes d'", "aucun signes de ", "pas de signe d'","pas de signe de ", "pas de signes de ", 
          "pas de signes d'", "pas d'autre signe d'", "pas d'autre signe de ", "pas de signes en faveur d'une "]

list_neg_post=[" absent", " absente", " non trouve", " non trouvee", 
               " non detecte",  " non detectee", " traite",  " traitee",
               " deja traite", " deja traitee"," pas trouve", " pas trouvee"," pas detecte"," pas detectee", " arrete", " arretee"]




## Hypertension extraction
# combiner les éléments dans la liste de négation avec les éléments dans la liste des symptômes pour former la négation de la présence de symptôtme
list_hyper=["hypertension", "hypertensive",  "hta", ]
patterns_neg_hyper=[]
for a in list_hyper:
    for b in list_neg:
        patterns_neg_hyper.append(b+a)
    for c in list_neg_post:
        patterns_neg_hyper.append(a+c)


def preannotate_hyper(sent):
    sent=unidecode.unidecode(sent.strip().lower())
    hyper="0"
    for p in patterns_neg_hyper:
        pat=re.compile("\W+"+p+"s?\W+")
        #if " "+p+" " in " "+sent+" ":
        if re.search(pat,nsent):
            hyper="1"
            return hyper
    for m in list_hyper:
        pat=re.compile("\W+"+m+"s?\W+")
        #if " "+p+" " in " "+sent+" ":
        if re.search(pat,nsent):
            hyper="2"
            return hyper
    return hyper




## Diabetes extraction
list_diab=["diabete", "diabetes", "diabetique", "did", "dnid", "dt2"]
patterns_neg_diab=[]
for a in list_diab:
    for b in list_neg:
        patterns_neg_diab.append(b+a)
    for c in list_neg_post:
        patterns_neg_diab.append(a+c)


def preannotate_diab(sent):
    #patterns_neg=["pas diabetique", "pas de diabete", "absence de diabete", "absence diabete", ]
    #patterns_pos=["diabete", "DNID", "DID", "DT2", diabete de type OR diabetique"]
    sent=unidecode.unidecode(sent.strip().lower())
    diab="0"
    nsent=" "+sent+" " 
    for p in patterns_neg_diab:
        pat=re.compile("\W+"+p+"s?\W+")
        #if " "+p+" " in " "+sent+" ":
        if re.search(pat,nsent):
            diab="1"
            return diab
    for m in list_diab:
        pat=re.compile("\W+"+m+"s?\W+")
        if re.search(pat,nsent):
            diab="2"
            return diab
    return diab



## Heart failure extraction
list_cardia=["insuffisance cardiaque", "ic", "ivd", "ivg", "decompensation cardiaque", "decompensation cardiaque globale", "dc", "absence cardiaque", "decompensation cardique", "ivg/ivd", "icd", "icg", "insuffisance cardique", "dysfonction bi ventriculaire"]
patterns_neg_cardia=[]
for a in list_cardia:
    for b in list_neg:
        patterns_neg_cardia.append(b+a)
    for c in list_neg_post:
        patterns_neg_cardia.append(a+c)

def preannotate_cardia(sent):
    sent=unidecode.unidecode(sent.strip().lower())
    cardia="0"
    nsent=" "+sent+" " 
    for p in patterns_neg_cardia:
        pat=re.compile("\W+"+p+"s?\W+")
        #if " "+p+" " in " "+sent+" ":
        if re.search(pat,nsent):
        #if p in sent:
            cardia="1"
            return cardia
    for m in list_cardia:
        pat=re.compile("\W+"+m+"s?\W+")
        if re.search(pat,nsent):
        #if m in sent:
            cardia="2"
            return cardia
    return cardia





## COPD extraction

list_copd=["bronchite chronique", "bpco", "bcpo", "pbco", 
           "broncho pneumopathie chronique obstructive", 
           "bronchopneumopathie chronique obstructive"]
patterns_neg_copd=[]
for a in list_copd:
    for b in list_neg:
        patterns_neg_copd.append(b+a)
    for c in list_neg_post:
        patterns_neg_copd.append(a+c)

# on obtient une liste qui contient la négation des symptôme, comme "pas de BPCO" etc.


# preannotate COPD
#sortir en str -, 0, 1,  qui représentent respectivement "phrase indifférente pour COPD", "pas de COPD" et "avoir COPD"

def preannotate_copd(sent):
    sent=unidecode.unidecode(sent.strip().lower())
    copd="0"
    nsent=" "+sent+" " 
    for p in patterns_neg_copd:
        pat=re.compile("\W+"+p+"s?\W+")
        #if " "+p+" " in " "+sent+" ":
        if re.search(pat,nsent):
        #if " "+p+" " in " "+sent+" ":
            copd="1"
            return copd
    for m in list_copd:
        pat=re.compile("\W+"+m+"s?\W+")
        if re.search(pat,nsent):
        #if " "+m+" " in " "+sent+" ":
            copd="2"
            return copd
    return copd






def preannotate_any(sent, list_keywords, list_neg=list_neg, list_neg_post=list_neg_post):
    patterns_neg_any=[]
    for a in list_keywords:
        for b in list_neg:
            patterns_neg_any.append(b+a)
        for c in list_neg_post:
            patterns_neg_any.append(a+c)
    
    sent=unidecode.unidecode(sent.strip().lower())
    label="0"
    nsent=" "+sent+" " 
    for p in patterns_neg_any:
        pat=re.compile("\W+"+p+"s?\W+")
        #if " "+p+" " in " "+sent+" ":
        if re.search(pat,nsent):
        #if " "+p+" " in " "+sent+" ":
            label="1"
            return label
    for m in list_keywords:
        pat=re.compile("\W+"+m+"s?\W+")
        if re.search(pat,nsent):
        #if " "+m+" " in " "+sent+" ":
            label="2"
            return label
    return label
            
            
    

