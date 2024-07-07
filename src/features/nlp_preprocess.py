def extract_feat_spacy(text,nlp_model):
    
    
    doc = nlp_model(text)
    
    tokens = []

    dict_an_snt = {}

    for i,token in enumerate(doc):
        
        tokens.append(token.text)
        
        dict_an_snt.update({
            
            
            i: {"text": token.text,
            "lemma": token.lemma_,
            "pos":token.pos_,
            "tag":token.tag_,
            "dep":token.dep_,
            "shape":token.shape_,
            "is_alpha":token.is_alpha,
            "is_stop":token.is_stop,
            "morph": str(token.morph),
            "head_index": token.head.i,
            "ner": None 
            }
        })
        
    for ent in doc.ents:
        
        for i in range(ent.start,ent.end): 
            dict_an_snt[i]["ner"] = ent.label_
            dict_an_snt[i]["ner_start_end"] = (ent.start,ent.end)
            
    response = {
        "sentence": text,
        "tokens": tokens,
        "annotated_sentence": dict_an_snt
    }
            
        
    return response