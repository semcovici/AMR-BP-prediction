from nltk.tokenize import word_tokenize
import re

def remove_num_text(string):
    
    if type(string) != str:
        return None 
    
    
    
    # Expressão regular para corresponder ao padrão [palavra]-[numero]
    padrao = re.compile(r'^(\w+)-\d+$')
    
    # Tentativa de encontrar correspondência
    correspondencia = padrao.match(string)
    
    if correspondencia:
        # Se encontrar, retorna apenas a palavra (primeiro grupo da regex)
        return correspondencia.group(1)
    else:
        # Caso contrário, retorna a string original
        return string

def parse_alignment(text):
    alignment_line = ""
    
    # Procura pela linha que contém os alinhamentos bramr
    for line in text.split('\n'):
        if line.startswith("# ::alignments-bramr"):
            alignment_line = line
            break
    
    # Verifica se a linha de alinhamento está vazia ou não foi encontrada
    if not alignment_line or alignment_line.strip() == "# ::alignments-bramr":
        return None
    
    # Remove o prefixo para obter os alinhamentos
    alignment_line = alignment_line.replace("# ::alignments-bramr ", "").strip()
    
    # Inicializa o dicionário de resultados
    alignment_dict = {}
    
    # Processa cada par de alinhamento
    for pair in alignment_line.split():
        token_index, node = pair.split('-')
        alignment_dict[node] = int(token_index)
    
    return alignment_dict

def amr_to_dict(amr):
    
    dict_amr = {}
    id_snt = amr.id
    nodes = amr.nodes
    metadata = amr.metadata
    tokens = amr.tokens
    graph = amr.graph_string()
    
    dict_edges = {}
    id = 0
    # cria dict dos nos
    for node1_id, edge_value, node2_id in amr.edges:
        id +=1
        
        dict_edges.update({
            f'edge {id}': {
                'nodes_ids': (node1_id,node2_id),
                'nodes': (nodes.get(node1_id),nodes.get(node2_id)),
                'value': edge_value
            }})
        
    if tokens == []:
        snt = metadata['snt']
        tokens_nltk = word_tokenize(snt, language='portuguese')
    else: 
        snt = " ".join(tokens)
        tokens_nltk = word_tokenize(snt, language='portuguese')
      
    
    # verfica se ha tokens na anotacao (se nao tiver tokens, ele considera snt como tokens)
    if "tok pt" not in amr.amr_string():
        tokens = []
        
    dict_amr.update({'id': id_snt})
    dict_amr.update({'nodes': nodes})
    dict_amr.update({"edges": dict_edges})
    dict_amr.update(metadata)
    dict_amr.update({"graph": graph})
    dict_amr.update({"tok pt": tokens})
    dict_amr.update({"tokens_nltk": tokens_nltk})
    dict_amr.update({"snt": snt})
    return dict_amr

def pos_tagger_nltk(
    tokenized_sentence,
    tagger_nltk
):
    
    # anota os tokens 
    pos_tags_annotation = tagger_nltk.tag(tokenized_sentence)
    
    # cria lista apenas com os tags, sem o token
    pos_tags = [tag[1] for tag in pos_tags_annotation]
    
    return pos_tags


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