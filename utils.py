import json
import torch
import pickle
from transformers import AutoTokenizer
from transformers import AutoModel
from typing import Tuple, List, Any, Dict
from nltk.corpus import wordnet as wn

def read_dataset_filtered(path: str, mapping, coarse_to_fine) -> Tuple[List[Dict], List[List[List[str]]]]:
    sentences_s = [] 
    clusters =  []

    with open(path) as f:
        data = json.load(f)
    
    
    correct_same_lemmas = 0
    non_correct_same_lemmas = 0
    correct_diff_lemmas = 0
    non_correct_diff_lemmas = 0
    total_correct = 0
    total_non_correct = 0

    both_same = 0
    none_same = 0 
    correct_same = 0
    non_correct_same = 0

    for sentence_id, sentence_data in data.items():
        #instances in the dataset have 2 different structures, try-except is used to account for this, and process the data accordingly
        try:
            candidate_clusters = list(sentence_data["candidate_clusters"].values())
            keys = list(sentence_data["candidate_clusters"].keys())
            candidate_senses = list(sentence_data["wn_candidates"].values())
            lemmas_total = sentence_data["lemmas"]
            lemmas = []
            for k in keys:
                lemmas.append(lemmas_total[int(k)])
        except:
            candidate_clusters = [sentence_data["candidate_clusters"]]
            candidate_senses = [sentence_data["wn_candidates"]]
            lemmas = [sentence_data["lemma"]]
        #candidate clusters is a list of lists, where the inner list is the list of candidate clusters for an instance
        for i,current_clusters in enumerate(candidate_clusters):

            #we are only intrested in instances that have at least 2 candidate clusters
            if len(current_clusters)>1:
                #to keep track of wheter the correct or wrong example have the same lemma,
                #  element 0 indicates that the correct example has the same lemma
                #  element 1 indicates that the wrong example has the same lemma
                correct_lemmas_check = [False, False] 
                for cluster in current_clusters:
                    succesfull = False
                    #for all the candidate senses for the current instance
                    for cand in candidate_senses[i]:
                        #check if there are examples in the mapping
                        try: 
                            m = mapping[cand]
                        except:
                            #no examples, succesfull remains False and we move on
                            continue
                        #if the current sense is part of the current cluster
                        if cand in [item for sublist in coarse_to_fine[cluster] for item in sublist]:
                            total_correct += len(m["example_tokens"])#keep track of how many correct examples there are
                            succesfull = True #for cluster c we have at least one example available
                            
                            for lemma in m["lemma"]:
                                if lemma == lemmas[i]:
                                    correct_same_lemmas += 1
                                    correct_lemmas_check[0] = True
                                else: correct_diff_lemmas += 1 
                            break
                        else: 
                            total_non_correct += len(m["example_tokens"])
                            for lemma in m["lemma"]:  
                                if lemma == lemmas[i]:
                                    non_correct_same_lemmas += 1
                                    correct_lemmas_check[1] = True
                                else: non_correct_diff_lemmas += 1 
                            pass
                    #if a cluster doesn't have any synset with an example associated to it we move to the next instance 
                    if not succesfull: break

                if succesfull: 
                    data_to_append = {}
                    #instances in the dataset have 2 different structures, try-except is used to account for this, and process the data accordingly
                    try:
                        key_instance_id, value = list(sentence_data["instance_ids"].items())[i]
                        data_to_append["instance_ids"] = {key_instance_id: value}
                        key, value = list(sentence_data["wn_candidates"].items())[i]
                        data_to_append["wn_candidates"] =  {key: value}
                        key, value = list(sentence_data["candidate_clusters"].items())[i]
                        data_to_append["candidate_clusters"] =  {key: value}
                        key, value = list(sentence_data["senses"].items())[i]
                        data_to_append["senses"] =  {key: value}
                        key, value = list(sentence_data["gold_clusters"].items())[i]
                        data_to_append["gold_clusters"] =  {key: value}
                        data_to_append["words"] =  sentence_data["words"]
                    
                    except Exception as e:
                        key_instance_id = sentence_data["instance_ids"][0]
                        data_to_append["instance_ids"] = {key_instance_id: sentence_data["instance_ids"]}
                        data_to_append["candidate_clusters"] =  {key_instance_id: sentence_data["candidate_clusters"]}
                        data_to_append["gold_clusters"] =  {key_instance_id: [sentence_data["cluster_name"]]}
                        data_to_append["words"] =  sentence_data["example_tokens"]
                        data_to_append["wn_candidates"] = {str(key_instance_id):sentence_data["wn_candidates"]}

                    sentences_s.append(data_to_append)
                    clusters.append(data_to_append["gold_clusters"])

                    if correct_lemmas_check[0] and not correct_lemmas_check[1]:
                        correct_same += 1
                    if not correct_lemmas_check[0] and correct_lemmas_check[1]:
                        non_correct_same += 1
                    if correct_lemmas_check[0] and correct_lemmas_check[1]:
                        both_same += 1
                    if not correct_lemmas_check[0] and not correct_lemmas_check[1]:
                        none_same += 1
                else: pass
    print("Number of instances where the correct example contains the same lemma:", correct_same_lemmas)
    print("Number of instances where the wrong example contains the same lemma", non_correct_same_lemmas)
    print("Number of instances where the correct example contains a different lemma:", correct_diff_lemmas)
    print("Number of instances where the wrong example contains a different lemma:", non_correct_diff_lemmas)
    print("Number of instances where only the correct example contains the same lemma", correct_same)
    print("Number of instances where only the wrong example contains the same lemma", non_correct_same)
    print("Number of instances where both the correct and wrong example contain the same lemma", both_same)
    print("Number of instances where neither the correct nor the wrong example contain the same lemma", none_same)
    print("Total correct examples:", total_correct)
    print("Total wrong examples:", total_non_correct)
    return sentences_s, clusters


def read_examples_mapping(path: str) -> Tuple[List[Dict], List[List[List[str]]]]:
    sentences_s, senses_s = [], []
    mapping = {}
    with open(path) as f:
        file = json.load(f)
    case = 0
    for sentence_id, data in file.items():
            
            #instances in the dataset have 2 different structures, try-except is used to account for this, and process the data accordingly
            try:
                if data['synset_name'] not in mapping.keys():
                    mapping[data["synset_name"]] = {
                        "cluster_name" : data["cluster_name"],
                        "example_tokens" : [data["example_tokens"]],
                        "instance_ids" : [data["instance_ids"]],
                        "lemma" : [data["lemma"]]
                    }
                else: 
                    mapping[data["synset_name"]]["instance_ids"].append(data["instance_ids"])
                    mapping[data["synset_name"]]["example_tokens"].append(data["example_tokens"])
                    mapping[data["synset_name"]]["lemma"].append(data["lemma"])
            except:
                for idx in data["instance_ids"].keys():
                    current_sense = data["senses"][idx][0]
                    if current_sense not in mapping.keys():
                        mapping[current_sense] = {
                            "cluster_name" : data["gold_clusters"][idx],
                            "example_tokens" : [data["words"]],
                            "instance_ids" : [[int(idx)]],
                            "lemma" : [data["lemmas"][int(idx)]]
                        }
                    else: 
                        mapping[current_sense]["instance_ids"].append([int(idx)])
                        mapping[current_sense]["example_tokens"].append(data["words"])
                        mapping[current_sense]["lemma"].append(data["lemmas"][int(idx)])

    return mapping




def embed_words(tokenizer, model, words, target_idx):
    '''
    This function takes as input a list of words and the index of the target word.
    It returns the contextualized embeddings of the first bpe of the word.
    '''

    encoding = tokenizer(words, add_special_tokens=True, is_split_into_words=True)
    tokens = torch.as_tensor(encoding["input_ids"]).reshape(1, -1)
    output_idx = encoding.word_to_tokens(target_idx[0]).start
    
    tokens = tokens.to("cuda")

    # embed the example and get the embeddings of the target word
    with torch.no_grad():
        output = model(tokens)
        last_hidden_state = output.last_hidden_state.to("cpu")
        embeddings = last_hidden_state[0, output_idx, :]
        embeddings = embeddings.squeeze(0)
    
    return embeddings


def load_coarse_to_fine(path):
    with open(path) as f:
        data = json.load(f)
    coarse_senses=list(data.keys())
    dict = {}
    for sense in coarse_senses:
        f = data[sense]
        dict[sense] = f
    return dict


def evaluate(path_predictions, path_labels):

    with open(path_predictions, "rb") as f:
        predictions=pickle.load( f)
    with open(path_labels, "rb") as f:
        total_clusters=pickle.load( f)
    correct = 0
    
    for i, pred in enumerate(predictions):
        if type(pred) == type([]):
                pred = pred[0]
        if pred == total_clusters[i]:
                correct+=1

    print(correct,"/",i, "=", correct/i)
