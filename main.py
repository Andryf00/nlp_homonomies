import torch
import pickle

from utils import *
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModel
from statistics import mean

#this section is just needed to get the instances where bart fails
mapping = read_examples_mapping("new_split/train.json")

coarse_to_fine = load_coarse_to_fine("data/cluster2fine_map.json")

fails={}
for checkpoint in ["bert-large-cased","roberta-large", "google/electra-large-discriminator", "microsoft/deberta-v3-large"]:
       
       for dataset in ["test", "dev"]:
              add_prefix_bool = checkpoint == "roberta-large"
              tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, add_prefix_space=add_prefix_bool)
              model = AutoModel.from_pretrained(checkpoint).to("cuda")
              print(checkpoint, dataset)
              sentences, clusters = read_dataset_filtered("new_split/"+dataset+".json", mapping, coarse_to_fine)
              print(len(sentences))

              #remove the / in the checkpoint name to avoid problems when saving the pickle file 
              if '/' in checkpoint:
                     checkpoint_split = checkpoint.split('/')[1]
              else: checkpoint_split = checkpoint
              print(checkpoint)

              results = {}

              total_senses = []
              total_clusters = []

              predictions_mean = []
              predictions_euc_mean = []
              predictions_min = []
              predictions_euc_min = []

              for i,(sample, gold_cluster) in enumerate(tqdm(zip(sentences, clusters))):
                     for idx in (list(sample["instance_ids"].keys())):
                            current_gold_cluster = gold_cluster[idx][0]
                            current_candidate_cluster = sample["candidate_clusters"][idx]
                            idx = int(idx)
                            sentence = sample["words"]

                            encoding_target = embed_words(tokenizer, model, sentence, [idx])    
                            means = []
                            max_mean_similarity = -1.1
                            max_similarity = -1.1
                            max_similarity_gold = -1.1
                            min_distance_euclidean = float("inf")
                            min_mean_euclidean = float("inf")

                            for sense in sample["wn_candidates"][str(idx)]:
                                   #check if the current candidate sense has some examples available
                                   try:
                                          examples = mapping[sense]["example_tokens"]
                                          lemmas = mapping[sense]["lemma"]
                                   except:
                                          continue
                                          
                                   distances = []
                                   distances_euclidean = []
                                   
                                   for k,(example, lemma) in enumerate(zip(examples,lemmas)):
                                          #for each example get the encoding of the target word
                                          #consider only the examples where the target has the same lemma as in the instance
                                          if lemma != sample["lemma"][str(idx)]:
                                                 continue
                                          else: pass
                                          target_idx = mapping[sense]["instance_ids"][k]

                                          encoding_example =  embed_words(tokenizer, model, example, target_idx)
                                          #we're intrested in bot the cosine and euclidean distance
                                          distance = torch.cosine_similarity(encoding_target.unsqueeze(dim=0), encoding_example.unsqueeze(dim=0)).item()
                                          distance_euclidean = torch.norm(encoding_target - encoding_example).item()
     
                                          
                                          distances.append(distance)      
                                          distances_euclidean.append(distance_euclidean)

                                          if sense in [item for sublist in coarse_to_fine[current_gold_cluster] for item in sublist]:
                                                 distance_gold = distance
                                                 if distance_gold > max_similarity_gold:
                                                        max_similarity_gold = distance_gold

                                          
                                          if distance > max_similarity:
                                                 prediction_min = sense
                                                 predicted_example = example
                                                 max_similarity = distance
                                          if distance_euclidean < min_distance_euclidean:
                                                 prediction_euc_min = sense
                                                 predicted_example = example
                                                 min_distance_euclidean = distance_euclidean

                                   #this additional check is needed, because it might be the case that for a sense there is no example that has the same lemma
                                   #infact the requirement is that at least a sense accros the many associated to a cluster has at least an example that has the same lemma
                                   if len(distances)==0:
                                          mean = 0
                                          mean_euclidean = float('inf')
                                   else:
                                          mean = sum(distances)/len(distances)
                                          mean_euclidean = sum(distances_euclidean)/len(distances_euclidean)
                                   

                                   means.append(mean)
                                   
                                   #update the current prediction
                                   if mean > max_mean_similarity:
                                          prediction_mean = sense
                                          max_mean_similarity = mean
                                   if mean_euclidean < min_mean_euclidean:
                                          prediction_euc_mean = sense
                                          min_mean_euclidean = mean_euclidean
                                   
                                   
                            #store the prediction 
                            total_clusters.append(current_gold_cluster)
                            
                            for c in current_candidate_cluster:
                                   if prediction_mean in [item for sublist in coarse_to_fine[c] for item in sublist]:
                                          predictions_mean.append(c)
                                          predicted = c
                                   if prediction_min in [item for sublist in coarse_to_fine[c] for item in sublist]:
                                          predictions_min.append(c)
                                          predicted2 = c
                                   if prediction_euc_mean in [item for sublist in coarse_to_fine[c] for item in sublist]:
                                          
                                          predictions_euc_mean.append(c)
                                          predicted3 = c
                                          
                                   if prediction_euc_min in [item for sublist in coarse_to_fine[c] for item in sublist]:
                                          predictions_euc_min.append(c)
                                          predicted4 = c

                            
                                          
                            #print("pred:", predicted, current_gold_cluster)
                            print("\n pred2:", predicted2, current_gold_cluster)
                            print(predicted2==current_gold_cluster)
                            #fail cases of bert
                                          
                            if checkpoint == "bert-large-cased" and dataset=="test":
                                   fail={}
                                   fail["instance"] = sample
                                   fail["predicted"] = predicted2
                                   fail["similarity"] = max_similarity
                                   fail["similarity_gold"] = max_similarity_gold
                                   fail["correct"] = predicted2 == current_gold_cluster
                                   fails[i] = fail
              if checkpoint == "bert-large-cased" and dataset=="test":
                     file_name = "fails.json"
                     import json
                     # Using json.dump() to save the dictionary as a JSON file
                     with open(file_name, 'w') as json_file:
                            json.dump(fails, json_file, indent=4)
                     with open("fails.pkl", "wb") as f:
                            pickle.dump(fails, f)

              with open("prediction_pickles/predictions_mean_"+checkpoint_split+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(predictions_mean, f)
              with open("prediction_pickles/prediction_min_"+checkpoint_split+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(predictions_min, f)
              with open("prediction_pickles/total_clusters_"+checkpoint_split+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(total_clusters, f)
              with open("prediction_pickles/prediction_euc_mean_"+checkpoint_split+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(predictions_euc_mean, f)
              with open("prediction_pickles/prediction_euc_min_"+checkpoint_split+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(predictions_euc_min, f)

       

              


