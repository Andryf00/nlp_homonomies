import math
import pickle
from utils import *
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModel
from typing import Tuple, List, Any, Dict
from nltk.corpus import wordnet as wn
from statistics import mean


mapping = read_examples_mapping("new_split/train.json")

coarse_to_fine = load_coarse_to_fine("data/cluster2fine_map.json")


for checkpoint in ["bert-large-cased", "RoBERTa-large", "microsoft/deberta-v3-large", "google/electra-large-discriminator"]:
       for dataset in ["dev", "test"]:
              tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, add_prefix_space=True)

              model = AutoModel.from_pretrained(checkpoint).to("cuda")
              print(checkpoint, dataset)

              sentences, clusters = read_dataset_filtered("new_split/"+dataset+".json", mapping, coarse_to_fine)


              results = {}

              total_senses = []
              total_clusters = []

              predictions_mean = []
              predictions_euc_mean = []
              predictions_min = []
              predictions_euc_min = []

              for i,(sample, gold_cluster) in enumerate(tqdm(zip(sentences, clusters))):
                     for j, idx in enumerate(list(sample["instance_ids"].keys())):
                            current_gold_cluster = gold_cluster[idx][0]
                            current_candidate_cluster = sample["candidate_clusters"][idx]
                            idx = int(idx)
                            sentence = sample["words"]

                            encoding_target = embed_words(tokenizer, model, sentence, [idx])    
                            means = []
                            max_mean_similarity = -1.1
                            max_similarity = -1.1
                            min_distance_euclidean = float("inf")
                            min_mean_euclidean = float("inf")

                            failure = False
                            for sense in sample["wn_candidates"][str(idx)]:
                                   #check if the current candidate sense has some examples available
                                   try:
                                          examples = mapping[sense]["example_tokens"]
                                   except:
                                          continue
                                          
                                   distances = []
                                   distances_euclidean = []

                                   for k,example in enumerate(examples):
                                          #for each example get the encoding of the target word
                                          target_idx = mapping[sense]["instance_ids"][k]

                                          encoding_example =  embed_words(tokenizer, model, example, target_idx)
                                          #we're intrested in bot the cosine and euclidean distance
                                          distance = torch.cosine_similarity(encoding_target.unsqueeze(dim=0), encoding_example.unsqueeze(dim=0))
                                          distance_euclidean = math.dist(encoding_target, encoding_example)        
                                          
                                          distances.append(distance)      
                                          distances_euclidean.append(distance_euclidean)

                                          
                                          
                                          if distance > max_similarity:
                                                 prediction_min = sense
                                                 predicted_example = example
                                                 max_similarity = distance
                                          if distance_euclidean < min_distance_euclidean:
                                                 prediction_euc_min = sense
                                                 predicted_example = example
                                                 min_distance_euclidean = distance_euclidean
                                   
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
                            if len(means) > 1:
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
                                                 
                                   print("pred:", predicted, current_gold_cluster)
                                   print("pred2:", predicted2, current_gold_cluster)
                                   print("pred3:", predicted3, current_gold_cluster)
                                   print("pred4:", predicted4, current_gold_cluster)
                                   #print("DISTANCE(min/mean):", max_similarity, max_distance, "\n Sentence:", sentence, "\n Example:", predicted_example, "\n target", target )

                            #input("...")
              

              with open("prediction_pickles/predictions_mean_"+checkpoint+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(predictions_mean, f)
              with open("prediction_pickles/prediction_min_"+checkpoint+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(predictions_min, f)
              with open("prediction_pickles/total_clusters_"+checkpoint+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(total_clusters, f)
              with open("prediction_pickles/prediction_euc_mean_"+checkpoint+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(predictions_euc_mean, f)
              with open("prediction_pickles/prediction_euc_min_"+checkpoint+"_"+dataset+".pkl", "wb") as f:
                     pickle.dump(predictions_euc_min, f)


              with open("prediction_pickles/prediction_min_"+checkpoint+"_"+dataset+".pkl", "rb") as f:
                     predictions=pickle.load( f)
              with open("prediction_pickles/total_clusters_"+checkpoint+"_"+dataset+".pkl", "rb") as f:
                     total_clusters=pickle.load( f)
              correct = 0
              for i, pred in enumerate(predictions):
                     print(i, pred, total_clusters[i])
                     if type(pred) == type([]):
                            pred = pred[0]
                            print("fixed pred", pred)
                     if pred == total_clusters[i]:
                            correct+=1
                            print("giusto")

              print(correct, i)
              print(correct/i)


       

              


