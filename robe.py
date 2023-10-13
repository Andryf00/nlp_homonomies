import pickle
from transformers import AutoTokenizer
from utils import *

"""with open("data/wn_homonyms.pkl", "rb") as f:
       dict=pickle.load( f)
lemma = "china"
pos = "n"
print(dict[lemma+"."+pos])
it = dict[lemma+"."+pos]
for i in it:
       print(i)
       for fine, _ in it[i]:
              print(fine)"""
print("mean")
evaluate("prediction_pickles/predictions_mean_bert-large-cased_dev.pkl", "prediction_pickles/total_clusters_bert-large-cased_dev.pkl")
print("min")
evaluate("prediction_pickles/prediction_min_bert-large-cased_dev.pkl", "prediction_pickles/total_clusters_bert-large-cased_dev.pkl")
print("mean euc")
evaluate("prediction_pickles/prediction_euc_mean_bert-large-cased_dev.pkl", "prediction_pickles/total_clusters_bert-large-cased_dev.pkl")
print("min euc")
evaluate("prediction_pickles/prediction_euc_min_bert-large-cased_dev.pkl", "prediction_pickles/total_clusters_bert-large-cased_dev.pkl")
exit()

dataset = "dev"
mapping = read_examples_mapping("new_split/train.json")

coarse_to_fine = load_coarse_to_fine("data/cluster2fine_map.json")
sentences, clusters = read_dataset_filtered("new_split/"+dataset+".json", mapping, coarse_to_fine)


for i,(sample, gold_cluster) in enumerate((zip(sentences, clusters))):
       for j, idx in enumerate(list(sample["instance_ids"].keys())):
              current_candidate_cluster = sample["candidate_clusters"][idx]
              idx = int(idx)
              for sense in sample["wn_candidates"][str(idx)]:
                     found = 0
                     for c in current_candidate_cluster:
                            if sense in [item for sublist in coarse_to_fine[c] for item in sublist]:
                                   found += 1
                     if found != 1:
                            print("_______________")
                            print(sample)
                            print("---")
                            print(sense)



