Code to run the experiments I carried out for the paper [Analyzing Homonymy Disambiguation Capabilities of Pretrained Language Models](https://aclanthology.org/2024.lrec-main.83/).

The goal is to evaluate the ability of pre-trained language models to disambiguate homonymous senses. This set of experiments relies on a dataset where each entry consists of:
- a test phrase which contains the target word to disambiguate 
- a list of candidate senses, each with an associated phrase

Then we measure the cosine and euclidean distance between two embeddings: 
- embedding of the word to disambiguate in the test phrase
- embedding of the word to disambiguate in the phrase associated with each sense

The sense associated with the closest embedding is the predicted sense. The performance measure is the accuracy of the prediction (ground truth about the correct sense is given).


