# pick-me-up
> Low Latency Solution to Product categorization
* This is a submission to the Pre Inter IIT-DL Hackathon conducted by TechSoc IITM. 
* Link to the [code](https://colab.research.google.com/drive/1onBXq_Aioqe-1fYAKuSyo9rsrw1WIcFc?usp=sharing)
* Link to the [submission folder](https://drive.google.com/drive/folders/1dCZN3uUF_LRwAvk44SogGbc_GvxW-gFz?usp=sharing)
## Objective
The aim of the competition is to provide a solution to the problem of node assignment to consumer products using a data-driven approach. The key judging criteria includes the practicality of the solution (measured by its latency) and its classification accuracy.
## Approach
The FastText model is used  here to approach the problem. FastText uses a bag-of-words model to extract features and a linear classifier to train the model. Bag-of-words model is preferred in the place of Ngrams as sentence word order doesn't play a decisive role in Product categoization and also Ngrams are computationally intensive. Low dimensional vectors were used to improve performance.
## EDA
The given dataset had a disproportionate split between the classes which could adversely affet the model's training. Hence the data was resampled to balance the dataset. 'Title' and 'Content' were merged together to form the 'text' column and columns like uid were dropped.
## Pre-Processing
Libraries like spacy, gensim, nlkt were used to for preprocessing the data. Firstly Punctuations, Digits, single/bi charecter words were removed. Then stopwords were removed. Finally the sentences were tokenized and lemmatized to minimize the vocabulary and improve the speed.
## Training and Evaluation
The following hyperparameters were finalized after thorough  experimentation. The final train time took about 1 minute.\

Epochs : 9\
Learning Rate : 1.5\
LearningUpdateRate : 100\
minCount : 1\
wordNgrams : 1
Negative Sampling : 5\
Loss : Softmax\
Context Window size : 5\
WordVec Dimension : 100\
Bucket : 20,00,000
## Latency
Some of the initial models that were considered to solve this problems included pre-trained models like BERT classifier, RoBERTa and DistilBERT coupled with CNNs/Feedforward networks. Though these approaches gave comparable results, The models were so computationally expensive and caused a lot of latency. FASTTEXT on the other hand is atleast 15,000x faster than usual approach(while training and testing), hence the most suitable model for our use-case. This is clear by the fact that the model was both trained and tested on CPU.
## References
1. A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
