# Inferring Symmetry from Natural Language

Computational framework for symmetry inference in natural language.
If you make use of our SIS dataset or symmetry inference systems in your research, please cite our work as:
```
Tanchip, C., Yu, L., Xu, A., and Xu, Y. (2020) Inferring symmetry in natural language. Findings of EMNLP 2020.
```



## Getting Started
### File Introductions
#### Scripts
`feature_extraction.py`: pipeline to extract linguistic features for feature model
`encoders.py`: implementations of different encoders
`model_learning.py`: script for training static embedding model, feature model, and BERT model without fine tuning
`bert_learning.py`: script for training BERT model with fine-tuning
`hybrid_learning.py`: script for training hybrid model
#### Data
`sis.csv`: SIS dataset with sentences and human rating scores collected from Amazon MTurks
`sis_features.csv`: SIS sentences with their linguistc features extracted by the feature encoder


### Requirements
Required Python interpreter: `python3>=3.6`
Required packages:
```
torch
transformers
sentence_transformers
tqdm
matplotlib_venn
nltk
spacy
gensim
```
To run static embedding models, you also need GloVe/word2vec embeddings ready:
```
brew install wget
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz -P ./static_embeddings/

wget http://nlp.stanford.edu/data/glove.6B.zip -P ./static_embeddings/
```
For feature-based models, you also need the Stanford Named Entity Recognizer ready in the `./stanford-ner-4.0.0` folder:
```
https://nlp.stanford.edu/software/CRF-NER.html#Download
```

### Installing Prerequisites

```
pip3 install torch
pip3 install transformers
pip3 install sentence_transformers
pip3 install tqdm
pip3 install nltk
pip3 install matplotlib_venn
pip3 install spacy
pip3 install gensim
```

## Running experiments
### Feature Extraction
To extract linguistic features:
```
python feature_extraction.py 
```
You can then find the extracted features at `./data/sis_with_features.csv`
### Training 
To train simple feature-based linear regression models (using extracted features):
```
python model_learning.py --model_type ft 
```

To train static embedding models:
```
python model_learning.py --model_type static \
                         --embedding glove 

                         
python model_learning.py --model_type static \
                         --embedding word2vec                       
```
To train BERT-based models (without fine tuning):
```
python model_learning.py --model_type bert                         
```
To train BERT-based models (with fine tuning):
```
python bert_learning.py --model_type bert                         
```
To train hybrid transfer learning models:
```
python bert_learning.py --model_type hybrid 
```
After training is completed, relevant statistics will be printed out, and prediction scores ara saved in the SIS dataframe at `./data/sis.csv`
### Evaluation

To compare performances of feature model, BERT model and hybrid model:
```
python model_evaluation.py --analysis sentence_level \
                           --threshold 1.0
```
The error case Venn diagram of three models will be saved to `./figures/error_analysis_venn.pdf`.

The error cases of three models can be found at: `

```
./data/error_cases/bert_error_cases.csv
./data/error_cases/ft_error_cases.csv
./data/error_cases/hybrid_error_cases.csv
```


