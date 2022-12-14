# nlp-wine-reviews-prediction
NLP sentiment analysis project which aims to predict wine reviews scores using various NLP &amp; Deep Learning techniques. The whole project summary is presented inside [raport.ipynb](raport.ipynb) (in Polish). Wine reviews dataset comes from [kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews).

## Techniques

- Bag Of Words, TF-IDF, Bigrams for text representation using classic ML models
- Naive Bayes (Baseline), Random Forest, SGD, CatBoost using above text representations
- Deep Learning NLP methods such as pretrained word embeddings (word2vec, GloVe)
- LSTM Neural Networks using above word embeddings
- pretrained BERT model from [hugging face](https://github.com/huggingface)
- Final model: stacking CatBoost & LSTM GloVe
