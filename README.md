# nlp-wine-reviews-prediction
NLP sentiment analysis project which aims to predict wine reviews scores using various NLP &amp; Deep Learning techniques. <br>
The whole project summary is presented inside [raport.ipynb](raport.ipynb) (in Polish).

## Techniques

- Bag Of Words, TF-IDF, Bigrams for text representation using classic ML models
- Naive Bayes (Baseline), Random Forest, SGD, CatBoost using above text representations
- Deep Learning NLP methods such as pretrained word embeddings (word2vec, GloVe)
- LSTM Neural Networks using above word embeddings
- Final model: stacking CatBoost & LSTM GloVe, which achieved 80% accuracy
