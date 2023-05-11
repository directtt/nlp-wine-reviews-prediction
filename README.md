# nlp-wine-reviews-prediction

## Table of Contents
- [Project summary](#project-summary)
- [Techniques](#techniques)
- [Results](#results)

## Project summary
NLP text classification project which aims to predict wine reviews scores using various NLP &amp; Deep Learning techniques. The whole project summary is presented inside [raport.ipynb](raport.ipynb) (in Polish). Wine reviews dataset comes from [kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews).


## Techniques

- Bag Of Words, TF-IDF, Bigrams for text representation using classic ML models
- Naive Bayes (Baseline), Random Forest, SGD, CatBoost using above text representations
- Deep Learning NLP methods such as pretrained word embeddings (word2vec, GloVe)
- LSTM Neural Networks using above word embeddings
- pretrained huggingface transformer models ([DistilBERT](https://huggingface.co/distilbert-base-uncased), [RoBERTa](https://huggingface.co/roberta-base), [GPT-2](https://huggingface.co/gpt2))
- stacking LSTM & CatBoost models

## Results
| Model                                               | Accuracy                        |
|-----------------------------------------------------|---------------------------------|
| Naive Bayes (baseline)                              | 0.66                            |
| Random Forest (all features)                        | 0.73                            |
|**CatBoost (all features)**  | **0.78** |
| LSTM (GloVe)                                        | 0.75                            |
| **RoBERTa (huggingface)**   | **0.77** |
| **Stacking LSTM & CatBoost**| **0.80** |

In terms of accuracy results, the best model turned out to be stacking LSTM & CatBoost models. 
It's worth to mention that CatBoost model itself was the best performing classic ML model. 
However, if I would choose final model for production I would go with RoBERTa model, 
because of its reliability and superiority in terms of other metrics (MCC, precision, recall, f1-score).
