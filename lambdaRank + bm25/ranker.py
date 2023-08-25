import lightgbm as lgb
import pickle
import os
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
import random

class Ranker:
    
    def __init__(self, model_path):
        self.model_path = model_path

    def create_model(self, config, lsi):
        self.model = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = config['n_estimators'],
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = config['num_leaves'],
                    learning_rate = config['learning_rate'],
                    max_depth = -1)
        X = []
        Y = []
        dataset, group_qid_count = lsi.get_dataset()
        for (query, doc, rel) in dataset:
            X.append(lsi.features(query, doc))
            Y.append(rel)
        X = np.array(X)
        Y = np.array(Y)
        self.model.fit(X, Y,
           group = group_qid_count,
           verbose = 10)

    def save(self):
        if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
        pickle.dump(self.model, open(os.path.join(self.model_path, 'model.pkl'), 'wb+'))
        print("Model saved")
    
    def load(self):
        try:
            self.model = pickle.load(open(os.path.join(self.model_path, 'model.pkl'), 'rb'))
            return True
        except:
            print("Model not found")
            return False
    
    def predict(self, query, docs, lsi):
        X_unseen = []
        for doc_id, doc in docs:
            X_unseen.append(lsi.features(query.split(), doc.replace("\n", " ").split()))
        X_unseen = np.array(X_unseen)
        return self.model.predict(X_unseen)

