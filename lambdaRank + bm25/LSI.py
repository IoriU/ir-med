import lightgbm as lgb
import pickle
import os
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
import random

class LSI:
    def __init__(self, output_dir, document_path, query_path, train_path):
        self.document_path = document_path
        self.query_path = query_path
        self.train_path = train_path
        self.output_dir = output_dir
        self.doc = None
        self.query = None
        self.qrels = None
        self.dataset = None

    def parse_doc(self):
        documents = {}
        with open(self.document_path, encoding='utf-8') as file:
            for line in file:
                doc_id, content = line.split("\t")
                documents[doc_id] = content.split()
        return documents

    def parse_query(self):
        queries = {}
        with open(self.query_path, encoding='utf-8') as file:
            for line in file:
                q_id, content = line.split("\t")
                queries[q_id] = content.split()
        return queries

    def create_qrels(self):
        documents = self.get_doc()
        queries = self.get_query()
        q_docs_rel = {} # grouping by q_id terlebih dahulu
        with open(self.train_path, encoding='utf-8') as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in queries) and (doc_id in documents):
                    if q_id not in q_docs_rel:
                        q_docs_rel[q_id] = []
                    q_docs_rel[q_id].append((doc_id, int(rel)))
        return q_docs_rel

    def create_dataset(self):
        documents = self.get_doc()
        queries = self.get_query()
        q_docs_rel = self.get_qrels()
        NUM_NEGATIVES = 1
        # group_qid_count untuk model LGBMRanker
        group_qid_count = []
        dataset = []
        for q_id in q_docs_rel:
            docs_rels = q_docs_rel[q_id]
            group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                dataset.append((queries[q_id], documents[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            dataset.append((queries[q_id], random.choice(list(documents.values())), 0))
        return dataset, group_qid_count

    def get_doc(self):
        if (not self.doc):
            self.doc = self.parse_doc()
        return self.doc
    
    def get_query(self):
        if (not self.query):
            self.query = self.parse_query()
        return self.query

    def get_qrels(self):
        if (not self.qrels):
            self.qrels = self.create_qrels()
        return self.qrels

    def get_dataset(self):
        if (not self.dataset):
            self.dataset = self.create_dataset()
        return self.dataset
    
    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.lsi[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]


    def create_model(self, num_latent_topics):
        self.NUM_LATENT_TOPICS = num_latent_topics
        documents = self.get_doc()
        self.dictionary = Dictionary()
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
        self.lsi = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS) # 200 latent topics

    def save(self):
        if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        pickle.dump(self.lsi, open(os.path.join(self.output_dir, 'lsi.pkl'), 'wb+'))
        data = {"dictionary": self.dictionary, "NUM_LATENT_TOPICS": self.NUM_LATENT_TOPICS}
        pickle.dump(data, open(os.path.join(self.output_dir, 'data.pkl'), 'wb+'))
        print("LSI saved")

    def load(self):
        try:
            self.lsi = pickle.load(open(os.path.join(self.output_dir, 'lsi.pkl'), 'rb'))
            data = pickle.load(open(os.path.join(self.output_dir, 'data.pkl'), 'rb'))
            self.dictionary = data["dictionary"]
            self.NUM_LATENT_TOPICS = data["NUM_LATENT_TOPICS"]
            return True
        except:
            print("LSI not found")
            return False