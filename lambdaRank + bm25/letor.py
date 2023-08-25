from ranker import Ranker
from LSI import LSI
from bsbi import BSBIIndex
from compression import VBEPostings
import warnings
warnings.filterwarnings("ignore")

class Letor:
    def __init__(self, model_path):
        self.lsi = LSI(model_path, "nfcorpus/train.docs", "nfcorpus/train.vid-desc.queries", "nfcorpus/train.3-2-1.qrel")
        self.ranker = Ranker(model_path)
    
    def prepare_model(self, config, latent_topic):
        self.lsi.create_model(latent_topic)
        self.ranker.create_model(config, self.lsi)

    def save_model(self):
        self.lsi.save()
        self.ranker.save()

    def load_model(self):
        self.lsi.load()
        self.ranker.load()

    def reranking(self, query, docs_path):
        docs = []
        for doc_path in docs_path:
            docs.append((doc_path[1], open(doc_path[1], "r").read()))
        scores = self.ranker.predict(query, docs, self.lsi)
        did_scores = [x for x in zip(scores, [did for (did, _) in docs])]
        return sorted(did_scores, key = lambda tup: tup[0], reverse = True)


if __name__ == "__main__":
    # {'n_estimators' : 10, 'num_leaves': 5, 'learning_rate' : 0.001}
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')
    query = "alkylated"
    bsbi = BSBI_instance.retrieve_tfidf(query, k = 10, notation="bpn")
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in bsbi:
        print(f"{doc:30} {score:>.3f}")
    print()
    letor_instance = Letor("trained")
    letor_instance.prepare_model({'n_estimators' : 100, 'num_leaves': 40, 'learning_rate' : 0.02}, 200)
    letor_instance.save_model()
    letor_instance.load_model()
    letor = letor_instance.reranking(query, bsbi)
    print("Results :")
    for (score, did) in letor:
        print(did, score)
