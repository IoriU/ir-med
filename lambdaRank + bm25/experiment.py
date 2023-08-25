import math
import re
from letor import Letor
from bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

######## >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
  """ menghitung search effectiveness metric score dengan 
      Discounted Cumulative Gain

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score DCG
  """
  score = 0
  for i in range(len(ranking)):
    score += (1/math.log2(i+2)) * ranking[i]
  return score

def prec(ranking, k):
  score = 0
  for i in range(1, k+1):
    score += (1/k) * ranking[i-1] 
  return score 

def ap(ranking):
  """ menghitung search effectiveness metric score dengan 
      Average Precision

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score AP
  """
  # TODO
  score = 0
  R = len(list(filter(lambda x: x == 1, ranking)))
  for i in range(len(ranking)):
    if (ranking[i]):
      score += prec(ranking, i+1)/R
  return score
######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels


######## >>>>> EVALUASI !

def eval_tfidf(qrels, query_file = "queries.txt", notation="ltn", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ap_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k, notation=notation):
          did = int(re.search(r'.*\\.*\\(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))

  print("Hasil evaluasi TF-IDF terhadap 30 queries dengan notasi {notation}".format(notation=notation))
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("AP score  =", sum(ap_scores) / len(ap_scores))


def eval_bm25(qrels, query_file = "queries.txt", k = 1000, k1 = 1.2, b = 0.75):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ap_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      for (score, doc) in BSBI_instance.retrieve_bm25(query, k = k, k1=k1, b=b):
          did = int(re.search(r'.*\\.*\\(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))

  print("Hasil evaluasi BM25 terhadap 30 queries dengan k1 {k1} dan b {b}".format(k1=k1, b=b))
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("AP score  =", sum(ap_scores) / len(ap_scores))

def eval_letor_bm25(qrels, config, latent_topic, query_file = "queries.txt", k = 1000, k1 = 1.2, b = 0.75):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')
  letor_instance = Letor("trained")
  letor_instance.prepare_model(config, latent_topic)
  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ap_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      bsbi = BSBI_instance.retrieve_bm25(query, k = k, k1=k1, b=b)
      letor = letor_instance.reranking(query, bsbi)
      for (score, doc) in letor:
          did = int(re.search(r'.*\\.*\\(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))

  print("Hasil evaluasi BM25 Letor terhadap 30 queries dengan k1 {k1} dan b {b}".format(k1=k1, b=b))
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("AP score  =", sum(ap_scores) / len(ap_scores))
  return sum(rbp_scores) / len(rbp_scores)

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  # eval_bm25(qrels)
  # config = {'n_estimators' : 5, 'num_leaves': 5, 'learning_rate' : 0.001}
  # eval_letor_bm25(qrels, config, 50)

  print("Raw BM25")
  eval_bm25(qrels)
  print("BM25 + Letor")
  param = {'n_estim' : [5, 20, 50],
          'leaf': [5,  20, 50],
          'learn' : [0.001, 0.01, 0.1],
          'latent': [50, 200, 350]}
  n = len(param["n_estim"]) * len(param["leaf"]) * len(param["learn"]) * len(param["latent"])
  with tqdm(total=n) as pbar:
    for i in param["n_estim"]:
      for j in param["leaf"]:
        for k in param["learn"]:
          for l in param["latent"]:
            config = {'n_estimators' : i, 'num_leaves': j, 'learning_rate' : k}
            print(config, "latent", l)
            eval_letor_bm25(qrels, config, l)        
            pbar.update(1)