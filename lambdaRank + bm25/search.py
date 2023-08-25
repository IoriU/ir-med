from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy", \
            ]
queries = ["alkylated"]
# for query in queries:
#     print("Query  : ", query)
#     print("Results:")
#     for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 1, notation="bpn"):
#         print(f"{doc:30} {score:>.3f}")
#     print()
print(BSBI_instance.retrieve_tfidf(queries[0], k = 10, notation="bpn"))