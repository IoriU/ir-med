import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.stemmer = SnowballStemmer(language = "english")
        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []
        self.load()

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""
        try:
            with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
                self.term_id_map = pickle.load(f)
            with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
                self.doc_id_map = pickle.load(f)
        except:
            print("Create File")

    def clean_text(self, text):
        text = text.lower()
        text = re.sub("\s+", " ", text) # Menghilangkan spasi berlebih
        text = re.sub("[^\w\s]", " ", text) # Menghilangkan tanda baca'
        stop_words = set(stopwords.words('english'))
        res = word_tokenize(text)
        res = [w for w in res if w not in stop_words]
        return [self.stemmer.stem(w) for w in res]

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        # TODO
        collection_path = os.path.join(self.data_dir, block_dir_relative)
        td_pairs = []
        for filename in os.listdir(collection_path):
            with open(os.path.join(collection_path, filename), 'r') as f: # open in readonly mode
                doc_id = self.doc_id_map[os.path.join(collection_path, filename)]
                text = f.read()
                tokens = self.clean_text(text)
                for token in tokens:
                    term_id = self.term_id_map[token]
                    td_pairs.append((term_id, doc_id))
        return td_pairs
    
    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = {}
            if (doc_id not in term_dict[term_id].keys()):
                term_dict[term_id][doc_id] = 0
            term_dict[term_id][doc_id] +=1
        for term_id in sorted(term_dict.keys()):
            posting_list = sorted(list(term_dict[term_id]))
            tf_list = []
            for i in posting_list:
                tf_list.append(term_dict[term_id][i])
            index.append(term_id, posting_list, tf_list)
        
        

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def bm25_okapi(self, tf, df, N, dl, avdl,k1 = 1.2, b=0.75):
        """
        Menghitung score dokumen dengan metode BM25 tanpa normalisasi panjang dokumen.

        Parameters
        ----------
        tf: int
            Term frequency 
        df: int
            Document frequency
        N:  int
            Total document
        k1: float
            Fine tuning
        """
        return math.log10(N/df) * ((k1+1) * tf)/(k1 * ((1-b) + b * (dl/avdl)) + tf)

    def tf_n(self, tf):
        return tf
    def tf_l(self, tf):
        return (1 + math.log10(tf)) if tf > 0 else 0
    def tf_b(self, tf):
        return 1 if tf > 0 else 0
    def df_n(self, df, N):
        return 1
    def df_t(self, df, N):
        return (math.log10(N/df))
    def df_p(self, df, N):
        return max(0, math.log10((N-df)/df))

    def tfidf(self, notation, raw_tf, raw_df, N):
        """
        Menghitung score dokumen dengan metode BM25 tanpa normalisasi panjang dokumen.

        Parameters
        ----------
        notation: string
            Smart notation for tf-idf
        tf: int
            Term frequency 
        df: int
            Document frequency
        N:  int
            Total document
        """
        tf = 0
        df = 0
        if (notation[0] == "n"):
            tf = self.tf_n(raw_tf)
        elif (notation[0] == "l"):
            tf = self.tf_l(raw_tf)
        elif (notation[0] == "b"):
            tf = self.tf_b(raw_tf)
        if (notation[1] == "n"):
            df = self.df_n(raw_df, N)
        elif (notation[1] == "t"):
            df = self.df_t(raw_df, N)
        elif (notation[1] == "p"):
            df = self.df_p(raw_df, N)
        return tf * df

    def retrieve_tfidf(self, query, k = 10, notation = "ltn"):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        tokens = self.clean_text(query)
        page_dict = {}
        result = []
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            for token in tokens:
                term_id = self.term_id_map[token]
                postings = merged_index.get_postings_list(term_id)
                for (doc_id, tf) in postings: 
                    doc_str = self.doc_id_map[doc_id]
                    if (doc_str not in page_dict):
                        page_dict[doc_str] = 0
                    page_dict[doc_str] += self.tfidf(notation, tf, merged_index.postings_dict[doc_id][1], len(self.doc_id_map))
            result = list(zip(page_dict.values(), page_dict.keys()))
        result = sorted(result, key=lambda x: x[0], reverse=True)
        return result[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        tokens = self.clean_text(query)
        page_dict = {}
        result = []
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            for token in tokens:
                term_id = self.term_id_map[token]
                postings = merged_index.get_postings_list(term_id)
                for (doc_id, tf) in postings: 
                    doc_str = self.doc_id_map[doc_id]
                    if (doc_str not in page_dict):
                        page_dict[doc_str] = 0
                    page_dict[doc_str] += self.bm25_okapi(tf, merged_index.postings_dict[doc_id][1], len(self.doc_id_map), \
                        merged_index.doc_length[doc_id], merged_index.doc_length_average, k1, b)
            result = list(zip(page_dict.values(), page_dict.keys()))
        result = sorted(result, key=lambda x: x[0], reverse=True)
        return result[:k]

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
        self.save()
        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
