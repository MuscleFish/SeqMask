from DataPreprocess import cut_sentences
import joblib
import numpy as np

class FastTextDeal(object):
    def __init__(self):
        self.FastTextModel = joblib.load("./FastTextModels/fastText_20210126.model")
        self.paper_emb_list = []

    def getToken(self, words=""):
        if isinstance(words, str):
            return [cut_sentences(words)]
        if isinstance(words, list):
            return [cut_sentences(w) for w in words]
        return []

    def clean_sentences(self, max_sentence):
        clear_pem = []
        for pem in self.paper_emb_list:
            pem_shape = pem.shape
            if len(pem_shape) == 3:
                npem = np.pad(pem, ((0, max_sentence - pem.shape[0]), (0, 0), (0, 0)), 'constant',
                              constant_values=(0, 0))
                npem = np.reshape(npem, (1, npem.shape[0], npem.shape[1], npem.shape[2]))
                clear_pem.append(npem)
            else:
                clear_pem.append(
                    np.zeros((1, max_sentence, self.paper_emb_list[0].shape[1], self.paper_emb_list[0].shape[2])))
        return np.concatenate(clear_pem, axis=0)

    def getEmbedding(self, words="", tokens=None, max_len=512):
        if not tokens:
            tokens = self.getToken(words)
        if len(tokens) <= 0:
            return None
        if len(tokens[0]) <= 1:
            return self.FastTextModel.embedding(tokens, max_row=max_len)
        else:
            max_sentence = 0
            self.paper_emb_list = []
            for i, paper in enumerate(tokens):
                paper_text = [[w] for w in paper]
                print(i + 1, '/', len(tokens))
                paper_emb = self.FastTextModel.embedding(paper_text, max_row=max_len)
                max_sentence = max(paper_emb.shape[0], max_sentence)
                self.paper_emb_list.append(paper_emb)
                print(self.paper_emb_list[-1].shape, max_sentence)
            return self.clean_sentences(max_sentence)