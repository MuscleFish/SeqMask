from DataDeal import FastTextDeal
from SeqMask import tact_ids, tech_ids
from tensorflow.keras.models import load_model, Model

class TextDeal(object):
    def __init__(self, tact_model_path='./models/tactics_ar_mask_model',
                 tech_model_path='./models/techniques_ar_mask_model'):
        self.tact_model = load_model(tact_model_path)
        self.tech_model = load_model(tech_model_path)
        self.text_deal = FastTextDeal()

    def classify_text(self, text='', max_len=128):
        tokens = self.text_deal.getToken(text)
        embedding_text = self.text_deal.getEmbedding(text, tokens, max_len)
        # print(embedding_text.shape)
        tact_predict = self.tact_model.predict(embedding_text)
        tech_predict = self.tech_model.predict(embedding_text)
        results = {"text": text, "deal_text": tokens, "embedding": embedding_text,
                   "tactics": [], "techniques": []}
        if type(text) == str:
            results["total_tactics"] = {}
            results["total_techniques"] = {}
        for i in range(embedding_text.shape[0]):
            results["tactics"].append({})
            results["techniques"].append({})
            for ti, tid in enumerate(tact_ids):
                results["tactics"][-1][tid] = float(tact_predict[i, ti])
                if type(text) == str:
                    if tid not in results["total_tactics"].keys():
                        results["total_tactics"][tid] = 0
                    results["total_tactics"][tid] = max(float(tact_predict[i, ti]),
                                                        results["total_tactics"][tid])
            for ti, tid in enumerate(tech_ids):
                results["techniques"][-1][tid] = float(tech_predict[i, ti])
                if type(text) == str:
                    if tid not in results["total_techniques"].keys():
                        results["total_techniques"][tid] = 0
                    results["total_techniques"][tid] = max(float(tech_predict[i, ti]),
                                                           results["total_techniques"][tid])
        return results

class TextImportance(object):
    def __init__(self, model_path='./models/techniques_mp_mask_model', weights_layer_name='mask_scores_0'):
        self.text_deal = FastTextDeal()
        out_model = load_model(model_path)
        self.model = Model(out_model.input, out_model.get_layer(weights_layer_name).output)

    def score_text(self, text='', max_len=128):
        tokens = self.text_deal.getToken(text)
        embedding_text = self.text_deal.getEmbedding(text, tokens, max_len)
        scores = self.model.predict(embedding_text)
        tokens_score = []
        for i, t in enumerate(tokens):
            tokens_score.append([])
            for j, w in enumerate(t):
                tokens_score[-1].append({'text': w, 'score': scores[i, j],
                                         'percent': (scores[i, j]-scores[i, 0:len(t)].min()) /
                                                    (scores[i, 0:len(t)].max()-scores[i, 0:len(t)].min())})
        return tokens_score