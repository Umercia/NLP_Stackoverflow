import joblib
import csv
import pandas as pd
import re
from app.model_data.NLP_functions import cleantext, supervised_predict
from app.model_data.NLP_functions import tag_pred, display_topics2


data_directory = "app/model_data/"   # "/home/Umercia/mysite/app/"

with open(data_directory+'top_t_162.csv', 'r') as f:
    reader = csv.reader(f)
    top_t = list(reader)[0]

# supervised
vectorizer_sup = joblib.load(data_directory+"tf_800_162.gz")
model_sup = joblib.load(data_directory+"logi_reg_tf.gz")

# semi_supervised
vectorizer_sem = joblib.load(data_directory+"tfidf_vectorizer_semis.gz")
model_sem = joblib.load(data_directory+"tag_pred_lda_tfidf-semi_supervised.gz")

# unsupervised
vectorizer_uns = joblib.load(data_directory+"tf_vectorizer_uns.gz")
model_uns = joblib.load(data_directory+"lda_tf_uns.gz")

def tagline(title, text, model_type="sup"):
    A = title + " " + text
    A = cleantext(A)
    if model_type == "sup":
        A = vectorizer_sup.fit_transform([A])
        voc = vectorizer_sup.get_feature_names()
        A = pd.DataFrame(A.A, columns=voc, index=[0])
        A = supervised_predict(model_sup, A, top_t)
        A = A.T
        mask = A.loc[:,0] == 1
        return list(A.loc[mask,:].index)

    elif model_type == "sem":
        A = vectorizer_sem.fit_transform([A])
        voc = vectorizer_sem.get_feature_names()
        A = pd.DataFrame(A.A, columns=voc, index=[0])
        A = tag_pred(model_sem, A, vectorizer_sem.get_feature_names(), top_t)
        return A[0]

    elif model_type == "uns":
        A = vectorizer_uns.fit_transform([A])
        voc = vectorizer_uns.get_feature_names()
        A = pd.DataFrame(A.A, columns=voc, index=[0])
        A = tag_pred(model_uns, A, vectorizer_uns.get_feature_names(), top_t)
        return A[0]
