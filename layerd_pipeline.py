import os
import random
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, brier_score_loss
)

np.random.seed(42)
random.seed(42)

def generate_synthetic_data(n_samples=3000):
    data = []
    emp_options = ["angajat cu contract permanent", "angajat sezonier", "somer"]
    tenure_options = ["de 4 ani", "de 1 an", "de 6 luni"]
    inc_options = ["am un venit net de 4500 lei", "am un venit de 1500 lei", "castig putin"]
    debt_options = ["si nu am datorii", "si am niste restante", "cu datorii la alte banci"]
    
    for _ in range(n_samples):
        e, t, i, d = random.choice(emp_options), random.choice(tenure_options), \
                     random.choice(inc_options), random.choice(debt_options)
        text = f"Sunt {e} {t}, {i} {d}."
        label = 0 if ("somer" in e or "restante" in d or "datorii" in d or "putin" in i) else 1
        data.append({"text": text, "label": label})
    return pd.DataFrame(data)

def load_german_credit(file_path="german.data"):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, sep=' ', header=None)
    verbalized = []
    for _, row in df.iterrows():
        label = 1 if row[20] == 1 else 0
        sum_d = "suma mica" if row[4] < 2000 else "suma medie" if row[4] < 5000 else "suma mare"
        emp_d = "<1" if row[6] in ['A71', 'A72'] else "1-4" if row[6] == 'A73' else ">7"
        text = f"Sunt angajat de {emp_d} ani si doresc un credit in valoare de o {sum_d}, pe {row[1]} luni."
        verbalized.append({"text": text, "label": label})
    return pd.DataFrame(verbalized[:200])


def get_layered_explanation(model, vectorizer, dt_model, text):
    x = vectorizer.transform([text])
    prob = model.predict_proba(x)[0, 1]
    
    feats = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    active = x.nonzero()[1]
    attrs = {feats[i]: x[0, i] * weights[i] for i in active}

    rule = export_text(dt_model, feature_names=list(feats), max_depth=1)
    
    return attrs, prob, rule

def run_faithfulness(model, vectorizer, text, attrs, k=3):
    orig_prob = model.predict_proba(vectorizer.transform([text]))[0, 1]
    sorted_words = sorted(attrs, key=lambda x: abs(attrs[x]), reverse=True)[:k]
    perturbed = text
    for w in sorted_words: perturbed = perturbed.replace(w, "")
    new_prob = model.predict_proba(vectorizer.transform([perturbed]))[0, 1]
    return abs(orig_prob - new_prob)

def run_robustness(model, vectorizer, text):
    orig_prob = model.predict_proba(vectorizer.transform([text]))[0, 1]
    noisy = "".join([c if random.random() > 0.05 else random.choice('abc') for c in text])
    noise_prob = model.predict_proba(vectorizer.transform([noisy]))[0, 1]
    return abs(orig_prob - noise_prob)

if _name_ == "_main_":
 
    df_s = generate_synthetic_data(3000)
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=5)
    X = vec.fit_transform(df_s['text'])
    model = LogisticRegression(C=1.0, solver='liblinear').fit(X, df_s['label'])
    dt = DecisionTreeClassifier(max_depth=3).fit(X, df_s['label'])
    
    y_proba = model.predict_proba(X)[:, 1]
    print(f"F1: {f1_score(df_s['label'], model.predict(X)):.2f}")
    print(f"Brier: {brier_score_loss(df_s['label'], y_proba):.4f}")

    df_g = load_german_credit("german.data")
    if df_g is not None:
        X_g = vec.transform(df_g['text'])
        
        sample = df_g.iloc[0]['text']
        attrs, p, rule = get_layered_explanation(model, vec, dt, sample)
        faith = run_faithfulness(model, vec, sample, attrs, k=3)
        print(f"\nExample: {sample}")
        print(f"Faithfulness Delta (k=3): {faith:.4f}")