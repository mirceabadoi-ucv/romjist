import os
import random
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def generate_synthetic_data(n_samples: int = 3000) -> pd.DataFrame:
    emp_options = ["angajat cu contract permanent", "angajat sezonier", "somer"]
    emp_weights = [0.55, 0.25, 0.20]

    tenure_eligible   = ["de 4 ani", "de 7 ani", "de 2 ani"]
    tenure_ineligible = ["de 6 luni", "de 3 luni"]

    inc_high   = ["am un venit net de 4500 lei", "am un venit net de 6000 lei"]
    inc_low    = ["am un venit de 1500 lei", "castig putin"]
    inc_medium = ["am un venit net de 2800 lei"]

    debt_none    = ["si nu am datorii", "fara datorii"]
    debt_arrears = ["si am niste restante", "cu datorii la alte banci"]

    data = []
    for _ in range(n_samples):
        emp = random.choices(emp_options, weights=emp_weights, k=1)[0]

        if emp == "somer":
            tenure = random.choice(tenure_ineligible)
        else:
            tenure = random.choice(tenure_eligible + tenure_ineligible)

        if emp == "angajat sezonier":
            inc = random.choice(inc_medium + inc_low)
        elif emp == "somer":
            inc = random.choice(inc_low)
        else:
            inc = random.choice(inc_high + inc_medium + inc_low)

        debt = random.choice(debt_none + debt_arrears)

        text = f"Sunt {emp} {tenure}, {inc} {debt}."

        ineligible = (
            emp == "somer"
            or any(d in debt for d in ["restante", "datorii la"])
            or inc in inc_low
            or tenure in tenure_ineligible
        )
        label = 0 if ineligible else 1
        data.append({"text": text, "label": label})

    return pd.DataFrame(data)


def load_german_credit(file_path: str = "german.data",
                       n_samples: int = 200) -> pd.DataFrame | None:
    if not os.path.exists(file_path):
        print(f"[WARNING] {file_path} not found — skipping semi-real evaluation.")
        return None

    df = pd.read_csv(file_path, sep=" ", header=None)

    def tenure_band(code: str) -> str:
        mapping = {
            "A71": "<1",
            "A72": "<1",
            "A73": "1-4",
            "A74": "4-7",
            "A75": ">7",
        }
        return mapping.get(code, "1-4")

    def amount_band(amount: float) -> str:
        if amount < 2000:
            return "suma mica"
        elif amount < 5000:
            return "suma medie"
        else:
            return "suma mare"

    verbalized = []
    for _, row in df.iterrows():
        label = 1 if row[20] == 1 else 0
        t = tenure_band(str(row[6]))
        a = amount_band(float(row[4]))
        d = int(row[1])
        text = (
            f"Sunt angajat de {t} ani si doresc un credit "
            f"in valoare de o {a}, pe aproximativ {d} de luni."
        )
        verbalized.append({"text": text, "label": label})

    return pd.DataFrame(verbalized[:n_samples])


FACTOR_KEYWORDS = {
    "employment_stability": ["permanent", "sezonier", "somer", "angajat"],
    "income_level":         ["venit", "castig", "lei"],
    "debt_indicators":      ["datorii", "restante", "fara"],
    "tenure":               ["ani", "luni", "4-7", "1-4", "<1", ">7"],
}

RATIONALE_TEMPLATES = {
    "employment_stability": {
        "positive": "Tipul de angajare indica stabilitate financiara.",
        "negative": "Tipul de angajare indica instabilitate sau lipsa venitului.",
    },
    "income_level": {
        "positive": "Nivelul venitului declarat sustine eligibilitatea.",
        "negative": "Nivelul venitului declarat este insuficient.",
    },
    "debt_indicators": {
        "positive": "Nu exista indicatori de datorii sau restante.",
        "negative": "Exista indicatori de datorii sau restante.",
    },
    "tenure": {
        "positive": "Durata angajarii/creditului este favorabila.",
        "negative": "Durata angajarii/creditului este nefavorabila.",
    },
}


def get_layered_explanation(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    dt_model: DecisionTreeClassifier,
    text: str,
) -> tuple[dict, float, str, str]:
    x = vectorizer.transform([text])
    prob = model.predict_proba(x)[0, 1]

    feats   = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    active  = x.nonzero()[1]
    attrs   = {feats[i]: float(x[0, i] * weights[i]) for i in active}

    factor_scores: dict[str, float] = {f: 0.0 for f in FACTOR_KEYWORDS}
    for token, phi in attrs.items():
        for factor, keywords in FACTOR_KEYWORDS.items():
            if any(kw in token for kw in keywords):
                factor_scores[factor] += phi

    rationale_parts = []
    for factor, score in sorted(factor_scores.items(),
                                key=lambda x: abs(x[1]), reverse=True)[:2]:
        direction = "positive" if score >= 0 else "negative"
        rationale_parts.append(RATIONALE_TEMPLATES[factor][direction])
    layer2 = " ".join(rationale_parts) if rationale_parts else "Informatii insuficiente."

    rule = export_text(dt_model, feature_names=list(feats), max_depth=2)

    return attrs, prob, layer2, rule


def lime_surrogate(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    text: str,
    n_perturbations: int = 200,
) -> dict:
    x0     = vectorizer.transform([text]).toarray()[0]
    tokens = text.split()
    n_tok  = len(tokens)

    X_perturbed, y_perturbed, weights_local = [], [], []
    for _ in range(n_perturbations):
        mask           = np.random.randint(0, 2, n_tok)
        perturbed_text = " ".join(t for t, m in zip(tokens, mask) if m)
        x_p            = vectorizer.transform([perturbed_text]).toarray()[0]
        y_p            = model.predict_proba(vectorizer.transform([perturbed_text]))[0, 1]
        dist           = np.sqrt(np.sum((x0 - x_p) ** 2))
        w              = np.exp(-dist)
        X_perturbed.append(x_p)
        y_perturbed.append(y_p)
        weights_local.append(w)

    surrogate = Ridge(alpha=1.0)
    surrogate.fit(np.array(X_perturbed), np.array(y_perturbed),
                  sample_weight=np.array(weights_local))

    feats = vectorizer.get_feature_names_out()
    return dict(zip(feats, surrogate.coef_))


def run_faithfulness(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    text: str,
    attrs: dict,
    k: int = 3,
) -> float:
    orig_prob    = model.predict_proba(vectorizer.transform([text]))[0, 1]
    top_k_tokens = sorted(attrs, key=lambda t: abs(attrs[t]), reverse=True)[:k]
    perturbed    = text
    for token in top_k_tokens:
        perturbed = re.sub(r"\b" + re.escape(token) + r"\b", "", perturbed)
    perturbed = re.sub(r"\s+", " ", perturbed).strip()
    new_prob  = model.predict_proba(vectorizer.transform([perturbed]))[0, 1]
    return float(abs(orig_prob - new_prob))


def remove_diacritics(text: str) -> str:
    replacements = {"ă": "a", "â": "a", "î": "i", "ș": "s", "ț": "t",
                    "Ă": "A", "Â": "A", "Î": "I", "Ș": "S", "Ț": "T"}
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text


def add_char_noise(text: str, rate: float = 0.05) -> str:
    return "".join(
        c if random.random() > rate else random.choice("abcdefghijklmnopqrstuvwxyz")
        for c in text
    )


def mask_numerics(text: str) -> str:
    return re.sub(r"\b\d+\b", "NUM", text)


def run_robustness(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    text: str,
) -> dict:
    orig = model.predict_proba(vectorizer.transform([text]))[0, 1]
    return {
        "diacritic_removal": abs(orig - model.predict_proba(
            vectorizer.transform([remove_diacritics(text)]))[0, 1]),
        "char_noise_5pct":   abs(orig - model.predict_proba(
            vectorizer.transform([add_char_noise(text)]))[0, 1]),
        "numeric_masking":   abs(orig - model.predict_proba(
            vectorizer.transform([mask_numerics(text)]))[0, 1]),
    }


def evaluate_model(model, X, y) -> dict:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy":  accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred, zero_division=0),
        "f1":        f1_score(y, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y, y_proba),
        "brier":     brier_score_loss(y, y_proba),
    }


def convergence_pct(n_iter: int, n_max: int = 500) -> float:
    return round(n_iter / n_max * 100, 1)


if __name__ == "__main__":

    print("=" * 60)
    print("SYNTHETIC DATASET")
    print("=" * 60)

    df_s = generate_synthetic_data(3000)
    X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(
        df_s["text"], df_s["label"],
        test_size=0.20, stratify=df_s["label"], random_state=SEED
    )

    vec_s = TfidfVectorizer(ngram_range=(1, 2), min_df=5)
    Xtr_s = vec_s.fit_transform(X_s_tr)
    Xte_s = vec_s.transform(X_s_te)

    lr_s = LogisticRegression(
        C=1.0, solver="liblinear", class_weight="balanced",
        max_iter=500, random_state=SEED
    ).fit(Xtr_s, y_s_tr)

    dt_s = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr_s, y_s_tr)

    metrics_s = evaluate_model(lr_s, Xte_s, y_s_te)
    print("Metrics (test set):")
    for k, v in metrics_s.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"  Conv.%      : {convergence_pct(lr_s.n_iter_[0])}")

    sample_s = X_s_te.iloc[0]
    attrs_s, prob_s, rationale_s, rule_s = get_layered_explanation(
        lr_s, vec_s, dt_s, sample_s
    )
    dp3_s = run_faithfulness(lr_s, vec_s, sample_s, attrs_s, k=3)
    dp5_s = run_faithfulness(lr_s, vec_s, sample_s, attrs_s, k=5)
    print(f"\nFaithfulness  Δp3={dp3_s:.4f}  Δp5={dp5_s:.4f}")

    rob_s = run_robustness(lr_s, vec_s, sample_s)
    print("Robustness:", {k: f"{v:.4f}" for k, v in rob_s.items()})

    print("\nLayer 2 rationale:", rationale_s)
    print("\nLayer 3 rule trace (truncated):\n", rule_s[:300])

    print("\n" + "=" * 60)
    print("SEMI-REAL DATASET (German Credit)")
    print("=" * 60)

    df_g = load_german_credit("german.data", n_samples=200)

    if df_g is not None:
        X_g_tr, X_g_te, y_g_tr, y_g_te = train_test_split(
            df_g["text"], df_g["label"],
            test_size=0.20, stratify=df_g["label"], random_state=SEED
        )

        vec_g = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        Xtr_g = vec_g.fit_transform(X_g_tr)
        Xte_g = vec_g.transform(X_g_te)

        lr_g = LogisticRegression(
            C=1.0, solver="liblinear",
            max_iter=500, random_state=SEED
        ).fit(Xtr_g, y_g_tr)

        dt_g = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr_g, y_g_tr)

        metrics_g = evaluate_model(lr_g, Xte_g, y_g_te)
        print("Metrics (test set):")
        for k, v in metrics_g.items():
            print(f"  {k:12s}: {v:.4f}")
        print(f"  Conv.%      : {convergence_pct(lr_g.n_iter_[0])}")

        sample_g = X_g_te.iloc[0]
        attrs_g, prob_g, rationale_g, rule_g = get_layered_explanation(
            lr_g, vec_g, dt_g, sample_g
        )
        dp3_g = run_faithfulness(lr_g, vec_g, sample_g, attrs_g, k=3)
        dp5_g = run_faithfulness(lr_g, vec_g, sample_g, attrs_g, k=5)
        print(f"\nFaithfulness  Δp3={dp3_g:.4f}  Δp5={dp5_g:.4f}")

        rob_g = run_robustness(lr_g, vec_g, sample_g)
        print("Robustness:", {k: f"{v:.4f}" for k, v in rob_g.items()})

        surrogate_coeffs = lime_surrogate(lr_g, vec_g, sample_g)
        top_surrogate = sorted(surrogate_coeffs.items(),
                               key=lambda x: abs(x[1]), reverse=True)[:5]
        top_shap = sorted(attrs_g.items(),
                          key=lambda x: abs(x[1]), reverse=True)[:5]
        print("\nTop-5 LR additive (Layer 1):", [t for t, _ in top_shap])
        print("Top-5 LIME surrogate        :", [t for t, _ in top_surrogate])

        print("\nLayer 2 rationale:", rationale_g)
