import os
import random
import re
import time
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

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup
    from captum.attr import LayerIntegratedGradients
    ROBERT_AVAILABLE = True
except ImportError:
    ROBERT_AVAILABLE = False

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
if ROBERT_AVAILABLE:
    torch.manual_seed(SEED)


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


ROBERT_CHECKPOINT = "readerbench/ro-bert"
ROBERT_MAX_LEN    = 128
ROBERT_EPOCHS     = 5
ROBERT_LR         = 2e-5
ROBERT_BATCH      = 16
ROBERT_WARMUP     = 0.10


class CreditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=ROBERT_MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]


def finetune_robert(
    X_train, y_train, X_test, y_test,
    class_weight=None,
) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(ROBERT_CHECKPOINT)
    model     = AutoModelForSequenceClassification.from_pretrained(
        ROBERT_CHECKPOINT, num_labels=2
    ).to(device)

    train_ds = CreditDataset(X_train, y_train, tokenizer)
    test_ds  = CreditDataset(X_test,  y_test,  tokenizer)
    train_dl = DataLoader(train_ds, batch_size=ROBERT_BATCH, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=ROBERT_BATCH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=ROBERT_LR)
    total_steps  = len(train_dl) * ROBERT_EPOCHS
    warmup_steps = int(total_steps * ROBERT_WARMUP)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    if class_weight is not None:
        counts   = np.bincount(list(y_train))
        weights  = torch.tensor(
            len(y_train) / (2.0 * counts), dtype=torch.float
        ).to(device)
        loss_fn  = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn  = torch.nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    model.train()
    for _ in range(ROBERT_EPOCHS):
        for batch, labels in train_dl:
            batch   = {k: v.to(device) for k, v in batch.items()}
            labels  = labels.to(device)
            outputs = model(**batch)
            loss    = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    train_time = time.perf_counter() - t0

    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    t1 = time.perf_counter()
    with torch.no_grad():
        for batch, labels in test_dl:
            batch   = {k: v.to(device) for k, v in batch.items()}
            logits  = model(**batch).logits
            probs   = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds   = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    infer_time = (time.perf_counter() - t1) / len(all_labels) * 1000

    metrics = {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc":   roc_auc_score(all_labels, all_probs),
        "brier":     brier_score_loss(all_labels, all_probs),
    }
    return model, tokenizer, metrics, train_time, infer_time


def integrated_gradients_attributions(
    model, tokenizer, text: str, n_steps: int = 50
) -> list[tuple[str, float]]:
    device = next(model.parameters()).device
    model.eval()

    encoding   = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=ROBERT_MAX_LEN
    )
    input_ids  = encoding["input_ids"].to(device)
    attn_mask  = encoding["attention_mask"].to(device)
    token_ids  = input_ids[0].tolist()
    tokens     = tokenizer.convert_ids_to_tokens(token_ids)

    def forward_fn(input_embeds):
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask
        )
        return torch.softmax(outputs.logits, dim=-1)[:, 1]

    embed_layer = model.roberta.embeddings if hasattr(model, "roberta") \
                  else model.bert.embeddings

    lig = LayerIntegratedGradients(forward_fn, embed_layer)

    input_embeds   = embed_layer.word_embeddings(input_ids)
    baseline_ids   = torch.zeros_like(input_ids)
    baseline_embed = embed_layer.word_embeddings(baseline_ids)

    attributions, _ = lig.attribute(
        input_embeds,
        baselines=baseline_embed,
        n_steps=n_steps,
        return_convergence_delta=True,
        additional_forward_args=()
    )

    attr_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    return list(zip(tokens, attr_scores.tolist()))


def compare_attributions(lr_attrs: dict, ig_attrs: list[tuple[str, float]], top_k: int = 5):
    top_lr = [t for t, _ in sorted(lr_attrs.items(),
                                    key=lambda x: abs(x[1]), reverse=True)[:top_k]]
    clean_ig = [(t.lstrip("Ġ▁").lower(), s) for t, s in ig_attrs
                if t not in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>")]
    top_ig = [t for t, _ in sorted(clean_ig,
                                    key=lambda x: abs(x[1]), reverse=True)[:top_k]]
    overlap = len(set(top_lr) & set(top_ig))
    print(f"  Top-{top_k} LR attributions : {top_lr}")
    print(f"  Top-{top_k} IG attributions : {top_ig}")
    print(f"  Overlap ({overlap}/{top_k})")


def tune_hyperparameters(X_train, y_train, class_weight=None) -> tuple[float, int]:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=SEED
    )
    C_grid    = [0.25, 0.5, 1.0, 2.0, 4.0]
    best_C, best_score = 1.0, -1.0
    for C in C_grid:
        m = LogisticRegression(
            C=C, solver="liblinear", class_weight=class_weight,
            max_iter=500, random_state=SEED
        ).fit(X_tr, y_tr)
        f1  = f1_score(y_val, m.predict(X_val), zero_division=0)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        score = 0.5 * f1 + 0.5 * auc
        if score > best_score:
            best_score, best_C = score, C
    return best_C, best_score


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

    best_C_s, _ = tune_hyperparameters(Xtr_s, y_s_tr, class_weight="balanced")

    t0 = time.perf_counter()
    lr_s = LogisticRegression(
        C=best_C_s, solver="liblinear", class_weight="balanced",
        max_iter=500, random_state=SEED
    ).fit(Xtr_s, y_s_tr)
    train_time_s = time.perf_counter() - t0

    dt_s = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr_s, y_s_tr)

    metrics_s = evaluate_model(lr_s, Xte_s, y_s_te)
    print("Metrics (test set):")
    for k, v in metrics_s.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"  Conv.%      : {convergence_pct(lr_s.n_iter_[0])}")
    print(f"  Best C      : {best_C_s}")
    print(f"  Train time  : {train_time_s:.4f}s")

    t1 = time.perf_counter()
    _ = lr_s.predict_proba(Xte_s)
    infer_time_s = (time.perf_counter() - t1) / len(y_s_te) * 1000
    print(f"  Infer/sample: {infer_time_s:.4f}ms")

    feats_s  = vec_s.get_feature_names_out()
    coefs_s  = lr_s.coef_[0]
    top_pos  = sorted(zip(feats_s, coefs_s), key=lambda x: x[1], reverse=True)[:5]
    top_neg  = sorted(zip(feats_s, coefs_s), key=lambda x: x[1])[:5]
    print("\nTop positive features (eligibil):", [(f, round(c, 4)) for f, c in top_pos])
    print("Top negative features (neeligibil):", [(f, round(c, 4)) for f, c in top_neg])

    dp3_list, dp5_list = [], []
    for txt in X_s_te:
        a, _, _, _ = get_layered_explanation(lr_s, vec_s, dt_s, txt)
        dp3_list.append(run_faithfulness(lr_s, vec_s, txt, a, k=3))
        dp5_list.append(run_faithfulness(lr_s, vec_s, txt, a, k=5))
    print(f"\nFaithfulness (mean over test set)  Δp3={np.mean(dp3_list):.4f}  Δp5={np.mean(dp5_list):.4f}")

    rob_vals = [run_robustness(lr_s, vec_s, txt) for txt in X_s_te]
    rob_mean_s = {k: np.mean([r[k] for r in rob_vals]) for k in rob_vals[0]}
    print("Robustness (mean):", {k: f"{v:.4f}" for k, v in rob_mean_s.items()})

    surrogate_coeffs_s = lime_surrogate(lr_s, vec_s, X_s_te.iloc[0])
    top_surrogate_s = sorted(surrogate_coeffs_s.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_shap_s = sorted(
        {feats_s[i]: float(vec_s.transform([X_s_te.iloc[0]])[0, i] * coefs_s[i])
         for i in vec_s.transform([X_s_te.iloc[0]]).nonzero()[1]}.items(),
        key=lambda x: abs(x[1]), reverse=True
    )[:5]
    print("\nTop-5 LR additive (Layer 1):", [t for t, _ in top_shap_s])
    print("Top-5 LIME surrogate        :", [t for t, _ in top_surrogate_s])

    sample_s = X_s_te.iloc[0]
    attrs_s, prob_s, rationale_s, rule_s = get_layered_explanation(lr_s, vec_s, dt_s, sample_s)
    print("\nLayer 2 rationale:", rationale_s)
    print("\nLayer 3 rule trace (truncated):\n", rule_s[:300])

    if ROBERT_AVAILABLE:
        print("\n--- RoBERT baseline (synthetic) ---")
        try:
            rb_model_s, rb_tok_s, rb_metrics_s, rb_train_s, rb_infer_s = finetune_robert(
                X_s_tr, y_s_tr, X_s_te, y_s_te, class_weight="balanced"
            )
            print("RoBERT metrics (test set):")
            for k, v in rb_metrics_s.items():
                print(f"  {k:12s}: {v:.4f}")
            print(f"  Train time  : {rb_train_s:.2f}s")
            print(f"  Infer/sample: {rb_infer_s:.4f}ms")
            ig_attrs_s = integrated_gradients_attributions(rb_model_s, rb_tok_s, sample_s)
            print("\nCross-approach attribution comparison (synthetic sample):")
            compare_attributions(attrs_s, ig_attrs_s)
        except OSError as e:
            print(f"  [SKIPPED] Could not load {ROBERT_CHECKPOINT}: {e}")

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

        best_C_g, _ = tune_hyperparameters(Xtr_g, y_g_tr, class_weight=None)

        t0 = time.perf_counter()
        lr_g = LogisticRegression(
            C=best_C_g, solver="liblinear",
            max_iter=500, random_state=SEED
        ).fit(Xtr_g, y_g_tr)
        train_time_g = time.perf_counter() - t0

        dt_g = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr_g, y_g_tr)

        metrics_g = evaluate_model(lr_g, Xte_g, y_g_te)
        print("Metrics (test set):")
        for k, v in metrics_g.items():
            print(f"  {k:12s}: {v:.4f}")
        print(f"  Conv.%      : {convergence_pct(lr_g.n_iter_[0])}")
        print(f"  Best C      : {best_C_g}")
        print(f"  Train time  : {train_time_g:.4f}s")

        t1 = time.perf_counter()
        _ = lr_g.predict_proba(Xte_g)
        infer_time_g = (time.perf_counter() - t1) / len(y_g_te) * 1000
        print(f"  Infer/sample: {infer_time_g:.4f}ms")

        dp3_list_g, dp5_list_g = [], []
        for txt in X_g_te:
            a, _, _, _ = get_layered_explanation(lr_g, vec_g, dt_g, txt)
            dp3_list_g.append(run_faithfulness(lr_g, vec_g, txt, a, k=3))
            dp5_list_g.append(run_faithfulness(lr_g, vec_g, txt, a, k=5))
        print(f"\nFaithfulness (mean over test set)  Δp3={np.mean(dp3_list_g):.4f}  Δp5={np.mean(dp5_list_g):.4f}")

        rob_vals_g = [run_robustness(lr_g, vec_g, txt) for txt in X_g_te]
        rob_mean_g = {k: np.mean([r[k] for r in rob_vals_g]) for k in rob_vals_g[0]}
        print("Robustness (mean):", {k: f"{v:.4f}" for k, v in rob_mean_g.items()})

        sample_g = X_g_te.iloc[0]
        attrs_g, prob_g, rationale_g, rule_g = get_layered_explanation(
            lr_g, vec_g, dt_g, sample_g
        )

        surrogate_coeffs = lime_surrogate(lr_g, vec_g, sample_g)
        top_surrogate = sorted(surrogate_coeffs.items(),
                               key=lambda x: abs(x[1]), reverse=True)[:5]
        top_shap = sorted(attrs_g.items(),
                          key=lambda x: abs(x[1]), reverse=True)[:5]
        print("\nTop-5 LR additive (Layer 1):", [t for t, _ in top_shap])
        print("Top-5 LIME surrogate        :", [t for t, _ in top_surrogate])

        print("\nLayer 2 rationale:", rationale_g)

        if ROBERT_AVAILABLE:
            print("\n--- RoBERT baseline (semi-real) ---")
            try:
                rb_model_g, rb_tok_g, rb_metrics_g, rb_train_g, rb_infer_g = finetune_robert(
                    X_g_tr, y_g_tr, X_g_te, y_g_te, class_weight=None
                )
                print("RoBERT metrics (test set):")
                for k, v in rb_metrics_g.items():
                    print(f"  {k:12s}: {v:.4f}")
                print(f"  Train time  : {rb_train_g:.2f}s")
                print(f"  Infer/sample: {rb_infer_g:.4f}ms")
                ig_attrs_g = integrated_gradients_attributions(rb_model_g, rb_tok_g, sample_g)
                print("\nCross-approach attribution comparison (semi-real sample):")
                compare_attributions(attrs_g, ig_attrs_g)
            except OSError as e:
                print(f"  [SKIPPED] Could not load {ROBERT_CHECKPOINT}: {e}")
