"""
train_models.py  —  Trains fixed Logistic Regression + Random Forest pipelines
"""
import json, time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ── 1. LOAD & CLEAN DATA ─────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv("cardio_train.csv", sep=";")
df = df.drop(columns="id")
df["age"] = (df["age"] / 365.25).astype(int)

# Hard physiological bounds (removes impossible readings)
before = len(df)
df = df[df["ap_hi"].between(70, 250)]
df = df[df["ap_lo"].between(40, 200)]
df = df[df["ap_hi"] > df["ap_lo"]]
df = df[df["height"].between(100, 220)]
df = df[df["weight"].between(30, 200)]
df = df.drop_duplicates()
print(f"Rows after cleaning: {len(df)}  (removed {before - len(df)})")

# ── 2. FEATURES  (NO BMI — eliminates multicollinearity) ─────────
FEATURES = ["age","gender","height","weight","ap_hi","ap_lo",
            "cholesterol","gluc","smoke","alco","active"]

X = df[FEATURES]
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 3. EVALUATION HELPER ─────────────────────────────────────────
def evaluate(name, pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    y_pred    = pipe.predict(X_te)
    y_prob    = pipe.predict_proba(X_te)[:,1]
    y_tr_pred = pipe.predict(X_tr)
    acc  = accuracy_score(y_te, y_pred)*100
    prec = precision_score(y_te, y_pred)*100
    rec  = recall_score(y_te, y_pred)*100
    f1   = f1_score(y_te, y_pred)*100
    roc  = roc_auc_score(y_te, y_prob)*100
    tr_acc = accuracy_score(y_tr, y_tr_pred)*100
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="accuracy", n_jobs=-1)*100
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(f"  Accuracy : {acc:.2f}%  |  ROC-AUC: {roc:.2f}%")
    print(f"  Precision: {prec:.2f}%  |  Recall:  {rec:.2f}%  |  F1: {f1:.2f}%")
    print(f"  Train Acc: {tr_acc:.2f}%  (diff: {abs(tr_acc-acc):.2f}%)")
    print(f"  CV Mean  : {cv.mean():.2f}%  ±{cv.std():.2f}%")
    return {"name":name,"accuracy":round(acc,2),"precision":round(prec,2),"recall":round(rec,2),
            "f1":round(f1,2),"roc_auc":round(roc,2),"train_acc":round(tr_acc,2),
            "diff":round(abs(tr_acc-acc),2),"cv_mean":round(cv.mean(),2),"cv_std":round(cv.std(),2),
            "cv_folds":[round(s,2) for s in cv.tolist()],"features":FEATURES}

# ── 4. LOGISTIC REGRESSION (fixed) ──────────────────────────────
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
])
t0 = time.time()
lr_meta = evaluate("Logistic Regression (Fixed)", lr_pipe, X_train, y_train, X_test, y_test)
print(f"  Time: {time.time()-t0:.1f}s")
joblib.dump(lr_pipe, "cardiovascular_lr_pipeline_v2.pkl")
print("  Saved: cardiovascular_lr_pipeline_v2.pkl")

coefs = lr_pipe.named_steps["model"].coef_[0]
print("\n  Coefficient sanity check:")
for f,c in zip(FEATURES, coefs):
    print(f"    {f:12s}: {c:+.4f}  ({'↑ risk' if c>0 else '↓ risk'})")

# ── 5. RANDOM FOREST ────────────────────────────────────────────
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=15,
        min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1)),
])
t0 = time.time()
rf_meta = evaluate("Random Forest", rf_pipe, X_train, y_train, X_test, y_test)
print(f"  Time: {time.time()-t0:.1f}s")
joblib.dump(rf_pipe, "cardiovascular_rf_pipeline_v2.pkl")
print("  Saved: cardiovascular_rf_pipeline_v2.pkl")

# ── 6. SAVE METADATA ─────────────────────────────────────────────
with open("model_metadata.json","w") as f:
    json.dump({"logistic_regression":lr_meta,"random_forest":rf_meta}, f, indent=2)
print("\n  Saved: model_metadata.json")
print("\nAll done! ✅")
