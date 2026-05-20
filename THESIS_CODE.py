#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline: Predicting Secondary Psychopathy from Childhood Trauma
# 
# ---
# This notebook contains the complete pipeline:
# - **Part 0** — Imports
# - **Part 1** — Data loading & cleaning
# - **Part 2** — Scale scoring (LSRP & CTQ-SF)
# - **Part 3** — Clinical severity encodings (descriptive only)
# - **Part 4** — Range validation & save
# - **Part 5** — Feature selection
# - **Part 6** — Target distribution & binning
# - **Part 7** — Train/test split
# - **Part 8** — Feature scaling
# - **Part 9** — Random oversampling (class imbalance)
# - **Part 10** — Regression pipeline (not reported)
# - **Part 11** — Classification pipeline
# - **Part 12** — Classification test set evaluation
# - **Part 13** — Confusion matrices
# - **Part 14** — ROC curves
# - **Part 15** — Feature importance
# - **Part 16** — SHAP explainability

# ## Part 0 — Imports

# In[3]:


get_ipython().system('pip install shap')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    accuracy_score, RocCurveDisplay
)

# Regression models
from sklearn.linear_model import LinearRegression, ElasticNet

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC

import shap
shap.initjs()


# ## Part 1 — Data Loading & Cleaning

# In[5]:


cols_to_keep = [
    "participant",
    "CTQ.SF_1",  "CTQ.SF_2",  "CTQ.SF_3",  "CTQ.SF_4",  "CTQ.SF_5",
    "CTQ.SF_6",  "CTQ.SF_7",  "CTQ.SF_8",  "CTQ.SF_9",  "CTQ.SF_10",
    "CTQ.SF_11", "CTQ.SF_12", "CTQ.SF_13", "CTQ.SF_14", "CTQ.SF_15",
    "CTQ.SF_16", "CTQ.SF_17", "CTQ.SF_18", "CTQ.SF_19", "CTQ.SF_20",
    "CTQ.SF_21", "CTQ.SF_22", "CTQ.SF_23", "CTQ.SF_24", "CTQ.SF_25",
    "CTQ.SF_26", "CTQ.SF_27", "CTQ.SF_28",
    "LSRP_1",  "LSRP_2",  "LSRP_3",  "LSRP_4",  "LSRP_5",
    "LSRP_6",  "LSRP_7",  "LSRP_8",  "LSRP_9",  "LSRP_10",
    "LSRP_11", "LSRP_12", "LSRP_13", "LSRP_14", "LSRP_15",
    "LSRP_16", "LSRP_18", "LSRP_19", "LSRP_20", "LSRP_21",
    "LSRP_22", "LSRP_23", "LSRP_24", "LSRP_25", "LSRP_26", "LSRP_27"
]

# Participants removed due to out-of-range item responses (where found manually in the dataset)
participants_to_remove = ["150", "141", "123", "99", "79"] 

df = pd.read_csv("df_wide.csv", usecols=cols_to_keep)
df = df[~df["participant"].astype(str).isin(participants_to_remove)]

# Convert all item columns to numeric
cols_to_convert = [c for c in cols_to_keep if c != "participant"]
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")

print(f"Sample size after exclusions: N = {len(df)}")
df.head()


# ## Part 2 — Scale Scoring (LSRP & CTQ-SF)

# ### 2a — LSRP Secondary Psychopathy

# In[7]:


# Reverse score items 20 and 24 (original scale: 1–5)
reverse_items_LSRP = ["LSRP_20", "LSRP_24"]
df[reverse_items_LSRP] = 5 - df[reverse_items_LSRP]

# Sum secondary psychopathy subscale (10 items, range 10–40)
secondary_items = [
    "LSRP_18", "LSRP_19", "LSRP_20", "LSRP_21", "LSRP_22",
    "LSRP_23", "LSRP_24", "LSRP_25", "LSRP_26", "LSRP_27"
]
df["secondary_psychopathy"] = df[secondary_items].sum(axis=1)

print(f"Secondary psychopathy: min={df['secondary_psychopathy'].min():.0f}, "
      f"max={df['secondary_psychopathy'].max():.0f}, "
      f"mean={df['secondary_psychopathy'].mean():.2f}, "
      f"std={df['secondary_psychopathy'].std():.2f}")


# ### 2b — CTQ-SF Subscales

# In[8]:


# Reverse score aka protective items (scale: 1–5)
reverse_items_CTQ = [
    "CTQ.SF_2",  "CTQ.SF_5",  "CTQ.SF_7",  "CTQ.SF_10", "CTQ.SF_13",
    "CTQ.SF_16", "CTQ.SF_19", "CTQ.SF_22", "CTQ.SF_26", "CTQ.SF_28"
]
df[reverse_items_CTQ] = 6 - df[reverse_items_CTQ]

# Subscale sums
df["ctq_emotional_abuse"]   = df[["CTQ.SF_3",  "CTQ.SF_8",  "CTQ.SF_14", "CTQ.SF_18", "CTQ.SF_25"]].sum(axis=1)
df["ctq_physical_abuse"]    = df[["CTQ.SF_9",  "CTQ.SF_11", "CTQ.SF_12", "CTQ.SF_15", "CTQ.SF_17"]].sum(axis=1)
df["ctq_sexual_abuse"]      = df[["CTQ.SF_20", "CTQ.SF_21", "CTQ.SF_23", "CTQ.SF_24", "CTQ.SF_27"]].sum(axis=1)
df["ctq_emotional_neglect"] = df[["CTQ.SF_5",  "CTQ.SF_7",  "CTQ.SF_13", "CTQ.SF_19", "CTQ.SF_28"]].sum(axis=1)
df["ctq_physical_neglect"]  = df[["CTQ.SF_1",  "CTQ.SF_2",  "CTQ.SF_4",  "CTQ.SF_6",
                                    "CTQ.SF_10", "CTQ.SF_16", "CTQ.SF_22", "CTQ.SF_26"]].sum(axis=1)

# CTQ total score
ctq_items = [f"CTQ.SF_{i}" for i in range(1, 29)]
df["ctq_total"] = df[ctq_items].sum(axis=1)

subscale_cols = [
    "ctq_emotional_abuse", "ctq_physical_abuse", "ctq_sexual_abuse",
    "ctq_emotional_neglect", "ctq_physical_neglect"
]

print("CTQ subscale descriptives:")
df[subscale_cols].agg(["min", "max", "mean", "std"]).round(2)


# ## Part 3 — Clinical Severity Encodings
# **For sample characterisation only — NOT used as model input features.**  
# Cutoffs from: Bernstein & Fink (1998). *Childhood Trauma Questionnaire: A Retrospective Self-Report.* The Psychological Corporation.  
# Categories: 0 = none/minimal, 1 = low, 2 = moderate, 3 = severe

# In[9]:


cutoffs = {
    "ctq_emotional_abuse":   [8,  12, 15],   # 5-item, max 25
    "ctq_physical_abuse":    [7,   9, 12],   # 5-item, max 25
    "ctq_sexual_abuse":      [5,   7, 12],   # 5-item, max 25
    "ctq_emotional_neglect": [9,  14, 17],   # 5-item, max 25
    "ctq_physical_neglect":  [14, 18, 26],   # 8-item, thresholds scaled x(8/5)
}

def encode_severity(score, thresholds):
    if score <= thresholds[0]:   return 0  # none/minimal
    elif score <= thresholds[1]: return 1  # low
    elif score <= thresholds[2]: return 2  # moderate
    else:                        return 3  # severe

for subscale, thresholds in cutoffs.items():
    df[f"{subscale}_severity"] = df[subscale].apply(
        lambda x: encode_severity(x, thresholds)
    )

# Report severity distributions for sample characterisation
print("Severity category distributions (0=none/minimal, 1=low, 2=moderate, 3=severe):\n")
severity_cols = [f"{s}_severity" for s in cutoffs.keys()]
labels = ["None/minimal", "Low", "Moderate", "Severe"]

for col in severity_cols:
    counts = df[col].value_counts().sort_index()
    pcts   = (counts / len(df) * 100).round(1)
    print(f"{col}:")
    for cat in range(4):
        n = counts.get(cat, 0)
        p = pcts.get(cat, 0.0)
        print(f"  {labels[cat]}: n={n} ({p}%)")
    print()

print("Note: Severity encodings are NOT included in model feature set.")


# ## Part 4 — Range Validation & Save Cleaned Data

# In[10]:


#sanity check of data
expected_min = {
    "ctq_emotional_abuse": 5,  "ctq_physical_abuse": 5,
    "ctq_sexual_abuse": 5,     "ctq_emotional_neglect": 5,
    "ctq_physical_neglect": 8
}
expected_max = {
    "ctq_emotional_abuse": 25, "ctq_physical_abuse": 25,
    "ctq_sexual_abuse": 25,    "ctq_emotional_neglect": 25,
    "ctq_physical_neglect": 40
}

print("Range validation:")
all_ok = True
for col in subscale_cols:
    lo = df[col].min(); hi = df[col].max()
    ok = lo >= expected_min[col] and hi <= expected_max[col]
    status = "OK" if ok else "PROBLEM"
    print(f"  {col:<30} min={lo:.0f}  max={hi:.0f}  [{status}]")
    if not ok: all_ok = False

if all_ok:
    print("\nAll subscales within valid range.")

df.to_csv("data_subset.csv", index=False)
print("Cleaned dataset saved to data_subset.csv")


# ## Part 5 — Feature Selection
# Five continuous CTQ subscale scores are used as input features.  

# In[11]:


feature_cols = [
    "ctq_emotional_abuse",
    "ctq_physical_abuse",
    "ctq_sexual_abuse",
    "ctq_emotional_neglect",
    "ctq_physical_neglect",
]

X = df[feature_cols].copy()
y = df["secondary_psychopathy"].copy()

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Feature matrix shape: {X.shape}")
print(f"Missing values in target: {y.isna().sum()}")

print("\nCorrelations with secondary psychopathy:")
print(X.corrwith(y).round(3))


# ## Part 6 — Target Distribution & Binning

# ### 6a — Initial 4-category exploration

# In[12]:


print("Secondary psychopathy score distribution:")
print(y.describe().round(2))

def bin_4cat(score):
    if score <= 22:   return "Low (<=22)"
    elif score <= 26: return "Medium (23-26)"
    elif score <= 30: return "High (27-30)"
    else:             return "Very High (>30)"

print("\nInitial 4-category distribution:")
print(y.apply(bin_4cat).value_counts().sort_index())
print("\nNote: Very High category has only 1 case — not viable for classification.")


# ### 6b — Final binary target (Low vs High+)

# In[13]:


# Binary target: Low (<=22) = 0, High+ (>22) = 1
# Medium (23-26) and High+ (27+) merged due to insufficient cases in upper categories
y_class = (y > 22).astype(int)

counts = y_class.value_counts().sort_index()
print("Binary class distribution:")
print(f"  Low   (0, score <=22): {counts[0]:>3}  ({counts[0]/len(y_class)*100:.1f}%)")
print(f"  High+ (1, score  >22): {counts[1]:>3}  ({counts[1]/len(y_class)*100:.1f}%)")


# ### 6c — Visualise target distribution

# In[14]:


fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].hist(y, bins=20, color="#378ADD", edgecolor="white", linewidth=0.5)
axes[0].axvline(22, color="red", linestyle="--", linewidth=2.0, label="Cut-off (22)")
axes[0].set_xlabel("Secondary psychopathy score", fontsize=20)
axes[0].set_ylabel("Frequency", fontsize=20)
axes[0].set_title("Distribution of secondary psychopathy scores", fontsize=22)
axes[0].tick_params(labelsize=16)
axes[0].legend(fontsize=16)

axes[1].bar(["Low ($\leq$22)", "High+ (>22)"],
            [counts[0], counts[1]],
            color=["#378ADD", "#EF9F27"], edgecolor="white")
axes[1].set_ylabel("Count", fontsize=20)
axes[1].set_title("Binary class distribution", fontsize=22)
axes[1].tick_params(labelsize=18)
for i, v in enumerate([counts[0], counts[1]]):
    axes[1].text(i, v + 1, str(v), ha="center", fontsize=20)

plt.tight_layout()
plt.savefig("target_distribution.png", dpi=200, bbox_inches="tight")
plt.show()


# ## Part 7 — Train/Test Split (80/20, stratified)

# In[15]:


# Regression split — stratified on quantile bins of continuous target
y_bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y_bins
)

# Classification split — stratified on binary class label
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.20, random_state=42, stratify=y_class
)

print(f"Regression    — Train: n={len(X_train)}, Test: n={len(X_test)}")
print(f"Classification — Train: n={len(X_train_c)}, Test: n={len(X_test_c)}")
print(f"  High+ in train: {y_train_c.sum()}, High+ in test: {y_test_c.sum()}")


# ## Part 8 — Feature Scaling (StandardScaler, fit on train only)

# In[16]:


# Regression scaling
scaler_reg     = StandardScaler()
X_train_scaled = pd.DataFrame(scaler_reg.fit_transform(X_train),   columns=feature_cols)
X_test_scaled  = pd.DataFrame(scaler_reg.transform(X_test),         columns=feature_cols)

# Classification scaling
scaler_clf       = StandardScaler()
X_train_c_scaled = pd.DataFrame(scaler_clf.fit_transform(X_train_c), columns=feature_cols)
X_test_c_scaled  = pd.DataFrame(scaler_clf.transform(X_test_c),      columns=feature_cols)

print("Scaling check — regression train set (mean≈0, std≈1):")
X_train_scaled.agg(["mean", "std"]).round(3)


# ## Part 9 — Random Oversampling (Class Imbalance)
# **Note:** Random oversampling duplicates existing minority cases with replacement.
# `class_weight='balanced'` in each model provides additional algorithmic-level compensation.

# In[17]:


rng          = np.random.default_rng(42)
X_arr        = X_train_c_scaled.values
y_arr        = y_train_c.values
minority_idx = np.where(y_arr == 1)[0]
majority_idx = np.where(y_arr == 0)[0]
n_to_add     = len(majority_idx) - len(minority_idx)

oversampled_idx = rng.choice(minority_idx, size=n_to_add, replace=True)

X_train_res = pd.DataFrame(
    np.vstack([X_arr, X_arr[oversampled_idx]]),
    columns=feature_cols
)
y_train_res = pd.Series(
    np.concatenate([y_arr, y_arr[oversampled_idx]]),
    name="secondary_psychopathy_binary"
)

print("Training set after oversampling:")
for cls, label in [(0, "Low  "), (1, "High+")]:
    n = (y_train_res == cls).sum()
    print(f"  {label} ({cls}): n={n}")
print(f"  Total: {len(y_train_res)}")


# ## Part 10 — Regression Pipeline 

# ### 10a — Model definitions & hyperparameter tuning

# In[15]:


cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

reg_models = {
    "Linear Regression": {
        "model":  LinearRegression(),
        "params": {}
    },
    "Elastic Net": {
        "model":  ElasticNet(max_iter=10000, random_state=42),
        "params": {
            "alpha":    [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    },
    "Random Forest (Reg)": {
        "model":  RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [None, 3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4]
        }
    },
    "KNN (Reg)": {
        "model":  KNeighborsRegressor(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"]
        }
    }
}

fitted_regs = {}
print("── Hyperparameter tuning (5-fold CV, scoring=neg_RMSE) ──\n")

for name, config in reg_models.items():
    if config["params"]:
        search = GridSearchCV(
            config["model"], config["params"],
            cv=cv_reg, scoring="neg_root_mean_squared_error",
            n_jobs=-1, refit=True
        )
        search.fit(X_train_scaled, y_train)
        fitted_regs[name] = search.best_estimator_
        print(f"{name}")
        print(f"  Best params: {search.best_params_}")
        print(f"  CV RMSE:     {-search.best_score_:.3f}\n")
    else:
        config["model"].fit(X_train_scaled, y_train)
        fitted_regs[name] = config["model"]
        cv_rmse = -cross_val_score(
            config["model"], X_train_scaled, y_train,
            cv=cv_reg, scoring="neg_root_mean_squared_error"
        ).mean()
        print(f"{name}\n  Best params: N/A\n  CV RMSE: {cv_rmse:.3f}\n")


# ### 10b — Regression test set evaluation

# In[16]:


reg_results = {}
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("─" * 50)

for name, model in fitted_regs.items():
    y_pred = model.predict(X_test_scaled)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    reg_results[name] = {"y_pred": y_pred, "RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"{name:<22} {rmse:>8.3f} {mae:>8.3f} {r2:>8.3f}")

print("\nNote: Poor R² values motivate reframing as binary classification.")


# ### 10c — Predicted vs Actual plots

# In[49]:


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # makes it easy to iterate

reg_colors = {
    "Linear Regression":   "#2196F3",
    "Elastic Net":         "#4CAF50",
    "Random Forest (Reg)": "#EF9F27",
    "KNN (Reg)":           "#D85A30"
}

for ax, (name, res) in zip(axes, reg_results.items()):
    ax.scatter(y_test, res["y_pred"], alpha=0.7,
               color=reg_colors[name], edgecolors="white", s=80)
    lo = min(y_test.min(), res["y_pred"].min()) - 1
    hi = max(y_test.max(), res["y_pred"].max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.6, label="Perfect fit")
    ax.set_xlabel("Actual score", fontsize=16)
    ax.set_ylabel("Predicted score", fontsize=16)
    ax.set_title(f"{name}\nRMSE={res['RMSE']:.3f}  $R^2$={res['R2']:.3f}", fontsize=16)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=15)

plt.suptitle("Regression: Predicted vs Actual secondary psychopathy scores",
             fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig("regression_predicted_vs_actual.png", dpi=200, bbox_inches="tight")
plt.show()


# ## Part 11 — Classification Pipeline (Low vs High+)

# ### 11a — Model definitions & hyperparameter tuning

# In[18]:


cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf_models = {
    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=10000, random_state=42, class_weight="balanced"
        ),
        "params": {
            "C":       [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver":  ["saga"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            random_state=42, class_weight="balanced"
        ),
        "params": {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [None, 3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"]
        }
    },
    "SVM": {
        "model": SVC(
            random_state=42, class_weight="balanced", probability=True
        ),
        "params": {
            "C":      [0.01, 0.1, 1.0, 5.0, 10.0],
            "kernel": ["linear", "rbf"],
            "gamma":  ["scale", "auto"]
        }
    }
}

fitted_clfs    = {}
clf_params_log = {}

print("── Hyperparameter tuning (5-fold stratified CV, scoring=balanced_accuracy) ──\n")

for name, config in clf_models.items():
    search = GridSearchCV(
        config["model"], config["params"],
        cv=cv_clf, scoring="balanced_accuracy",
        n_jobs=-1, refit=True
    )
    search.fit(X_train_res, y_train_res)
    fitted_clfs[name]    = search.best_estimator_
    clf_params_log[name] = search.best_params_
    print(f"{name}")
    print(f"  Best params:  {search.best_params_}")
    print(f"  CV bal. acc:  {search.best_score_:.3f}\n")

print("All classifiers trained.")


# ## Part 12 — Classification Test Set Evaluation

# In[19]:


clf_results = {}

print(f"{'Model':<22} {'Acc':>6} {'Bal Acc':>9} {'Sens':>7} {'Spec':>7} "
      f"{'PPV':>7} {'NPV':>7} {'F1':>7} {'AUC':>7}")
print("─" * 80)

for name, clf in fitted_clfs.items():
    y_pred  = clf.predict(X_test_c_scaled)
    y_proba = clf.predict_proba(X_test_c_scaled)[:, 1]

    cm               = confusion_matrix(y_test_c, y_pred, labels=[0, 1])
    tn, fp, fn, tp   = cm.ravel()

    acc     = accuracy_score(y_test_c, y_pred)
    bal_acc = balanced_accuracy_score(y_test_c, y_pred)
    sens    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv     = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv     = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1      = f1_score(y_test_c, y_pred, pos_label=1, zero_division=0)
    auc     = roc_auc_score(y_test_c, y_proba)

    clf_results[name] = {
        "y_pred": y_pred, "y_proba": y_proba,
        "acc": acc, "bal_acc": bal_acc,
        "sens": sens, "spec": spec,
        "ppv": ppv, "npv": npv,
        "f1": f1, "auc": auc
    }
    print(f"{name:<22} {acc:>6.3f} {bal_acc:>9.3f} {sens:>7.3f} {spec:>7.3f} "
          f"{ppv:>7.3f} {npv:>7.3f} {f1:>7.3f} {auc:>7.3f}")

print("\nSens=Sensitivity  Spec=Specificity  PPV=Positive Predictive Value  NPV=Negative Predictive Value")


# ### 12b — Full classification reports

# In[20]:


for name, res in clf_results.items():
    print(f"\n── {name} ──")
    print(classification_report(
        y_test_c, res["y_pred"],
        target_names=["Low (0)", "High+ (1)"],
        zero_division=0
    ))


# ## Part 13 — Confusion Matrices

# In[21]:


clf_colors = {
    "Logistic Regression": "Blues",
    "Random Forest":       "Oranges",
    "KNN":                 "Greens",
    "SVM":                 "Purples"
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, res) in zip(axes, clf_results.items()):
    cm = confusion_matrix(y_test_c, res["y_pred"], labels=[0, 1])
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap=clf_colors[name],
        xticklabels=["Low (pred)", "High+ (pred)"],
        yticklabels=["Low (actual)", "High+ (actual)"],
        ax=ax, cbar=False, annot_kws={"size": 18}
    )
    ax.set_title(
        f"{name}\nBal Acc={res['bal_acc']:.3f}  F1={res['f1']:.3f}  AUC={res['auc']:.3f}",
        fontsize=16
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

plt.suptitle(
    "Confusion matrices — binary classification (Low vs High+)",
    fontsize=18, y=1.02
)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.show()


# ## Part 14 — ROC Curves

# In[22]:


roc_colors = {
    "Logistic Regression": "#378ADD",
    "Random Forest":       "#EF9F27",
    "KNN":                 "#1D9E75",
    "SVM":                 "#9B59B6"
}

fig, ax = plt.subplots(figsize=(9, 8))

for name, res in clf_results.items():
    RocCurveDisplay.from_predictions(
        y_test_c, res["y_proba"],
        name=f"{name} (AUC={res['auc']:.3f})",
        color=roc_colors[name], ax=ax
    )

ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Chance")
ax.set_title("ROC curves — binary classification (Low vs High+)", fontsize=18)
ax.set_xlabel("False Positive Rate", fontsize=18)
ax.set_ylabel("True Positive Rate", fontsize=18)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=200, bbox_inches="tight")
plt.show()


# In[23]:


print("Checking y_proba values (should be floats between 0 and 1):")
for name, res in clf_results.items():
    print(f"\n{name}:")
    print(f"  Type: {type(res['y_proba'])}")
    print(f"  First 5 values: {res['y_proba'][:5].round(3)}")
    print(f"  Min: {res['y_proba'].min():.3f}, Max: {res['y_proba'].max():.3f}")


# ## Part 15 — Feature Importance

# ### 15a — Extract importance per model

# In[20]:


clf_importance = {}

# Logistic Regression — standardised coefficients
lr_coefs = pd.Series(
    fitted_clfs["Logistic Regression"].coef_[0],
    index=feature_cols
).sort_values(key=abs, ascending=False)
clf_importance["Logistic Regression"] = lr_coefs
print("Logistic Regression — standardised coefficients:")
print(lr_coefs.round(4))

# SVM — coefficients (linear kernel) or permutation importance (non-linear)
svm_kernel = fitted_clfs["SVM"].kernel
if svm_kernel == "linear":
    svm_imp = pd.Series(fitted_clfs["SVM"].coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
    print(f"\nSVM (linear kernel) — standardised coefficients:")
else:
    svm_perm = permutation_importance(
        fitted_clfs["SVM"], X_test_c_scaled, y_test_c,
        n_repeats=30, random_state=42, scoring="balanced_accuracy"
    )
    svm_imp = pd.Series(svm_perm.importances_mean, index=feature_cols).sort_values(ascending=False)
    print(f"\nSVM ({svm_kernel} kernel) — permutation importance:")
print(svm_imp.round(4))
clf_importance["SVM"] = svm_imp

# Random Forest — permutation importance
rf_perm = permutation_importance(
    fitted_clfs["Random Forest"], X_test_c_scaled, y_test_c,
    n_repeats=30, random_state=42, scoring="balanced_accuracy"
)
rf_imp = pd.Series(rf_perm.importances_mean, index=feature_cols).sort_values(ascending=False)
clf_importance["Random Forest"] = rf_imp
print("\nRandom Forest — permutation importance:")
print(rf_imp.round(4))

# KNN — permutation importance
knn_perm = permutation_importance(
    fitted_clfs["KNN"], X_test_c_scaled, y_test_c,
    n_repeats=30, random_state=42, scoring="balanced_accuracy"
)
knn_imp = pd.Series(knn_perm.importances_mean, index=feature_cols).sort_values(ascending=False)
clf_importance["KNN"] = knn_imp
print("\nKNN — permutation importance:")
print(knn_imp.round(4))


# ### 15b — Feature importance bar chart

# In[21]:


imp_colors = {
    "Logistic Regression": "#378ADD",
    "Random Forest":       "#EF9F27",
    "KNN":                 "#1D9E75",
    "SVM":                 "#9B59B6"
}

fig, ax = plt.subplots(figsize=(13, 5))
x     = np.arange(len(feature_cols))
width = 0.2

for i, (name, vals) in enumerate(clf_importance.items()):
    v      = vals.reindex(feature_cols).fillna(0).values
    abs_v  = np.abs(v)
    norm_v = abs_v / abs_v.max() if abs_v.max() > 0 else abs_v
    ax.bar(x + i * width, norm_v, width, label=name,
           color=imp_colors[name], alpha=0.85, edgecolor="white", linewidth=0.4)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(
    ["Emotional\nAbuse (EA)", "Physical\nAbuse (PA)", "Sexual\nAbuse (SA)",
     "Emotional\nNeglect (EN)", "Physical\nNeglect (PN)"],
    fontsize=18
)
ax.set_ylabel("Normalised absolute importance", fontsize=16)
ax.set_title("Feature importance across classifiers", fontsize=20)
ax.legend(fontsize=14)
ax.set_ylim(0, 1.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Part 16 — SHAP Explainability
# SHAP (SHapley Additive Explanations) provides individual-level feature contributions.  
# - **TreeExplainer** for Random Forest (exact)  
# - **LinearExplainer** for Logistic Regression  
# - **KernelExplainer** for KNN and SVM (model-agnostic, slower)

# ### 16a — Random Forest SHAP

# In[22]:


print("Computing SHAP values — Random Forest (TreeExplainer)...")
explainer_rf   = shap.TreeExplainer(fitted_clfs["Random Forest"])
shap_values_rf = explainer_rf.shap_values(X_test_c_scaled)
sv_rf = shap_values_rf[1] if isinstance(shap_values_rf, list) else shap_values_rf[:, :, 1]

plt.figure()
shap.summary_plot(sv_rf, X_test_c_scaled,
                  feature_names=feature_cols, plot_type="bar", show=False)
plt.title("SHAP — Random Forest (High+ class)", fontsize=12)
plt.tight_layout()
plt.savefig("shap_rf.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: shap_rf.png")


# ### 16b — Logistic Regression SHAP

# In[23]:


print("Computing SHAP values — Logistic Regression (LinearExplainer)...")
explainer_lr   = shap.LinearExplainer(fitted_clfs["Logistic Regression"], X_train_res)
shap_values_lr = explainer_lr.shap_values(X_test_c_scaled)
sv_lr = shap_values_lr[1] if isinstance(shap_values_lr, list) else shap_values_lr

plt.figure()
shap.summary_plot(sv_lr, X_test_c_scaled,
                  feature_names=feature_cols, plot_type="bar", show=False)
plt.title("SHAP — Logistic Regression (High+ class)", fontsize=12)
plt.tight_layout()
plt.savefig("shap_lr.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: shap_lr.png")


# ### 16c — KNN SHAP

# In[24]:


print("Computing SHAP values — KNN (KernelExplainer, may take a few minutes)...")
background_knn  = shap.sample(X_train_res, 50, random_state=42)
explainer_knn   = shap.KernelExplainer(fitted_clfs["KNN"].predict_proba, background_knn)
shap_values_knn = explainer_knn.shap_values(X_test_c_scaled)
sv_knn = shap_values_knn[1] if isinstance(shap_values_knn, list) else shap_values_knn[:, :, 1]

plt.figure()
shap.summary_plot(sv_knn, X_test_c_scaled,
                  feature_names=feature_cols, plot_type="bar", show=False)
plt.title("SHAP — KNN (High+ class)", fontsize=12)
plt.tight_layout()
plt.savefig("shap_knn.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: shap_knn.png")


# ### 16d — SVM SHAP

# In[25]:


print("Computing SHAP values — SVM (KernelExplainer, may take a few minutes)...")
background_svm  = shap.sample(X_train_res, 50, random_state=42)
explainer_svm   = shap.KernelExplainer(fitted_clfs["SVM"].predict_proba, background_svm)
shap_values_svm = explainer_svm.shap_values(X_test_c_scaled)
sv_svm = shap_values_svm[1] if isinstance(shap_values_svm, list) else shap_values_svm[:, :, 1]

plt.figure()
shap.summary_plot(sv_svm, X_test_c_scaled,
                  feature_names=feature_cols, plot_type="bar", show=False)
plt.title("SHAP — SVM (High+ class)", fontsize=12)
plt.tight_layout()
plt.savefig("shap_svm.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: shap_svm.png")


# In[26]:


fig, axes = plt.subplots(2, 2, figsize=(26, 22))
axes = axes.flatten()

shap_data = {
    "Random Forest":       (sv_rf,  X_test_c_scaled),
    "Logistic Regression": (sv_lr,  X_test_c_scaled),
    "KNN":                 (sv_knn, X_test_c_scaled),
    "SVM":                 (sv_svm, X_test_c_scaled),
}

feature_labels = [
    "Emotional Abuse",
    "Physical Abuse", 
    "Sexual Abuse",
    "Emotional Neglect",
    "Physical Neglect"
]

for ax, (name, (sv, X)) in zip(axes, shap_data.items()):
    mean_shap = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)
    sorted_vals = mean_shap[sorted_idx]
    sorted_labels = [feature_labels[i] for i in sorted_idx]

    bars = ax.barh(sorted_labels, sorted_vals,
                   color="#378ADD", edgecolor="white", height=0.6)
    ax.set_xlabel("mean(|SHAP value|)", fontsize=24, labelpad=20)
    ax.set_title(f"SHAP — {name} (High+ class)", fontsize=26, pad=25)
    ax.tick_params(axis='y', labelsize=24, pad=10)
    ax.tick_params(axis='x', labelsize=22, pad=10)
    ax.set_xlim(0, sorted_vals.max() * 1.25)

    for bar, val in zip(bars, sorted_vals):
        ax.text(val + sorted_vals.max() * 0.02,
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=22)

plt.suptitle("SHAP feature importance across all classifiers (High+ class)",
             fontsize=28, y=1.01)

plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.savefig("shap_all.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved: shap_all.png")

