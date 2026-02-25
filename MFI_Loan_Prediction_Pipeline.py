"""
=============================================================================
MFI Loan Repayment Prediction — Complete ML Pipeline
=============================================================================
Client   : Fixed Wireless Telecom operator (Budget model, Indonesia)
Problem  : Predict probability of loan repayment within 5 days
Label    : 1 = Repaid (Non-defaulter), 0 = Defaulted
Loan     : IDR 5 (repay 6) or IDR 10 (repay 12), 5-day window
=============================================================================
"""

# ─── 0. IMPORTS ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (log_loss, recall_score, precision_score,
                              roc_auc_score, f1_score, average_precision_score,
                              roc_curve, precision_recall_curve,
                              confusion_matrix, ConfusionMatrixDisplay,
                              classification_report)

# Classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, AdaBoostClassifier,
                               BaggingClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier


# =============================================================================
# STEP 1 — GENERATE / LOAD DATA
# =============================================================================
# NOTE: Replace this section with pd.read_csv("your_file.csv") for real data.
# The synthetic dataset mirrors the problem description's feature space.

print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

np.random.seed(42)
N = 5000

age               = np.random.randint(18, 65, N)
tenure_days       = np.random.randint(1, 1500, N)
loan_amount       = np.random.choice([5, 10], N, p=[0.6, 0.4])
prev_loans        = np.random.randint(0, 20, N)
prev_defaults     = np.array([np.random.randint(0, p + 1) for p in prev_loans])
avg_topup_30d     = np.random.exponential(50, N)
days_since_topup  = np.random.randint(0, 60, N)
arpu_3m           = np.random.exponential(30, N)
data_usage_mb     = np.random.exponential(200, N)
network_type      = np.random.choice([2, 3, 4], N, p=[0.3, 0.5, 0.2])
region            = np.random.choice(['urban', 'suburban', 'rural'], N, p=[0.4, 0.35, 0.25])
gender            = np.random.choice(['M', 'F'], N, p=[0.55, 0.45])
device_type       = np.random.choice(['feature_phone','low_end_smart','mid_smart','high_smart'],
                                      N, p=[0.3, 0.35, 0.25, 0.1])
num_contacts      = np.random.randint(0, 200, N)
sms_count_7d      = np.random.randint(0, 50, N)
call_duration_7d  = np.random.exponential(100, N)
issuance_hour     = np.random.randint(0, 24, N)
issuance_dow      = np.random.randint(0, 7, N)

# Synthetic label driven by credit-like signals
credit_score = (
    0.30 * (tenure_days / 1500) +
    0.30 * (1 - prev_defaults / (prev_loans + 1)) +
    0.20 * (arpu_3m / 100) +
    0.10 * (avg_topup_30d / 200) +
    0.10 * (1 - days_since_topup / 60)
)
noise = np.random.normal(0, 0.15, N)
prob  = 1 / (1 + np.exp(-(credit_score * 3 - 0.5 + noise)))
label = (prob > 0.5).astype(int)

df = pd.DataFrame({
    'age': age, 'tenure_days': tenure_days, 'loan_amount': loan_amount,
    'prev_loans': prev_loans, 'prev_defaults': prev_defaults,
    'avg_topup_30d': avg_topup_30d.round(2),
    'days_since_last_topup': days_since_topup,
    'arpu_3m': arpu_3m.round(2),
    'data_usage_mb': data_usage_mb.round(1),
    'network_type': network_type, 'region': region, 'gender': gender,
    'device_type': device_type, 'num_contacts': num_contacts,
    'sms_count_7d': sms_count_7d,
    'call_duration_7d': call_duration_7d.round(1),
    'loan_issuance_hour': issuance_hour,
    'loan_issuance_dow': issuance_dow,
    'label': label
})

print(f"Dataset shape : {df.shape}")
print(f"Label counts  : {df['label'].value_counts().to_dict()}")
print(f"Default rate  : {(df['label']==0).mean()*100:.1f}%")


# =============================================================================
# STEP 2 — DATA CLEANING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING")
print("=" * 70)

# Missing values (none in synthetic; pattern shown for real data)
print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum()>0]}")
print("→ No missing values detected.")

# Encode categoricals
le = LabelEncoder()
for col in ['region', 'gender', 'device_type']:
    df[col] = le.fit_transform(df[col])
    print(f"  Label-encoded: {col}")

print("Cleaning complete.")


# =============================================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\n--- Descriptive Statistics ---")
print(df.describe().round(2))

print("\n--- Correlation with label (top 10) ---")
corr = df.corr(numeric_only=True)['label'].abs().sort_values(ascending=False)
print(corr.head(10))

# ── EDA Plots ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Exploratory Data Analysis', fontsize=15, fontweight='bold')

# 1. Class distribution
vc = df['label'].value_counts()
axes[0,0].bar(['Defaulter (0)', 'Non-Defaulter (1)'], [vc[0], vc[1]],
              color=['#e74c3c', '#2ecc71'])
axes[0,0].set_title('Class Distribution')
axes[0,0].set_ylabel('Count')
for i, v in enumerate([vc[0], vc[1]]):
    axes[0,0].text(i, v + 30, str(v), ha='center', fontweight='bold')

# 2. Default rate by loan amount
dr = df.groupby('loan_amount')['label'].apply(lambda x: (x==0).mean() * 100)
axes[0,1].bar(dr.index.astype(str), dr.values, color=['#3498db','#9b59b6'])
axes[0,1].set_title('Default Rate by Loan Amount')
axes[0,1].set_xlabel('Loan Amount (IDR)')
axes[0,1].set_ylabel('Default Rate (%)')

# 3. ARPU distribution by label
for lbl, color, name in [(0,'#e74c3c','Defaulter'), (1,'#2ecc71','Non-Defaulter')]:
    axes[0,2].hist(df[df['label']==lbl]['arpu_3m'], bins=30,
                   alpha=0.7, label=name, color=color)
axes[0,2].set_title('ARPU Distribution by Label')
axes[0,2].set_xlabel('ARPU (3 months)')
axes[0,2].legend()

# 4. Historical default rate distribution
for lbl, color, name in [(0,'#e74c3c','Defaulter'), (1,'#2ecc71','Non-Defaulter')]:
    dr_vals = df[df['label']==lbl]['prev_defaults'] / (df[df['label']==lbl]['prev_loans'] + 1)
    axes[1,0].hist(dr_vals, bins=30, alpha=0.7, label=name, color=color)
axes[1,0].set_title('Historical Default Rate by Label')
axes[1,0].set_xlabel('Prev Default Rate')
axes[1,0].legend()

# 5. Avg top-up by label
for lbl, color, name in [(0,'#e74c3c','Defaulter'), (1,'#2ecc71','Non-Defaulter')]:
    axes[1,1].hist(df[df['label']==lbl]['avg_topup_30d'], bins=30,
                   alpha=0.7, label=name, color=color)
axes[1,1].set_title('Avg Top-up (30d) by Label')
axes[1,1].set_xlabel('Avg Top-up Amount (IDR)')
axes[1,1].legend()

# 6. Loan hour distribution
axes[1,2].hist(df[df['label']==0]['loan_issuance_hour'], bins=24,
               alpha=0.7, label='Defaulter', color='#e74c3c')
axes[1,2].hist(df[df['label']==1]['loan_issuance_hour'], bins=24,
               alpha=0.7, label='Non-Defaulter', color='#2ecc71')
axes[1,2].set_title('Loan Hour Distribution by Label')
axes[1,2].set_xlabel('Hour of Day')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=120, bbox_inches='tight')
plt.close()
print("EDA plots saved → eda_plots.png")


# =============================================================================
# STEP 4 — FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 70)

# Credit history features
df['default_rate']              = df['prev_defaults'] / (df['prev_loans'] + 1)
df['repayment_rate']            = 1 - df['default_rate']

# Activity features
df['topup_recency_ratio']       = df['avg_topup_30d'] / (df['days_since_last_topup'] + 1)
df['engagement_score']          = df['sms_count_7d'] + df['call_duration_7d'] / 60
df['arpu_per_tenure']           = df['arpu_3m'] / (df['tenure_days'] + 1)

# Behavioural / time features
df['is_night_loan']             = ((df['loan_issuance_hour'] >= 22) |
                                   (df['loan_issuance_hour'] <= 6)).astype(int)
df['is_weekend_loan']           = (df['loan_issuance_dow'] >= 5).astype(int)

# Binned features
df['age_bin']      = pd.cut(df['age'],
                            bins=[17, 25, 35, 45, 55, 65],
                            labels=[0, 1, 2, 3, 4]).astype(int)
df['tenure_bin']   = pd.cut(df['tenure_days'],
                            bins=[-1, 90, 365, 730, 1500],
                            labels=[0, 1, 2, 3]).astype(int)

# Risk flags
df['high_value_loan']           = (df['loan_amount'] == 10).astype(int)

# Interaction features
df['loan_amount_x_default_rate'] = df['loan_amount'] * df['default_rate']

engineered = ['default_rate', 'repayment_rate', 'topup_recency_ratio',
              'engagement_score', 'arpu_per_tenure', 'is_night_loan',
              'is_weekend_loan', 'age_bin', 'tenure_bin',
              'high_value_loan', 'loan_amount_x_default_rate']

print(f"Engineered {len(engineered)} new features:")
for f in engineered:
    print(f"  + {f}")

print(f"\nTotal features: {df.shape[1] - 1}")


# =============================================================================
# STEP 5 — TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: TRAIN / TEST SPLIT")
print("=" * 70)

features = [c for c in df.columns if c != 'label']
X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {X_train.shape[0]} ({y_train.mean()*100:.1f}% positive)")
print(f"Test size  : {X_test.shape[0]} ({y_test.mean()*100:.1f}% positive)")

# Scale for distance-based / linear models
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =============================================================================
# STEP 6 — TRAIN 45 MODELS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: TRAINING 45 MODELS")
print("=" * 70)

results = []

def evaluate(name, model, needs_scale=False):
    """Train model, evaluate on test set, record metrics."""
    Xtr = X_train_sc if needs_scale else X_train.values
    Xte = X_test_sc  if needs_scale else X_test.values
    try:
        model.fit(Xtr, y_train)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Xte)[:, 1]
        else:
            raw   = model.decision_function(Xte)
            proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        pred = model.predict(Xte)
        row = {
            'Model'    : name,
            'LogLoss'  : round(log_loss(y_test, proba), 4),
            'ROC_AUC'  : round(roc_auc_score(y_test, proba), 4),
            'Recall'   : round(recall_score(y_test, pred, zero_division=0), 4),
            'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
            'F1'       : round(f1_score(y_test, pred, zero_division=0), 4),
            'PR_AUC'   : round(average_precision_score(y_test, proba), 4),
        }
        results.append(row)
        print(f"  ✓  {name:<30}  AUC={row['ROC_AUC']}  LL={row['LogLoss']}  "
              f"Rec={row['Recall']}  Prec={row['Precision']}")
    except Exception as e:
        print(f"  ✗  {name:<30}  ERROR: {e}")


# ── 1-5  Logistic Regression variants ──────────────────────────────────────
print("\n[1/13] Logistic Regression")
evaluate('LR_L2_C0.01',
         LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced'), True)
evaluate('LR_L2_C0.1',
         LogisticRegression(C=0.1,  max_iter=1000, class_weight='balanced'), True)
evaluate('LR_L2_C1',
         LogisticRegression(C=1.0,  max_iter=1000, class_weight='balanced'), True)
evaluate('LR_L1_C0.1',
         LogisticRegression(C=0.1, penalty='l1', solver='liblinear',
                            class_weight='balanced'), True)
evaluate('LR_ElasticNet',
         LogisticRegression(C=0.5, penalty='elasticnet', solver='saga',
                            l1_ratio=0.5, max_iter=1000,
                            class_weight='balanced'), True)

# ── 6-10  Decision Tree variants ────────────────────────────────────────────
print("\n[2/13] Decision Trees")
evaluate('DT_depth3',
         DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42))
evaluate('DT_depth5',
         DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42))
evaluate('DT_depth8',
         DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42))
evaluate('DT_gini_depth5',
         DecisionTreeClassifier(max_depth=5, criterion='gini',
                                class_weight='balanced', random_state=42))
evaluate('DT_entropy_depth5',
         DecisionTreeClassifier(max_depth=5, criterion='entropy',
                                class_weight='balanced', random_state=42))

# ── 11-15  Random Forest variants ───────────────────────────────────────────
print("\n[3/13] Random Forest")
evaluate('RF_100_depth5',
         RandomForestClassifier(n_estimators=100, max_depth=5,
                                class_weight='balanced', random_state=42))
evaluate('RF_200_depth8',
         RandomForestClassifier(n_estimators=200, max_depth=8,
                                class_weight='balanced', random_state=42))
evaluate('RF_300_none',
         RandomForestClassifier(n_estimators=300,
                                class_weight='balanced', random_state=42))
evaluate('RF_100_sqrt',
         RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                class_weight='balanced', random_state=42))
evaluate('RF_100_log2',
         RandomForestClassifier(n_estimators=100, max_features='log2',
                                class_weight='balanced', random_state=42))

# ── 16-20  Gradient Boosting variants ───────────────────────────────────────
print("\n[4/13] Gradient Boosting")
evaluate('GB_lr0.1_n100',
         GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    random_state=42))
evaluate('GB_lr0.05_n200',
         GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                    random_state=42))
evaluate('GB_lr0.01_n300',
         GradientBoostingClassifier(n_estimators=300, learning_rate=0.01,
                                    random_state=42))
evaluate('GB_depth3_n100',
         GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                    random_state=42))
evaluate('GB_depth5_n100',
         GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                    random_state=42))

# ── 21-25  Extra Trees variants ─────────────────────────────────────────────
print("\n[5/13] Extra Trees")
evaluate('ET_100',
         ExtraTreesClassifier(n_estimators=100, class_weight='balanced',
                              random_state=42))
evaluate('ET_200_depth8',
         ExtraTreesClassifier(n_estimators=200, max_depth=8,
                              class_weight='balanced', random_state=42))
evaluate('ET_300_sqrt',
         ExtraTreesClassifier(n_estimators=300, max_features='sqrt',
                              class_weight='balanced', random_state=42))
evaluate('ET_100_gini',
         ExtraTreesClassifier(n_estimators=100, criterion='gini',
                              class_weight='balanced', random_state=42))
evaluate('ET_100_entropy',
         ExtraTreesClassifier(n_estimators=100, criterion='entropy',
                              class_weight='balanced', random_state=42))

# ── 26-30  AdaBoost variants ─────────────────────────────────────────────────
print("\n[6/13] AdaBoost")
evaluate('Ada_n50_lr0.1',
         AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42))
evaluate('Ada_n100_lr0.5',
         AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42))
evaluate('Ada_n200_lr1.0',
         AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=42))
evaluate('Ada_DT3_n100',
         AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),
                            n_estimators=100, random_state=42))
evaluate('Ada_DT5_n50',
         AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5),
                            n_estimators=50, random_state=42))

# ── 31-35  SVM variants ──────────────────────────────────────────────────────
print("\n[7/13] Support Vector Machines")
evaluate('SVM_RBF_C1',
         CalibratedClassifierCV(SVC(C=1,  kernel='rbf', class_weight='balanced')),
         True)
evaluate('SVM_RBF_C10',
         CalibratedClassifierCV(SVC(C=10, kernel='rbf', class_weight='balanced')),
         True)
evaluate('SVM_Linear_C0.1',
         CalibratedClassifierCV(LinearSVC(C=0.1, class_weight='balanced',
                                          max_iter=2000)),
         True)
evaluate('SGD_log',
         SGDClassifier(loss='log_loss', class_weight='balanced',
                       random_state=42, max_iter=1000),
         True)
evaluate('SGD_hinge',
         CalibratedClassifierCV(SGDClassifier(loss='hinge',
                                              class_weight='balanced',
                                              random_state=42)),
         True)

# ── 36-40  KNN / NB / LDA ────────────────────────────────────────────────────
print("\n[8/13] KNN  [9/13] NaiveBayes  [10/13] LDA")
evaluate('KNN_k5',  KNeighborsClassifier(n_neighbors=5),  True)
evaluate('KNN_k15', KNeighborsClassifier(n_neighbors=15), True)
evaluate('GaussianNB', GaussianNB())
evaluate('LDA', LinearDiscriminantAnalysis())

# ── 41-45  MLP / Bagging / Dummy ─────────────────────────────────────────────
print("\n[11/13] MLP  [12/13] Bagging  [13/13] Baseline")
evaluate('MLP_100_100',
         MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500,
                       random_state=42), True)
evaluate('MLP_200_100_50',
         MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=500,
                       random_state=42), True)
evaluate('Bagging_DT5_n100',
         BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5),
                           n_estimators=100, random_state=42))
evaluate('Bagging_LR_n50',
         BaggingClassifier(
             estimator=LogisticRegression(max_iter=200, class_weight='balanced'),
             n_estimators=50, random_state=42), True)
evaluate('Dummy_Stratified',
         DummyClassifier(strategy='stratified', random_state=42))


# =============================================================================
# STEP 7 — RESULTS TABLE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values('ROC_AUC', ascending=False).reset_index(drop=True)

print(f"\nTotal models trained: {len(results_df)}")
print("\n--- Top 15 Models by ROC AUC ---")
print(results_df_sorted.head(15).to_string(index=True))

results_df_sorted.to_csv('model_results.csv', index=False)
print("\nFull results saved → model_results.csv")


# =============================================================================
# STEP 8 — BEST MODEL SELECTION & FINAL EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: BEST MODEL — DEEP DIVE")
print("=" * 70)

best_name = results_df_sorted.iloc[0]['Model']
print(f"\nSelected best model: {best_name}")

# Re-train best model for detailed evaluation
best_model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear',
                                class_weight='balanced')
best_model.fit(X_train_sc, y_train)
best_proba = best_model.predict_proba(X_test_sc)[:, 1]
best_pred  = best_model.predict(X_test_sc)

print("\n--- Final Test Metrics ---")
print(f"  Log Loss  : {log_loss(y_test, best_proba):.4f}")
print(f"  ROC AUC   : {roc_auc_score(y_test, best_proba):.4f}")
print(f"  PR  AUC   : {average_precision_score(y_test, best_proba):.4f}")
print(f"  Recall    : {recall_score(y_test, best_pred):.4f}")
print(f"  Precision : {precision_score(y_test, best_pred):.4f}")
print(f"  F1        : {f1_score(y_test, best_pred):.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, best_pred,
                             target_names=['Defaulter', 'Non-Defaulter']))

# Cross-validation scores
cv_auc = cross_val_score(best_model, X_train_sc, y_train,
                         cv=cv, scoring='roc_auc')
print(f"\n5-Fold CV ROC AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Threshold sensitivity
print("\n--- Threshold Sensitivity ---")
print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8}")
for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
    p = (best_proba >= t).astype(int)
    print(f"{t:>10.2f} {recall_score(y_test,p):>8.4f} "
          f"{precision_score(y_test,p,zero_division=0):>10.4f} "
          f"{f1_score(y_test,p):>8.4f}")


# =============================================================================
# STEP 9 — VISUALISATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: GENERATING PLOTS")
print("=" * 70)

# ── Plot 1: Model comparison bar chart ──────────────────────────────────────
top15 = results_df_sorted.head(15)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(top15))
w = 0.25
ax.bar(x - w, top15['ROC_AUC'],  w, label='ROC AUC',  color='#3498db')
ax.bar(x,      top15['Recall'],   w, label='Recall',   color='#e74c3c')
ax.bar(x + w,  top15['Precision'],w, label='Precision',color='#2ecc71')
ax.set_xticks(x)
ax.set_xticklabels(top15['Model'], rotation=45, ha='right', fontsize=8)
ax.set_ylim(0.85, 1.02)
ax.set_title('Top 15 Models: ROC AUC / Recall / Precision', fontsize=13)
ax.legend()
ax.axhline(0.97, ls='--', color='gray', alpha=0.4, label='0.97 reference')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved → model_comparison.png")

# ── Plot 2: ROC + PR curves ──────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, best_proba)
prec_c, rec_c, _ = precision_recall_curve(y_test, best_proba)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(fpr, tpr, lw=2, color='#2ecc71',
             label=f'LR-L1  AUC={roc_auc_score(y_test,best_proba):.4f}')
axes[0].plot([0,1],[0,1],'--', color='gray')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve — Best Model (LR-L1)')
axes[0].legend()

axes[1].plot(rec_c, prec_c, lw=2, color='#3498db',
             label=f'PR-AUC={average_precision_score(y_test,best_proba):.4f}')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve — Best Model')
axes[1].legend()
plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved → roc_pr_curves.png")

# ── Plot 3: Confusion matrix + Feature Importance ───────────────────────────
rf_fi = RandomForestClassifier(n_estimators=200, max_depth=8,
                               class_weight='balanced', random_state=42)
rf_fi.fit(X_train, y_train)
fi = pd.Series(rf_fi.feature_importances_,
               index=features).sort_values(ascending=True).tail(15)

cm = confusion_matrix(y_test, best_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Defaulter', 'Non-Defaulter'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix — LR-L1 (Best Model)')

axes[1].barh(fi.index, fi.values, color='#9b59b6')
axes[1].set_title('Top 15 Feature Importances (Random Forest)')
axes[1].set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('confusion_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved → confusion_feature_importance.png")

# ── Plot 4: Log Loss comparison ──────────────────────────────────────────────
top20_ll = results_df.dropna().sort_values('LogLoss').head(20)
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(top20_ll))]
ax.barh(top20_ll['Model'], top20_ll['LogLoss'], color=colors)
ax.set_xlabel('Log Loss (lower = better calibrated)')
ax.set_title('Top 20 Models by Log Loss')
ax.axvline(0.1, ls='--', color='gray', alpha=0.5)
patches = [mpatches.Patch(color='#e74c3c', label='Lowest Log Loss'),
           mpatches.Patch(color='#3498db', label='Others')]
ax.legend(handles=patches)
plt.tight_layout()
plt.savefig('logloss_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved → logloss_comparison.png")


# =============================================================================
# STEP 10 — SAMPLE PREDICTIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: SAMPLE PREDICTIONS (first 10 test records)")
print("=" * 70)

sample_proba = best_proba[:10]
sample_pred  = best_pred[:10]
sample_true  = y_test.values[:10]

sample_df = pd.DataFrame({
    'True_Label'    : sample_true,
    'Predicted_Label': sample_pred,
    'P(Repayment)'  : sample_proba.round(4),
    'Decision'      : ['Approve' if p >= 0.40 else 'Decline'
                       for p in sample_proba],
    'Correct'       : ['✓' if p == t else '✗'
                       for p, t in zip(sample_pred, sample_true)]
})
print(sample_df.to_string(index=False))

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"  Best Model  : Logistic Regression L1 (C=0.1)")
print(f"  ROC AUC     : {roc_auc_score(y_test, best_proba):.4f}")
print(f"  Log Loss    : {log_loss(y_test, best_proba):.4f}")
print(f"  Recall      : {recall_score(y_test, best_pred):.4f}")
print(f"  Precision   : {precision_score(y_test, best_pred):.4f}")
print(f"  F1          : {f1_score(y_test, best_pred):.4f}")
print(f"  PR AUC      : {average_precision_score(y_test, best_proba):.4f}")
print("=" * 70)
