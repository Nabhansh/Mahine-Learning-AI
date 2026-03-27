"""
Credit Card Fraud Detection
Uses Random Forest + SMOTE to handle imbalanced fraud data.
Install: pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ── 1. Generate synthetic fraud dataset ──────────────────────────────────────
np.random.seed(42)
n_legit, n_fraud = 9700, 300  # ~3% fraud rate

def make_transactions(n, fraud=False):
    if fraud:
        return pd.DataFrame({
            'amount':        np.random.exponential(800, n),
            'hour':          np.random.choice([0,1,2,3,23], n),
            'merchant_risk': np.random.beta(5, 2, n),
            'distance_km':   np.random.exponential(500, n),
            'prev_txn_gap':  np.random.exponential(2, n),
            'card_age_days': np.random.randint(1, 60, n),
            'failed_attempts': np.random.randint(1, 5, n),
            'label': 1
        })
    return pd.DataFrame({
        'amount':        np.random.exponential(80, n),
        'hour':          np.random.randint(6, 22, n),
        'merchant_risk': np.random.beta(2, 8, n),
        'distance_km':   np.random.exponential(20, n),
        'prev_txn_gap':  np.random.exponential(24, n),
        'card_age_days': np.random.randint(30, 3650, n),
        'failed_attempts': np.zeros(n, int),
        'label': 0
    })

df = pd.concat([make_transactions(n_legit), make_transactions(n_fraud, fraud=True)]).sample(frac=1).reset_index(drop=True)
print(f"Dataset: {len(df)} transactions | Fraud rate: {df.label.mean()*100:.1f}%")

# ── 2. Feature Engineering ───────────────────────────────────────────────────
df['amount_log']      = np.log1p(df['amount'])
df['is_night']        = df['hour'].between(22, 6) | df['hour'].between(0, 5)
df['high_risk_combo'] = (df['merchant_risk'] > 0.7) & (df['failed_attempts'] > 0)

features = ['amount_log', 'hour', 'merchant_risk', 'distance_km',
            'prev_txn_gap', 'card_age_days', 'failed_attempts',
            'is_night', 'high_risk_combo']

X = df[features].astype(float)
y = df['label']

# ── 3. Train / Test Split + Scaling ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. SMOTE oversampling (manual lightweight version) ───────────────────────
fraud_idx = np.where(y_train == 1)[0]
synthetic = X_train_s[fraud_idx][np.random.choice(len(fraud_idx), len(fraud_idx)*15, replace=True)]
noise = np.random.normal(0, 0.1, synthetic.shape)
X_balanced = np.vstack([X_train_s, synthetic + noise])
y_balanced = np.concatenate([y_train, np.ones(len(synthetic), int)])
print(f"After oversampling — Legit: {(y_balanced==0).sum()} | Fraud: {(y_balanced==1).sum()}")

# ── 5. Train Models ──────────────────────────────────────────────────────────
models = {
    'Random Forest':   RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    'Gradient Boost':  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'Logistic Reg':    LogisticRegression(C=1.0, max_iter=500, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_balanced, y_balanced)
    y_pred  = model.predict(X_test_s)
    y_prob  = model.predict_proba(X_test_s)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)
    results[name] = {'model': model, 'pred': y_pred, 'prob': y_prob, 'auc': auc}
    print(f"\n{'─'*40}\n{name}  (AUC={auc:.4f})\n{'─'*40}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

# ── 6. Visualizations ────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Credit Card Fraud Detection Dashboard', fontsize=15, fontweight='bold')

# Class distribution
axes[0,0].bar(['Legit', 'Fraud'], [n_legit, n_fraud], color=['steelblue', 'crimson'])
axes[0,0].set_title('Class Distribution'); axes[0,0].set_ylabel('Count')

# Confusion matrix
cm = confusion_matrix(y_test, best['pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'])
axes[0,1].set_title(f'Confusion Matrix — {best_name}')

# ROC Curves
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['prob'])
    axes[0,2].plot(fpr, tpr, label=f"{name} ({r['auc']:.3f})")
axes[0,2].plot([0,1],[0,1],'k--'); axes[0,2].set_xlabel('FPR'); axes[0,2].set_ylabel('TPR')
axes[0,2].set_title('ROC Curves'); axes[0,2].legend(fontsize=8)

# Feature importance (RF only)
rf_model = results['Random Forest']['model']
imp = pd.Series(rf_model.feature_importances_, index=features).sort_values()
imp.plot(kind='barh', ax=axes[1,0], color='teal')
axes[1,0].set_title('Feature Importances (RF)')

# Amount distribution
axes[1,1].hist(df[df.label==0]['amount'], bins=50, alpha=0.6, label='Legit', color='steelblue')
axes[1,1].hist(df[df.label==1]['amount'], bins=50, alpha=0.6, label='Fraud', color='crimson')
axes[1,1].set_title('Transaction Amount Distribution'); axes[1,1].legend(); axes[1,1].set_xlim(0,3000)

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, best['prob'])
axes[1,2].plot(rec, prec, color='purple')
axes[1,2].set_xlabel('Recall'); axes[1,2].set_ylabel('Precision')
axes[1,2].set_title(f'Precision-Recall — {best_name}')

plt.tight_layout()
plt.savefig('fraud_detection_results.png', dpi=150)
print(f"\n✅ Best model: {best_name} | AUC: {best['auc']:.4f}")
print("📊 Saved: fraud_detection_results.png")

# ── 7. Real-time prediction demo ─────────────────────────────────────────────
def predict_transaction(amount, hour, merchant_risk, distance_km,
                        prev_txn_gap, card_age_days, failed_attempts):
    feats = pd.DataFrame([{
        'amount_log':      np.log1p(amount),
        'hour':            hour,
        'merchant_risk':   merchant_risk,
        'distance_km':     distance_km,
        'prev_txn_gap':    prev_txn_gap,
        'card_age_days':   card_age_days,
        'failed_attempts': failed_attempts,
        'is_night':        int(hour >= 22 or hour <= 5),
        'high_risk_combo': int(merchant_risk > 0.7 and failed_attempts > 0),
    }])
    prob = rf_model.predict_proba(scaler.transform(feats))[0, 1]
    tag  = '🚨 FRAUD' if prob > 0.5 else '✅ LEGIT'
    print(f"\n[Transaction] ${amount:.2f} at {hour:02d}:00 → {tag} (fraud prob: {prob:.1%})")

print("\n── Sample Predictions ──")
predict_transaction(2500, 2, 0.9, 800, 0.5, 5, 3)   # likely fraud
predict_transaction(45,  14, 0.1,  5,  18,  730, 0)   # likely legit
predict_transaction(320,  1, 0.8, 300,  1,  15, 2)    # suspicious
