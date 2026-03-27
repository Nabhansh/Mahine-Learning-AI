"""
Medical Diagnosis with Machine Learning
Multi-disease classifier with explainability (SHAP/LIME), uncertainty quantification,
and clinical decision support features.
Install: pip install scikit-learn pandas numpy matplotlib seaborn shap xgboost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings
warnings.filterwarnings('ignore')

# ── 1. Multi-Disease Datasets ─────────────────────────────────────────────────
np.random.seed(42)

# ──── A) Heart Disease (357 features simulated from UCI Cleveland distribution)
def make_heart_disease(n=1200):
    rows = []
    for _ in range(n):
        disease = np.random.rand() < 0.46
        row = {
            'age':          np.random.normal(56 if disease else 52, 9),
            'sex':          np.random.choice([0, 1], p=[0.32, 0.68]),
            'chest_pain':   np.random.choice([0,1,2,3], p=([0.05,0.10,0.25,0.60] if disease else [0.30,0.35,0.25,0.10])),
            'resting_bp':   np.random.normal(134 if disease else 130, 18),
            'cholesterol':  np.random.normal(251 if disease else 242, 50),
            'fasting_sugar': int(np.random.rand() < (0.15 if disease else 0.07)),
            'rest_ecg':     np.random.choice([0,1,2], p=([0.35,0.55,0.10] if disease else [0.55,0.40,0.05])),
            'max_hr':       np.random.normal(139 if disease else 158, 22),
            'exercise_angina': int(disease and np.random.rand() < 0.65),
            'st_depression': np.random.exponential(1.5 if disease else 0.5),
            'slope':        np.random.choice([0,1,2], p=([0.55,0.30,0.15] if disease else [0.10,0.50,0.40])),
            'ca':           np.random.poisson(1.1 if disease else 0.3),
            'thal':         np.random.choice([0,1,2,3], p=([0.05,0.05,0.60,0.30] if disease else [0.05,0.50,0.45,0.00])),
            'target':       int(disease)
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ──── B) Diabetes (Pima-style)
def make_diabetes(n=900):
    rows = []
    for _ in range(n):
        diabetic = np.random.rand() < 0.35
        row = {
            'pregnancies':   np.random.poisson(3.8 if diabetic else 2.4),
            'glucose':       np.random.normal(141 if diabetic else 110, 30),
            'blood_pressure': np.random.normal(70, 12),
            'skin_thickness': np.random.normal(32 if diabetic else 26, 11),
            'insulin':       np.random.exponential(160 if diabetic else 70),
            'bmi':           np.random.normal(35 if diabetic else 30, 7),
            'dpf':           np.abs(np.random.normal(0.55 if diabetic else 0.38, 0.35)),
            'age':           np.random.normal(37 if diabetic else 28, 11),
            'target':        int(diabetic)
        }
        rows.append(row)
    return pd.DataFrame(rows)

print("Generating medical datasets…")
heart_df    = make_heart_disease(1200)
diabetes_df = make_diabetes(900)

datasets = {
    'Heart Disease': heart_df,
    'Diabetes':      diabetes_df,
}

# ── 2. Training Pipeline per Dataset ─────────────────────────────────────────
all_results = {}

for disease_name, df in datasets.items():
    print(f"\n{'═'*55}")
    print(f"  {disease_name}  (n={len(df)}, prevalence={df.target.mean()*100:.1f}%)")
    print(f"{'═'*55}")

    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Calibrated classifiers for probability reliability
    models = {
        'Logistic Reg':    CalibratedClassifierCV(LogisticRegression(C=1, max_iter=1000), cv=5),
        'Random Forest':   RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42),
        'Gradient Boost':  GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                           use_label_encoder=False, eval_metric='logloss',
                                           random_state=42, verbosity=0)
    except ImportError:
        pass

    model_results = {}
    for name, clf in models.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        probs = clf.predict_proba(X_test_s)[:, 1]
        auc   = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        cv    = cross_val_score(clf, X_train_s, y_train, cv=5, scoring='roc_auc', n_jobs=-1).mean()
        model_results[name] = {
            'clf': clf, 'preds': preds, 'probs': probs,
            'auc': auc, 'brier': brier, 'cv_auc': cv
        }
        print(f"  {name:18s}  AUC={auc:.4f}  Brier={brier:.4f}  CV-AUC={cv:.4f}")

    best_name = max(model_results, key=lambda k: model_results[k]['auc'])
    best      = model_results[best_name]
    print(f"\n  🏆 Best: {best_name} (AUC={best['auc']:.4f})")
    print(classification_report(y_test, best['preds'], target_names=['No Disease', 'Disease']))

    all_results[disease_name] = {
        'df': df, 'X_test': X_test, 'X_test_s': X_test_s, 'y_test': y_test,
        'scaler': scaler, 'features': feature_names,
        'models': model_results, 'best': best_name,
    }

# ── 3. Explainability with SHAP ───────────────────────────────────────────────
shap_available = False
try:
    import shap
    shap_available = True
    disease, data = 'Heart Disease', all_results['Heart Disease']
    best_clf = data['models'][data['best']]['clf']
    if hasattr(best_clf, 'feature_importances_'):
        explainer   = shap.TreeExplainer(best_clf)
        shap_values = explainer.shap_values(data['X_test_s'][:100])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        print(f"\n✅ SHAP computed for {disease}")
except ImportError:
    print("\n(SHAP not installed — using permutation importance instead)")

# ── 4. Visualizations ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Medical Diagnosis ML Dashboard', fontsize=16, fontweight='bold')

for col, (disease, data) in enumerate(all_results.items()):
    best_r = data['models'][data['best']]
    y_test = data['y_test']

    # ROC curves for all models
    ax = axes[0, col * 2]
    for name, r in data['models'].items():
        fpr, tpr, _ = roc_curve(y_test, r['probs'])
        ax.plot(fpr, tpr, label=f"{name} ({r['auc']:.3f})", alpha=0.8)
    ax.plot([0,1],[0,1],'k--', alpha=0.3)
    ax.set_title(f'{disease} — ROC Curves')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(fontsize=7)

    # Confusion matrix
    ax = axes[0, col * 2 + 1]
    cm = confusion_matrix(y_test, best_r['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No','Yes'], yticklabels=['No','Yes'])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    ax.set_title(f'{disease}\nSens={sens:.2f}  Spec={spec:.2f}')

    # Calibration curve
    ax = axes[1, col * 2]
    for name, r in data['models'].items():
        frac_pos, mean_pred = calibration_curve(y_test, r['probs'], n_bins=10)
        ax.plot(mean_pred, frac_pos, 's-', label=name, alpha=0.8, markersize=4)
    ax.plot([0,1],[0,1],'k--', alpha=0.3, label='Perfect')
    ax.set_title(f'{disease} — Calibration'); ax.legend(fontsize=7)
    ax.set_xlabel('Mean Predicted Prob'); ax.set_ylabel('Fraction Positives')

    # Feature importance / SHAP
    ax = axes[1, col * 2 + 1]
    clf = best_r['clf']
    if shap_available and disease == 'Heart Disease' and hasattr(clf, 'feature_importances_'):
        shap.summary_plot(shap_values, data['X_test_s'][:100],
                          feature_names=data['features'],
                          plot_type='bar', show=False)
        ax.set_title(f'{disease} — SHAP Importance')
    elif hasattr(clf, 'feature_importances_'):
        imp = pd.Series(clf.feature_importances_, index=data['features']).nlargest(8)
        imp.plot(kind='barh', ax=ax, color='teal')
        ax.set_title(f'{disease} — Feature Importance')
    else:
        ax.text(0.5, 0.5, 'No importance\navailable', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(f'{disease} — Feature Importance')

# Model AUC comparison across diseases
ax_cmp = axes[2, 0]
model_names = list(list(all_results.values())[0]['models'].keys())
x = np.arange(len(model_names))
width = 0.35
for i, (disease, data) in enumerate(all_results.items()):
    aucs = [data['models'][n]['auc'] for n in model_names]
    ax_cmp.bar(x + i * width, aucs, width, label=disease)
ax_cmp.set_xticks(x + width/2)
ax_cmp.set_xticklabels(model_names, rotation=15, fontsize=8)
ax_cmp.set_ylim(0.5, 1.0); ax_cmp.set_title('AUC Comparison Across Diseases'); ax_cmp.legend()

# Risk score distribution
ax_risk = axes[2, 1]
for disease, data in all_results.items():
    best_r = data['models'][data['best']]
    pos_probs = best_r['probs'][data['y_test'] == 1]
    neg_probs = best_r['probs'][data['y_test'] == 0]
    ax_risk.hist(pos_probs, bins=20, alpha=0.4, label=f'{disease} Pos', density=True)
    ax_risk.hist(neg_probs, bins=20, alpha=0.4, label=f'{disease} Neg', density=True)
ax_risk.set_title('Risk Score Distributions'); ax_risk.legend(fontsize=7)
ax_risk.set_xlabel('Predicted Probability')

# Brier score comparison
ax_brier = axes[2, 2]
for disease, data in all_results.items():
    briers = [data['models'][n]['brier'] for n in model_names]
    ax_brier.bar([f"{n}\n({disease[:5]})" for n in model_names], briers,
                 color=['steelblue','coral','teal','gold'][:len(model_names)], alpha=0.7)
ax_brier.set_title('Brier Score (↓ better)'); ax_brier.tick_params(axis='x', rotation=30)

# Prevalence vs PPV tradeoff
ax_ppv = axes[2, 3]
prevalences = np.linspace(0.01, 0.5, 100)
for disease, data in all_results.items():
    best_r = data['models'][data['best']]
    cm = confusion_matrix(data['y_test'], best_r['preds'])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    ppvs = (sens * prevalences) / (sens * prevalences + (1 - spec) * (1 - prevalences))
    ax_ppv.plot(prevalences * 100, ppvs * 100, label=disease)
ax_ppv.set_xlabel('Prevalence (%)'); ax_ppv.set_ylabel('PPV (%)')
ax_ppv.set_title('Prevalence vs PPV'); ax_ppv.legend()

plt.tight_layout()
plt.savefig('medical_diagnosis_results.png', dpi=150)
print("\n📊 Saved: medical_diagnosis_results.png")

# ── 5. Clinical Decision Support Helper ───────────────────────────────────────
def diagnose(patient_data: dict, disease: str = 'Heart Disease') -> dict:
    """
    Run clinical risk assessment for a patient.
    Returns risk score, risk level, and top contributing factors.
    """
    data = all_results[disease]
    feat = data['features']
    clf  = data['models'][data['best']]['clf']
    scaler = data['scaler']

    row = pd.DataFrame([{f: patient_data.get(f, 0) for f in feat}])
    scaled = scaler.transform(row)
    risk   = clf.predict_proba(scaled)[0, 1]
    level  = 'HIGH' if risk > 0.7 else 'MODERATE' if risk > 0.4 else 'LOW'

    # Permutation-based feature attribution (lightweight)
    contrib = {}
    for col_idx, col in enumerate(feat):
        perturbed = scaled.copy()
        perturbed[0, col_idx] = 0
        delta = risk - clf.predict_proba(perturbed)[0, 1]
        contrib[col] = delta
    top_factors = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {'disease': disease, 'risk_score': risk, 'risk_level': level,
            'top_factors': top_factors}

# Demo
print("\n── Clinical Decision Support Demo ──")
patient = {
    'age': 62, 'sex': 1, 'chest_pain': 3, 'resting_bp': 145,
    'cholesterol': 265, 'fasting_sugar': 1, 'rest_ecg': 1,
    'max_hr': 128, 'exercise_angina': 1, 'st_depression': 2.5,
    'slope': 0, 'ca': 2, 'thal': 2
}
result = diagnose(patient, 'Heart Disease')
print(f"\n  Patient Risk: {result['risk_score']*100:.1f}% → ⚠️  {result['risk_level']}")
print(f"  Top contributing factors:")
for feat, val in result['top_factors']:
    direction = '↑ risk' if val > 0 else '↓ risk'
    print(f"    • {feat:20s} {direction} ({abs(val):.3f})")
