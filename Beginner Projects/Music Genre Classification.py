"""
Music Genre Classification
Extracts audio features (MFCC, chroma, spectral) and classifies genres.
Install: pip install librosa scikit-learn pandas numpy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# ── 1. Feature Extraction (with librosa) ─────────────────────────────────────
def extract_features(file_path: str) -> dict:
    """
    Extract audio features from a .wav file.
    Returns dict of feature_name → value.
    """
    try:
        import librosa
        y, sr = librosa.load(file_path, duration=30)
        features = {}

        # MFCCs (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = mfcc[i].mean()
            features[f'mfcc_{i}_std']  = mfcc[i].std()

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = chroma.mean()
        features['chroma_std']  = chroma.std()

        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr       = librosa.feature.zero_crossing_rate(y)
        rmse      = librosa.feature.rms(y=y)
        tempo, _  = librosa.beat.beat_track(y=y, sr=sr)

        features.update({
            'spectral_centroid_mean': spec_cent.mean(),
            'spectral_bandwidth_mean': spec_bw.mean(),
            'spectral_rolloff_mean':   spec_roll.mean(),
            'zero_crossing_rate_mean': zcr.mean(),
            'rmse_mean':               rmse.mean(),
            'tempo':                   float(tempo),
        })
        return features

    except ImportError:
        raise RuntimeError("Install librosa: pip install librosa")
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")


# ── 2. Synthetic Dataset (for demo without audio files) ──────────────────────
def generate_synthetic_dataset(n_per_genre: int = 150) -> pd.DataFrame:
    """
    Simulate audio feature distributions per genre based on real-world stats.
    Allows running the ML pipeline without actual audio files.
    """
    np.random.seed(42)
    genre_profiles = {
        'blues':     dict(tempo=80,  centroid=1800, zcr=0.05, chroma=0.45, rmse=0.12),
        'classical': dict(tempo=100, centroid=1200, zcr=0.02, chroma=0.40, rmse=0.06),
        'country':   dict(tempo=120, centroid=1700, zcr=0.06, chroma=0.50, rmse=0.10),
        'disco':     dict(tempo=125, centroid=2200, zcr=0.08, chroma=0.55, rmse=0.14),
        'hiphop':    dict(tempo=95,  centroid=1600, zcr=0.10, chroma=0.48, rmse=0.16),
        'jazz':      dict(tempo=110, centroid=1500, zcr=0.04, chroma=0.60, rmse=0.08),
        'metal':     dict(tempo=160, centroid=3000, zcr=0.15, chroma=0.35, rmse=0.20),
        'pop':       dict(tempo=118, centroid=2000, zcr=0.07, chroma=0.52, rmse=0.13),
        'reggae':    dict(tempo=90,  centroid=1400, zcr=0.04, chroma=0.45, rmse=0.09),
        'rock':      dict(tempo=140, centroid=2500, zcr=0.12, chroma=0.42, rmse=0.18),
    }
    rows = []
    for genre, p in genre_profiles.items():
        for _ in range(n_per_genre):
            row = {'genre': genre}
            row['tempo']             = np.random.normal(p['tempo'],     10)
            row['spectral_centroid'] = np.random.normal(p['centroid'],  200)
            row['zcr']               = np.abs(np.random.normal(p['zcr'], 0.015))
            row['chroma_mean']       = np.clip(np.random.normal(p['chroma'], 0.04), 0, 1)
            row['rmse_mean']         = np.abs(np.random.normal(p['rmse'], 0.02))
            row['spectral_bw']       = np.random.normal(p['centroid'] * 0.6, 150)
            row['rolloff']           = np.random.normal(p['centroid'] * 1.8, 300)
            for i in range(13):
                row[f'mfcc_{i}']     = np.random.normal(i * (p['centroid'] / 3000), 15)
            rows.append(row)
    return pd.DataFrame(rows)


# ── 3. Build Dataset ─────────────────────────────────────────────────────────
print("Generating dataset…")
df = generate_synthetic_dataset(n_per_genre=180)
print(f"Dataset: {df.shape} — {df.genre.value_counts().to_dict()}")

le = LabelEncoder()
y  = le.fit_transform(df['genre'])
X  = df.drop('genre', axis=1)

feature_names = X.columns.tolist()

# ── 4. Split + Scale ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 5. Train Classifiers ─────────────────────────────────────────────────────
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'SVM (RBF)':      SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    acc   = (preds == y_test).mean()
    cv    = cross_val_score(clf, X_train_s, y_train, cv=5, n_jobs=-1).mean()
    results[name] = {'clf': clf, 'preds': preds, 'acc': acc, 'cv': cv}
    print(f"\n{name}: Test={acc*100:.1f}%  CV={cv*100:.1f}%")

best_name = max(results, key=lambda k: results[k]['acc'])
best = results[best_name]
print(f"\n🏆 Best: {best_name} ({best['acc']*100:.1f}%)")
print(classification_report(y_test, best['preds'], target_names=le.classes_))

# ── 6. PCA for visualization ──────────────────────────────────────────────────
pca  = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_test_s)

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Music Genre Classification Dashboard', fontsize=15, fontweight='bold')
colors = plt.cm.tab10.colors

# Genre distribution
genre_counts = df['genre'].value_counts()
axes[0,0].bar(genre_counts.index, genre_counts.values,
              color=[colors[i % 10] for i in range(len(genre_counts))])
axes[0,0].set_xticklabels(genre_counts.index, rotation=45, ha='right')
axes[0,0].set_title('Genre Distribution')

# Confusion matrix
cm = confusion_matrix(y_test, best['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[0,1].set_title(f'Confusion Matrix — {best_name}')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].tick_params(axis='y', rotation=0)

# PCA scatter
for i, genre in enumerate(le.classes_):
    mask = y_test == i
    axes[0,2].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], label=genre, alpha=0.6, s=15)
axes[0,2].set_title('PCA — Genre Clusters')
axes[0,2].legend(fontsize=7, ncol=2)

# Model comparison
model_names = list(results.keys())
test_accs = [results[n]['acc'] for n in model_names]
cv_accs   = [results[n]['cv']  for n in model_names]
x = np.arange(len(model_names))
axes[1,0].bar(x - 0.2, test_accs, 0.4, label='Test Acc',  color='steelblue')
axes[1,0].bar(x + 0.2, cv_accs,   0.4, label='CV Acc',    color='coral')
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(model_names, rotation=15)
axes[1,0].set_ylim(0.5, 1.0); axes[1,0].set_title('Model Comparison'); axes[1,0].legend()

# Feature: Tempo by genre
axes[1,1].boxplot([df[df.genre==g]['tempo'].values for g in GENRES], labels=GENRES)
axes[1,1].set_xticklabels(GENRES, rotation=45, ha='right')
axes[1,1].set_title('Tempo Distribution by Genre'); axes[1,1].set_ylabel('BPM')

# Feature importance (RF)
rf_clf = results['Random Forest']['clf']
feat_imp = pd.Series(rf_clf.feature_importances_, index=feature_names).nlargest(12)
feat_imp.plot(kind='barh', ax=axes[1,2], color='teal')
axes[1,2].set_title('Top 12 Feature Importances (RF)')

plt.tight_layout()
plt.savefig('music_genre_results.png', dpi=150)
print("📊 Saved: music_genre_results.png")

# ── 8. Prediction helper ──────────────────────────────────────────────────────
def classify_audio_file(file_path: str) -> str:
    """Classify genre from an audio file path."""
    feats = extract_features(file_path)
    feat_df = pd.DataFrame([feats])
    # align columns
    for col in feature_names:
        if col not in feat_df:
            feat_df[col] = 0.0
    feat_df = feat_df[feature_names]
    X_scaled = scaler.transform(feat_df)
    pred = results[best_name]['clf'].predict(X_scaled)[0]
    return le.inverse_transform([pred])[0]

# Quick demo on synthetic sample
sample = scaler.transform(X_test_s[:1])
demo_pred = le.inverse_transform(results[best_name]['clf'].predict(sample))[0]
demo_true = le.inverse_transform([y_test[0]])[0]
print(f"\n[Demo] True: {demo_true} → Predicted: {demo_pred}")
print("Usage: classify_audio_file('song.wav')")
