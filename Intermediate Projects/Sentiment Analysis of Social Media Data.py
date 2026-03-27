"""
Sentiment Analysis of Social Media Data
Fine-tuned transformer + traditional ML pipeline for tweet sentiment.
Install: pip install transformers torch scikit-learn pandas numpy matplotlib seaborn wordcloud
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── 1. Synthetic Social Media Dataset ────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "I absolutely love {thing}! Best decision ever 🎉",
    "Wow, {thing} is absolutely amazing today! 😍",
    "Feeling so happy about {thing}! Highly recommend",
    "Great experience with {thing}! Will definitely do this again",
    "{thing} totally exceeded my expectations. Incredible! 🌟",
    "Just tried {thing} and I'm obsessed ❤️ Thank you!",
    "{thing} made my day! So grateful 🙌",
    "This is the best {thing} I've ever seen! Outstanding!",
    "Can't stop smiling thanks to {thing} 😊",
    "Shoutout to {thing} — you're awesome!",
]
NEGATIVE_TEMPLATES = [
    "Really disappointed with {thing}. Not worth it at all 😤",
    "{thing} is absolutely terrible. Avoid at all costs!",
    "Had the worst experience with {thing}. So frustrating 😡",
    "I hate {thing}. Complete waste of time and money",
    "{thing} let me down again. Utterly useless.",
    "Stop recommending {thing} to people. It's awful.",
    "{thing} ruined my day. I'm done with this.",
    "Terrible quality from {thing}. Never again.",
    "{thing} is broken and nobody cares to fix it 🤦",
    "Beyond frustrated with {thing}. What a disaster.",
]
NEUTRAL_TEMPLATES = [
    "Just checked out {thing}. It's okay I guess.",
    "{thing} exists. Make of that what you will.",
    "Not sure how I feel about {thing}. Interesting.",
    "Heard about {thing}. Might try it someday.",
    "{thing} is... there. Not much else to say.",
    "Saw {thing} on my timeline. Seems fine.",
    "People keep talking about {thing}.",
    "Tried {thing}. It's neither good nor bad.",
    "{thing} update dropped. Meh.",
    "Reading about {thing}. Still on the fence.",
]
THINGS = ['this product', 'the new app', 'today\'s update', 'the service', 'this feature',
          'the team', 'this restaurant', 'the API', 'the design', 'the support team']

def generate_tweets(n_per_class=600):
    rows = []
    templates = {'positive': POSITIVE_TEMPLATES,
                 'negative': NEGATIVE_TEMPLATES,
                 'neutral':  NEUTRAL_TEMPLATES}
    for label, tmpl_list in templates.items():
        for _ in range(n_per_class):
            tmpl  = np.random.choice(tmpl_list)
            thing = np.random.choice(THINGS)
            text  = tmpl.format(thing=thing)
            # add synthetic noise
            if np.random.random() < 0.2:
                text += f" #{''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 6))}"
            rows.append({'text': text, 'sentiment': label,
                         'likes': np.random.poisson(50 if label=='positive' else 10),
                         'retweets': np.random.poisson(10 if label=='positive' else 2)})
    return pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)

print("Generating tweet dataset…")
df = generate_tweets(n_per_class=700)
print(f"Dataset: {len(df)} tweets | {df.sentiment.value_counts().to_dict()}")

# ── 2. Text Preprocessing ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+', '', text)             # remove URLs
    text = re.sub(r'@\w+', '', text)                # remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)           # strip # from hashtags
    text = re.sub(r'[^\w\s!?.,]', '', text)         # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Emoji-based sentiment signals
df['has_positive_emoji'] = df['text'].str.contains(r'[🎉😍🌟❤️😊🙌]', regex=True, na=False).astype(int)
df['has_negative_emoji'] = df['text'].str.contains(r'[😤😡🤦]', regex=True, na=False).astype(int)
df['word_count']         = df['clean_text'].apply(lambda x: len(x.split()))
df['exclamation_count']  = df['text'].str.count('!')

# ── 3. TF-IDF + Classifiers ───────────────────────────────────────────────────
X_text = df['clean_text']
y      = df['sentiment']

label_order = ['positive', 'neutral', 'negative']
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, stratify=y, random_state=42)

pipelines = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)),
        ('clf',   LogisticRegression(C=5, max_iter=1000, multi_class='multinomial'))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ('clf',   MultinomialNB(alpha=0.1))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)),
        ('clf',   RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
    ]),
}

results = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc   = (preds == y_test).mean()
    results[name] = {'pipe': pipe, 'preds': preds, 'acc': acc}
    print(f"\n{name}: {acc*100:.1f}%")
    print(classification_report(y_test, preds))

best_name = max(results, key=lambda k: results[k]['acc'])
best_pipe = results[best_name]['pipe']
best_pred = results[best_name]['preds']
print(f"\n🏆 Best: {best_name}")

# ── 4. Transformer-based model (if available) ─────────────────────────────────
try:
    from transformers import pipeline as hf_pipeline
    print("\nLoading HuggingFace sentiment model (distilbert)…")
    hf_model = hf_pipeline('sentiment-analysis',
                            model='distilbert-base-uncased-finetuned-sst-2-english')
    sample_texts = df['text'].head(5).tolist()
    for t in sample_texts:
        result = hf_model(t[:512])[0]
        print(f"  [{result['label']} {result['score']:.2f}] {t[:60]}…")
    print("✅ Transformer model working!")
except Exception:
    print("(Transformers not installed — using TF-IDF models)")

# ── 5. Visualizations ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Social Media Sentiment Analysis Dashboard', fontsize=15, fontweight='bold')

# Sentiment distribution
counts = df['sentiment'].value_counts()
axes[0,0].pie(counts, labels=counts.index, autopct='%1.1f%%',
              colors=['#2ecc71','#95a5a6','#e74c3c'], startangle=140)
axes[0,0].set_title('Sentiment Distribution')

# Confusion matrix
cm = confusion_matrix(y_test, best_pred, labels=label_order)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0,1],
            xticklabels=label_order, yticklabels=label_order)
axes[0,1].set_title(f'Confusion Matrix — {best_name}')

# Top tokens per class
tfidf = best_pipe.named_steps['tfidf']
clf   = best_pipe.named_steps['clf']
vocab = np.array(tfidf.get_feature_names_out())
if hasattr(clf, 'coef_'):
    top_n = 8
    class_labels = clf.classes_
    ax_feat = axes[0,2]
    colors_map = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    for i, cls in enumerate(class_labels):
        top_idx  = clf.coef_[i].argsort()[-top_n:]
        top_feat = vocab[top_idx]
        top_val  = clf.coef_[i][top_idx]
        y_pos = np.arange(top_n) + i * (top_n + 1)
        ax_feat.barh(y_pos, top_val, color=colors_map.get(cls, 'gray'), alpha=0.7)
        ax_feat.set_yticks(y_pos)
        ax_feat.set_yticklabels(top_feat, fontsize=7)
    ax_feat.set_title('Top Tokens by Class (LR coefficients)')
else:
    axes[0,2].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
                   ha='center', va='center', transform=axes[0,2].transAxes)
    axes[0,2].set_title('Top Tokens')

# Likes distribution by sentiment
axes[1,0].boxplot([df[df.sentiment==s]['likes'].values for s in label_order], labels=label_order)
axes[1,0].set_title('Likes by Sentiment'); axes[1,0].set_ylabel('Likes')

# Model comparison
model_names = list(results.keys())
accs = [results[n]['acc'] for n in model_names]
axes[1,1].barh(model_names, accs, color=['steelblue','coral','teal'])
axes[1,1].set_xlim(0.5, 1.0); axes[1,1].set_title('Model Accuracy Comparison')
for i, v in enumerate(accs):
    axes[1,1].text(v + 0.005, i, f'{v*100:.1f}%', va='center')

# Word length distribution
axes[1,2].hist(df[df.sentiment=='positive']['word_count'], bins=20, alpha=0.6, label='Positive', color='green')
axes[1,2].hist(df[df.sentiment=='negative']['word_count'], bins=20, alpha=0.6, label='Negative', color='red')
axes[1,2].hist(df[df.sentiment=='neutral']['word_count'],  bins=20, alpha=0.6, label='Neutral',  color='gray')
axes[1,2].set_title('Word Count Distribution'); axes[1,2].legend()

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png', dpi=150)
print("📊 Saved: sentiment_analysis_results.png")

# ── 6. Prediction API ─────────────────────────────────────────────────────────
def analyze_sentiment(texts: list[str]) -> pd.DataFrame:
    """
    Analyze sentiment for a list of texts.
    Returns DataFrame with text, sentiment, confidence.
    """
    cleaned = [clean_text(t) for t in texts]
    preds   = best_pipe.predict(cleaned)
    if hasattr(best_pipe.named_steps['clf'], 'predict_proba'):
        probs = best_pipe.predict_proba(cleaned).max(axis=1)
    else:
        probs = np.ones(len(texts))
    return pd.DataFrame({'text': texts, 'sentiment': preds, 'confidence': probs})

# Demo
print("\n── Live Demo ──")
demo_tweets = [
    "This product is incredible! Changed my life completely! 🎉",
    "Absolutely hate this service. Never using it again 😡",
    "Just saw the announcement. Interesting I guess.",
    "Customer support helped me immediately. Amazing team! 😍",
    "The app crashed three times today. Unacceptable.",
]
results_df = analyze_sentiment(demo_tweets)
for _, row in results_df.iterrows():
    emoji = {'positive': '✅', 'negative': '❌', 'neutral': '➖'}[row.sentiment]
    print(f"  {emoji} [{row.sentiment:8s} {row.confidence:.2f}] {row.text[:60]}")
