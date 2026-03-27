"""
Conversational AI Chatbot
Full-featured chatbot with intent classification, entity extraction,
context management, sentiment awareness, and multi-turn dialogue.
Install: pip install scikit-learn numpy pandas nltk transformers torch (optional)
"""

import re
import json
import time
import random
import numpy as np
from datetime import datetime
from collections import deque, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── 1. Intent & Entity Definitions ───────────────────────────────────────────
INTENTS = {
    'greeting':       ['hi', 'hello', 'hey', 'howdy', 'good morning', 'good afternoon',
                        'good evening', 'what\'s up', 'greetings', 'yo'],
    'farewell':       ['bye', 'goodbye', 'see you', 'later', 'farewell', 'take care',
                        'good night', 'cya', 'adios', 'ttyl'],
    'weather':        ['weather', 'temperature', 'rain', 'sunny', 'forecast', 'humid',
                        'cold', 'hot', 'climate', 'snow', 'wind'],
    'time_date':      ['time', 'date', 'day', 'today', 'current time', 'what time',
                        'what day', 'hour', 'clock', 'calendar'],
    'help':           ['help', 'assist', 'support', 'how do i', 'can you', 'what can you',
                        'guide', 'instructions', 'confused', 'stuck'],
    'joke':           ['joke', 'funny', 'laugh', 'humor', 'make me laugh', 'tell me a joke',
                        'something funny', 'comedy', 'pun'],
    'math':           ['calculate', 'compute', 'solve', 'math', 'plus', 'minus', 'multiply',
                        'divide', 'sum', 'equation', 'percentage', 'square root'],
    'general_qa':     ['what is', 'who is', 'why', 'when', 'where', 'how', 'explain',
                        'define', 'tell me about', 'describe'],
    'recommendation': ['recommend', 'suggest', 'best', 'top', 'good', 'should i',
                        'which one', 'advice', 'opinion'],
    'small_talk':     ['how are you', 'how\'s it going', 'what\'s new', 'feeling',
                        'doing well', 'your name', 'who are you', 'are you human'],
    'thanks':         ['thank', 'thanks', 'thank you', 'appreciate', 'grateful', 'thx', 'ty'],
    'complaint':      ['problem', 'issue', 'error', 'broken', 'not working', 'bug',
                        'fail', 'wrong', 'bad', 'terrible', 'disappointed'],
    'affirmation':    ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'absolutely',
                        'correct', 'right', 'definitely'],
    'negation':       ['no', 'nope', 'nah', 'not really', 'never', 'disagree', 'incorrect'],
}

JOKES = [
    "Why don't scientists trust atoms? Because they make up everything! 😄",
    "What do you call a fake noodle? An impasta! 🍝",
    "Why did the math book look so sad? Because it had too many problems.",
    "I told my computer I needed a break… now it won't stop sending me vacation ads.",
    "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
    "What's a computer's favorite snack? Microchips! 💻",
    "Why did the AI go to therapy? It had too many neural issues.",
    "How many programmers does it take to change a light bulb? None — that's a hardware problem.",
]

TOPICS_KB = {
    'machine learning': "Machine Learning is a subset of AI where systems learn from data to make predictions or decisions without being explicitly programmed.",
    'neural network':   "Neural networks are ML models inspired by the human brain, consisting of layers of interconnected nodes that process information.",
    'python':           "Python is a versatile programming language widely used in data science, AI, web development, and automation.",
    'ai':               "Artificial Intelligence (AI) is the simulation of human intelligence by machines, enabling them to perform tasks like reasoning, learning, and problem-solving.",
    'blockchain':       "Blockchain is a decentralized, distributed ledger technology that records transactions securely across many computers.",
    'cloud computing':  "Cloud computing delivers computing services (servers, storage, databases, networking, software) over the internet on-demand.",
}

RECOMMENDATIONS = {
    'movie':  ['Inception', 'Interstellar', 'The Matrix', 'Parasite', 'Get Out'],
    'book':   ['Atomic Habits', 'Sapiens', 'The Pragmatic Programmer', 'Clean Code'],
    'food':   ['Try something new today — perhaps a local cuisine you haven\'t explored!'],
    'music':  ['Jazz for focus, Lo-fi for relaxing, or classical for studying — depends on your mood!'],
    'python library': ['NumPy, Pandas, Scikit-learn, TensorFlow, FastAPI — all excellent choices!'],
}

# ── 2. Intent Classifier ──────────────────────────────────────────────────────
class IntentClassifier:
    def __init__(self):
        # Build training data from intent keywords
        X_train, y_train = [], []
        for intent, phrases in INTENTS.items():
            for phrase in phrases:
                X_train.append(phrase)
                y_train.append(intent)
                # augment
                X_train.append(f"I want to know about {phrase}")
                y_train.append(intent)

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), analyzer='word')),
            ('clf',   LogisticRegression(C=5, max_iter=500, multi_class='multinomial'))
        ])
        self.pipeline.fit(X_train, y_train)

    def predict(self, text: str) -> tuple[str, float]:
        text_clean = text.lower().strip()
        probs = self.pipeline.predict_proba([text_clean])[0]
        top_idx = probs.argmax()
        return self.pipeline.classes_[top_idx], probs[top_idx]

# ── 3. Entity Extractor ───────────────────────────────────────────────────────
class EntityExtractor:
    PATTERNS = {
        'number':    r'\b\d+(?:\.\d+)?\b',
        'city':      r'\b(london|new york|paris|tokyo|delhi|mumbai|sydney|berlin|dubai)\b',
        'topic':     r'\b(ml|machine learning|ai|python|blockchain|neural network|cloud computing|nlp)\b',
        'rec_topic': r'\b(movie|book|music|food|python library|game|restaurant)\b',
        'math_expr': r'[\d\s\+\-\*\/\(\)\^\.]+(?:=\?|$)',
        'time_ref':  r'\b(today|tomorrow|yesterday|this week|next week|monday|tuesday|wednesday|thursday|friday)\b',
    }

    def extract(self, text: str) -> dict:
        entities = {}
        t = text.lower()
        for ent_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, t)
            if matches:
                entities[ent_type] = matches
        return entities

# ── 4. Simple Math Solver ─────────────────────────────────────────────────────
def solve_math(text: str) -> str:
    """Safely evaluate simple math expressions from text."""
    expr = re.sub(r'[^\d\s\+\-\*\/\(\)\.\%\^]', '', text)
    expr = expr.replace('^', '**')
    expr = expr.strip()
    if not expr:
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return f"🔢 `{expr}` = **{result}**"
    except Exception:
        return None

# ── 5. Sentiment Detector (rule-based) ────────────────────────────────────────
POSITIVE_WORDS = {'great','good','awesome','love','excellent','happy','fantastic','perfect','amazing','wonderful'}
NEGATIVE_WORDS = {'bad','terrible','awful','hate','horrible','sad','angry','frustrated','broken','fail','worst'}

def detect_sentiment(text: str) -> str:
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos > neg: return 'positive'
    if neg > pos: return 'negative'
    return 'neutral'

# ── 6. Context Manager ────────────────────────────────────────────────────────
class ConversationContext:
    def __init__(self, max_history=10):
        self.history   = deque(maxlen=max_history)
        self.user_name = None
        self.topic     = None
        self.turn      = 0
        self.sentiment_history = deque(maxlen=5)
        self.entity_memory = defaultdict(list)

    def add_turn(self, user_msg, bot_msg, intent, entities, sentiment):
        self.history.append({
            'turn': self.turn, 'user': user_msg, 'bot': bot_msg,
            'intent': intent, 'entities': entities, 'sentiment': sentiment,
        })
        self.turn += 1
        self.sentiment_history.append(sentiment)
        for k, v in entities.items():
            self.entity_memory[k].extend(v)

    def get_dominant_sentiment(self) -> str:
        if not self.sentiment_history: return 'neutral'
        from collections import Counter
        return Counter(self.sentiment_history).most_common(1)[0][0]

    def get_last_intent(self) -> str | None:
        return self.history[-1]['intent'] if self.history else None

    def summary(self) -> str:
        return (f"Turn {self.turn} | "
                f"Dominant mood: {self.get_dominant_sentiment()} | "
                f"Topics discussed: {list(set(self.entity_memory.get('topic', [])))}")

# ── 7. Response Generator ─────────────────────────────────────────────────────
class ResponseGenerator:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.extractor  = EntityExtractor()
        self.context    = ConversationContext()
        self.fallback_count = 0

    def _format(self, msg: str) -> str:
        return f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 {msg}"

    def generate(self, user_msg: str) -> str:
        intent, confidence = self.classifier.predict(user_msg)
        entities  = self.extractor.extract(user_msg)
        sentiment = detect_sentiment(user_msg)
        response  = self._craft_response(user_msg, intent, confidence, entities, sentiment)
        self.context.add_turn(user_msg, response, intent, entities, sentiment)
        return self._format(response)

    def _craft_response(self, text, intent, conf, entities, sentiment) -> str:
        t = text.lower().strip()

        # ── Name extraction
        name_match = re.search(r"(?:my name is|i(?:'m| am|'m called)) (\w+)", t)
        if name_match:
            self.context.user_name = name_match.group(1).capitalize()
            return f"Nice to meet you, {self.context.user_name}! 😊 How can I help?"

        name = self.context.user_name
        greet = f" {name}!" if name else "!"

        # ── Sentiment acknowledgment
        sentiment_prefix = ""
        if sentiment == 'negative' and intent != 'joke':
            sentiment_prefix = "I'm sorry to hear you're feeling that way. "
        elif sentiment == 'positive':
            sentiment_prefix = random.choice(["", "Great! ", "Awesome! "])

        # ── Intent handling
        if intent == 'greeting':
            hour = datetime.now().hour
            time_greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
            return f"{time_greet}{greet} I'm ChatBot AI. Ask me anything — weather, math, jokes, or just chat! 🌟"

        elif intent == 'farewell':
            return f"Goodbye{greet} It was great talking with you! Take care! 👋"

        elif intent == 'time_date':
            now = datetime.now()
            return f"🕐 It's currently **{now.strftime('%I:%M %p')}** on **{now.strftime('%A, %B %d, %Y')}**."

        elif intent == 'joke':
            return random.choice(JOKES)

        elif intent == 'thanks':
            return random.choice([
                f"You're welcome{greet} 😊",
                "Happy to help! Let me know if you need anything else.",
                "Anytime! That's what I'm here for. 🌟",
            ])

        elif intent == 'math':
            result = solve_math(text)
            if result:
                return f"Sure! {result}"
            return "I can help with basic math! Try something like '15 * 4 + 7' or '100 / 5'."

        elif intent == 'weather':
            city = entities.get('city', ['your area'])[0].capitalize()
            return (f"🌤️ I'd love to give you the weather for **{city}**! "
                    f"In a real deployment, I'd call a weather API here. "
                    f"For now, let's say it's a lovely day — but always carry an umbrella! ☂️")

        elif intent == 'help':
            caps = [
                "📅 **Date & Time** — 'What time is it?'",
                "🔢 **Math** — 'Calculate 25 * 4'",
                "😄 **Jokes** — 'Tell me a joke'",
                "🌤️ **Weather** — 'What's the weather in London?'",
                "📚 **Knowledge** — 'What is machine learning?'",
                "💡 **Recommendations** — 'Suggest a movie'",
                "💬 **Just chat** — I'm great at small talk too!",
            ]
            return "Here's what I can do:\n" + "\n".join(caps)

        elif intent == 'general_qa':
            for keyword, desc in TOPICS_KB.items():
                if keyword in t:
                    return f"📚 **{keyword.title()}**: {desc}"
            return (f"That's an interesting question! In a full deployment, "
                    f"I'd query a knowledge base or search the web. "
                    f"Could you be more specific about what you'd like to know?")

        elif intent == 'recommendation':
            for category, items in RECOMMENDATIONS.items():
                if category in t:
                    rec = random.choice(items) if isinstance(items, list) else items
                    return f"💡 **{category.title()} Recommendation**: {rec}"
            return "I'd love to make a recommendation! What category? (movies, books, music, food, Python libraries)"

        elif intent == 'small_talk':
            if 'name' in t or 'who are you' in t:
                return "I'm ChatBot AI 🤖 — an intelligent conversational assistant built with ML!"
            if 'how are you' in t or 'doing' in t:
                moods = ['fantastic', 'great', 'excellent', 'wonderful']
                return f"I'm feeling {random.choice(moods)}! Ready to help you with anything. How about you?"
            if 'human' in t or 'real' in t:
                return "I'm an AI, but I promise I'm a very friendly one! 😄"
            return "I love chatting! Ask me something or just talk — I'm all ears. 👂"

        elif intent == 'complaint':
            return (f"{sentiment_prefix}I'm sorry you're experiencing issues! "
                    f"Here's what I suggest:\n"
                    f"1. Try refreshing or restarting\n"
                    f"2. Check our documentation\n"
                    f"3. Contact support with error details\n"
                    f"Would you like to describe the problem in more detail?")

        elif intent == 'affirmation':
            last = self.context.get_last_intent()
            return f"Great! {'Let\'s continue then.' if last else 'What would you like to do?'} 😊"

        elif intent == 'negation':
            return "No problem at all! Is there something else I can help you with?"

        else:
            self.fallback_count += 1
            fallbacks = [
                "That's interesting! Can you tell me more?",
                "I'm not sure I fully understood. Could you rephrase?",
                "Hmm, I'm still learning! Try asking about math, weather, jokes, or tech topics.",
                f"I heard you say '{text[:50]}' — could you clarify what you need?",
            ]
            return fallbacks[self.fallback_count % len(fallbacks)]

# ── 8. Chatbot CLI ────────────────────────────────────────────────────────────
def run_chatbot(demo_mode=False):
    bot = ResponseGenerator()
    print("\n" + "="*60)
    print("  🤖  ChatBot AI  —  ML-Powered Conversational Assistant")
    print("="*60)
    print("  Type 'quit' to exit | 'stats' for session stats")
    print("="*60 + "\n")
    print(bot.generate("hello"))

    if demo_mode:
        demo_inputs = [
            "My name is Alex",
            "What time is it?",
            "Tell me a joke",
            "Calculate 256 / 16 + 8",
            "What is machine learning?",
            "Recommend a movie",
            "What's the weather in London?",
            "how are you?",
            "thanks a lot!",
            "bye",
        ]
        print("\n[Running in DEMO mode]\n")
        for user_input in demo_inputs:
            print(f"👤 You: {user_input}")
            response = bot.generate(user_input)
            print(response)
            print()
            time.sleep(0.1)
        return bot

    # Interactive mode
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🤖 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(bot.generate("goodbye"))
            break
        if user_input.lower() == 'stats':
            print(f"\n📊 Session: {bot.context.summary()}\n")
            continue

        response = bot.generate(user_input)
        print(response)

    return bot

# ── 9. Analytics & Visualization ─────────────────────────────────────────────
def visualize_chatbot_analytics(bot: ResponseGenerator):
    history = list(bot.context.history)
    if not history:
        return

    intents    = [h['intent'] for h in history]
    sentiments = [h['sentiment'] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Chatbot Session Analytics', fontsize=14, fontweight='bold')

    from collections import Counter
    # Intent distribution
    intent_counts = Counter(intents)
    axes[0].barh(list(intent_counts.keys()), list(intent_counts.values()), color='steelblue')
    axes[0].set_title('Intent Distribution')

    # Sentiment distribution
    sentiment_counts = Counter(sentiments)
    colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    axes[1].pie(sentiment_counts.values(),
                labels=sentiment_counts.keys(),
                colors=[colors[s] for s in sentiment_counts.keys()],
                autopct='%1.0f%%')
    axes[1].set_title('Sentiment Distribution')

    # Conversation length over time
    turn_lengths = [len(h['user'].split()) for h in history]
    axes[2].plot(range(len(turn_lengths)), turn_lengths, 'o-', color='teal')
    axes[2].set_title('User Message Length per Turn')
    axes[2].set_xlabel('Turn'); axes[2].set_ylabel('Word Count')

    plt.tight_layout()
    plt.savefig('chatbot_analytics.png', dpi=150)
    print("📊 Saved: chatbot_analytics.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    demo = '--demo' in sys.argv or True  # default to demo mode
    bot = run_chatbot(demo_mode=demo)
    visualize_chatbot_analytics(bot)
    print(f"\n✅ Session ended. {bot.context.turn} turns | "
          f"Fallbacks: {bot.fallback_count}")
