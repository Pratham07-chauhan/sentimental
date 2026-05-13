import streamlit as st
import nltk
import streamlit as st
import nltk
import os

# NLTK data download karo — Cloud deploy ke liye zaroori!
def download_nltk():
    packages = [
        'punkt', 'punkt_tab', 'stopwords',
        'wordnet', 'vader_lexicon',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    for p in packages:
        nltk.download(p, quiet=True)

download_nltk()

import pandas as pd
import re
import string
# ... baaki imports same rahenge
import pandas as pd
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Bilingual Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
    text-align: center; border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}
.main-header h1 {
    font-family: 'Poppins', sans-serif; font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #e94560, #7b68ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -1px;
}
.main-header p { color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-top: 0.5rem; letter-spacing: 2px; text-transform: uppercase; }
.stat-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
    padding: 1.2rem; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.stat-number { font-family: 'Poppins', sans-serif; font-size: 2rem; font-weight: 700; color: #e94560; }
.stat-label { color: rgba(255,255,255,0.5); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
.result-positive {
    background: linear-gradient(135deg, #0a4a2e, #0d6b3f); border: 1px solid #22c55e;
    border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 0 40px rgba(34,197,94,0.2);
}
.result-negative {
    background: linear-gradient(135deg, #4a0a0a, #6b0d0d); border: 1px solid #ef4444;
    border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 0 40px rgba(239,68,68,0.2);
}
.result-neutral {
    background: linear-gradient(135deg, #1a1a2e, #2a2a4e); border: 1px solid #7b68ee;
    border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 0 40px rgba(123,104,238,0.2);
}
.result-positive h1 { font-family:'Poppins',sans-serif; font-size:3rem; font-weight:800; margin:0.5rem 0; color:#22c55e; }
.result-negative h1 { font-family:'Poppins',sans-serif; font-size:3rem; font-weight:800; margin:0.5rem 0; color:#ef4444; }
.result-neutral h1 { font-family:'Poppins',sans-serif; font-size:3rem; font-weight:800; margin:0.5rem 0; color:#7b68ee; }
.result-positive p, .result-negative p, .result-neutral p { color:rgba(255,255,255,0.7); font-size:0.85rem; letter-spacing:1px; text-transform:uppercase; }
.section-header {
    font-family:'Poppins',sans-serif; font-size:1rem; font-weight:600; color:#e94560;
    text-transform:uppercase; letter-spacing:2px; margin:1.5rem 0 1rem 0;
    padding-bottom:0.5rem; border-bottom:2px solid rgba(233,69,96,0.3);
}
.info-box {
    background: linear-gradient(135deg, #0f3460, #533483); border-radius:10px;
    padding:0.8rem 1.2rem; margin:0.5rem 0; border-left:3px solid #e94560;
    color:rgba(255,255,255,0.85); font-size:0.9rem;
}
.sidebar-info {
    background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius:10px;
    padding:1rem; margin:0.5rem 0; border:1px solid rgba(255,255,255,0.08);
    font-size:0.82rem; color:rgba(255,255,255,0.7);
}
.confidence-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 12px; padding: 1.2rem; margin-top: 1rem;
    border: 1px solid rgba(255,255,255,0.08); text-align: center;
}
.vader-box {
    background: linear-gradient(135deg, #0f1a3e, #1a2a5e);
    border-radius: 12px; padding: 1.2rem; margin-top: 0.8rem;
    border: 1px solid rgba(123,104,238,0.3);
}
[data-testid="stSidebar"] { background: #0d0d1a !important; }
.stTabs [data-baseweb="tab-list"] { background:#1a1a2e; border-radius:10px; padding:0.3rem; }
.stTabs [data-baseweb="tab"] { color:rgba(255,255,255,0.6); font-weight:500; }
.stTabs [aria-selected="true"] { background:#e94560 !important; color:white !important; border-radius:8px !important; }
.stButton button {
    background: linear-gradient(135deg, #e94560, #c23152) !important; color:white !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def contractions():
    return {
        "ain't":"is not","aren't":"are not","can't":"cannot","couldn't":"could not",
        "didn't":"did not","doesn't":"does not","don't":"do not","hadn't":"had not",
        "hasn't":"has not","haven't":"have not","he's":"he is","i'm":"I am",
        "isn't":"is not","it's":"it is","i've":"I have","let's":"let us",
        "shouldn't":"should not","that's":"that is","there's":"there is",
        "they're":"they are","wasn't":"was not","we're":"we are","weren't":"were not",
        "what's":"what is","who's":"who is","won't":"will not","wouldn't":"would not",
        "you're":"you are","you've":"you have","u":"you","ur":"your","lol":"laugh",
        "omg":"oh my god","wtf":"angry","stfu":"angry","ya":"yes","yeah":"yes",
        "gonna":"going to","wanna":"want to","gotta":"got to","idk":"i do not know",
        "ily":"i love you","ihy":"i hate you","luv":"love","sux":"sucks",
        "tmr":"tomorrow","k":"okay","da":"the","yo":"greet","hey":"greet",
        "lmao":"laugh","rofl":"laugh","y":"why","wut":"what","wat":"what",
        "awww":"amazement","aww":"amazement","ugh":"sad","ughh":"sad",
    }

def emoticons():
    return {
        ":)":"smiley",":-)":" smiley",":D":"smiley","XD":"smiley",
        ":(":" sad",":-(":" sad",":/":" sad",":'(":" sad",
        "<3":"love",";)":"playful",":P":"playful","-_-":"angry"
    }

def lemmatization(sent):
    lemmatize = WordNetLemmatizer()
    sentence_after_lemmatization = []
    for word, tag in pos_tag(word_tokenize(sent)):
        if tag[0:2] == "NN": pos = 'n'
        elif tag[0:2] == "VB": pos = 'v'
        else: pos = 'a'
        lem = lemmatize.lemmatize(word, pos)
        sentence_after_lemmatization.append(lem)
    st_text = ""
    for i in sentence_after_lemmatization:
        if i != "be" and i != "is" and len(i) != 1:
            st_text = st_text + " " + i
    c = 0
    list_text = st_text.split()
    flag = 0
    new_st = ""
    for i in list_text:
        temp = i
        if flag == 1:
            flag = 0
            continue
        if i == "not" and (c + 1) < len(list_text):
            for syn in wordnet.synsets(list_text[c + 1]):
                antonyms = []
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                        temp = antonyms[0]
                        flag = 1
                        break
                if flag == 1:
                    break
        new_st = new_st + " " + temp
        c += 1
    return new_st

def removal_of_noise(sent):
    clean_sent = []
    temp_st = ""
    list_sent = sent.split(" ")
    c = 0
    d = contractions()
    emoji = emoticons()
    for word in list_sent:
        word = re.sub(r"http\S+", "", word)
        word = re.sub(r"[www.][a-zA-Z0-9_]+[.com]", "", word)
        word = re.sub("(@[A-Za-z0-9_]+)", "", word)
        if word in emoji.keys(): word = emoji[word]
        if word.lower() in d.keys(): word = d[word.lower()]
        if c == 0: temp_st = word
        else: temp_st = temp_st + " " + word
        c += 1
    sent = temp_st
    stop_words = set(stopwords.words('english'))
    stop_words.add('is')
    stop_words.remove('not')
    for word in word_tokenize(sent):
        if word.lower() not in stop_words and word.lower() not in string.punctuation and word != "'" and word != '"':
            word = word.lower()
            word = re.sub("[0-9]+", "", word)
            word = re.sub("[.]+", " ", word)
            word = re.sub("[-]+", " ", word)
            word = re.sub("[_]+", " ", word)
            word = re.sub("~", " ", word)
            if len(word) != 1:
                clean_sent.append(word.lower())
    cleaned_st = ""
    for i in clean_sent:
        cleaned_st = cleaned_st + " " + i
    return lemmatization(cleaned_st)

def start(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.replace("\u2019", "'")
    new_text = sent_tokenize(text)
    new_str = ""
    for i in new_text:
        j = deEmojify(i)
        res = removal_of_noise(j)
        new_str = new_str + " " + res
    return new_str

def list_to_dict(words_list):
    return dict([(word, True) for word in words_list])

def get_confidence(votes, result):
    matching_votes = votes.count(result)
    total_votes = len(votes)
    confidence = (matching_votes / total_votes) * 100
    return round(confidence, 1)

def get_confidence_level(confidence):
    if confidence >= 80:
        return "Very High", "#22c55e", "🟢"
    elif confidence >= 67:
        return "High", "#eab308", "🟡"
    else:
        return "Medium", "#f97316", "🟠"

def get_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        vader_label = "positive"
    elif compound <= -0.05:
        vader_label = "negative"
    else:
        vader_label = "neutral"
    return vader_label, score

def get_final_sentiment(hybrid_result, vader_result):
    # VADER neutral → Final = neutral
    # VADER positive/negative → Final = Hybrid result
    if vader_result == "neutral":
        return "neutral"
    else:
        return hybrid_result

def analyze_clauses(text):
    # Sentence ko clauses mein todo
    import re
    # Conjunctions pe split karo
    conjunctions = [' but ', ' however ', ' although ', ' though ', 
                   ' yet ', ' while ', ' whereas ', ' and ', ' or ']
    
    clauses = [text]
    for conj in conjunctions:
        new_clauses = []
        for clause in clauses:
            parts = clause.split(conj)
            if len(parts) > 1:
                new_clauses.extend(parts)
            else:
                new_clauses.append(clause)
        clauses = new_clauses
    
    # Filter empty clauses
    clauses = [c.strip() for c in clauses if len(c.strip()) > 3]
    return clauses

def get_clause_sentiments(clauses):
    sia = SentimentIntensityAnalyzer()
    results = []
    for clause in clauses:
        score = sia.polarity_scores(clause)
        compound = score['compound']
        if compound >= 0.05:
            sentiment = "POSITIVE"
            color = "#22c55e"
            icon = "😊"
        elif compound <= -0.05:
            sentiment = "NEGATIVE"
            color = "#ef4444"
            icon = "😔"
        else:
            sentiment = "NEUTRAL"
            color = "#7b68ee"
            icon = "😐"
        results.append({
            "clause": clause,
            "sentiment": sentiment,
            "color": color,
            "icon": icon,
            "compound": round(compound, 3)
        })
    return results

@st.cache_resource
def load_and_train():
    from nltk.classify.scikitlearn import SklearnClassifier
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC
    from nltk.classify import ClassifierI
    from statistics import mode
    from sklearn.model_selection import train_test_split

    data = pd.read_csv("test_data.csv", skip_blank_lines=True, encoding="latin")
    l = ["negative" if i == 0 else "positive" for i in data["ï»¿Label"]]
    data['label'] = l
    data = data.drop(columns=['number', 'date', 'name', 'no_query', 'ï»¿Label'])

    with open('cleaned_tweet.txt', 'r') as f:
        lines = f.read().splitlines()
    data["cleaned_tweets"] = lines

    with open('english-adjectives.txt') as f:
        adjectives = f.read().splitlines()

    all_words = []
    for i in data["cleaned_tweets"]:
        for word in word_tokenize(i):
            if word in adjectives or word == "not":
                all_words.append(word)

    BagOfWords = nltk.FreqDist(all_words)
    word_features = list(BagOfWords.keys())[:5000]

    new_list = []
    for i in data["cleaned_tweets"]:
        st_str = " ".join([j for j in i.split() if j in word_features])
        new_list.append(st_str)
    data["cleaned_tweets"] = new_list

    y = data["label"]
    x = data.drop('label', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    X_train = pd.concat([pd.DataFrame(columns=['Tweet', 'cleaned_tweets']), x_train])
    X_test = pd.concat([pd.DataFrame(columns=['Tweet', 'cleaned_tweets']), x_test])
    Y_train = list(y_train)
    Y_test = list(y_test)

    training_set_formatted = [(list_to_dict(i.split()), Y_train[idx]) for idx, i in enumerate(X_train["cleaned_tweets"])]
    test_set_formatted = [(list_to_dict(i.split()), Y_test[idx]) for idx, i in enumerate(X_test["cleaned_tweets"])]

    classifiers = []
    accuracy = []

    nb = nltk.NaiveBayesClassifier.train(training_set_formatted)
    classifiers.append([nb, "Naive Bayes"])
    accuracy.append([nltk.classify.accuracy(nb, test_set_formatted) * 100, "Naive Bayes"])

    mnb = SklearnClassifier(MultinomialNB()); mnb.train(training_set_formatted)
    classifiers.append([mnb, "Multinomial NB"])
    accuracy.append([nltk.classify.accuracy(mnb, test_set_formatted) * 100, "Multinomial NB"])

    bnb = SklearnClassifier(BernoulliNB()); bnb.train(training_set_formatted)
    classifiers.append([bnb, "Bernoulli NB"])
    accuracy.append([nltk.classify.accuracy(bnb, test_set_formatted) * 100, "Bernoulli NB"])

    lr = SklearnClassifier(LogisticRegression()); lr.train(training_set_formatted)
    classifiers.append([lr, "Logistic Regression"])
    accuracy.append([nltk.classify.accuracy(lr, test_set_formatted) * 100, "Logistic Regression"])

    sgd = SklearnClassifier(SGDClassifier()); sgd.train(training_set_formatted)
    classifiers.append([sgd, "SGD Classifier"])
    accuracy.append([nltk.classify.accuracy(sgd, test_set_formatted) * 100, "SGD Classifier"])

    svc = SklearnClassifier(SVC()); svc.train(training_set_formatted)
    classifiers.append([svc, "SVM Classifier"])
    accuracy.append([nltk.classify.accuracy(svc, test_set_formatted) * 100, "SVM Classifier"])

    class EnsembleClassifier(ClassifierI):
        def __init__(self, *cls): self._classifiers = cls
        def classify(self, features):
            votes = [c.classify(features) for c in self._classifiers]
            try:
                from statistics import mode
                return mode(votes)
            except: return self._classifiers[3].classify(features)

    ensemble = EnsembleClassifier(*[c[0] for c in classifiers])
    preds = [ensemble.classify(f[0]) for f in test_set_formatted]
    correct = sum(1 for i in range(len(preds)) if preds[i] == Y_test[i])
    accuracy.append([100 * correct / len(preds), "Hybrid Model"])

    # Precision, Recall, F1, Confusion Matrix calculate karo
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
    
    metrics_data = []
    all_preds = {}
    
    clf_objects = [nb, mnb, bnb, lr, sgd, svc]
    clf_names = ["Naive Bayes", "Multinomial NB", "Bernoulli NB", "Logistic Regression", "SGD Classifier", "SVM Classifier"]
    
    for clf_obj, clf_name in zip(clf_objects, clf_names):
        y_pred = [clf_obj.classify(f[0]) for f in test_set_formatted]
        all_preds[clf_name] = y_pred
        
        precision = precision_score(Y_test, y_pred, pos_label='positive', average='weighted', zero_division=0) * 100
        recall = recall_score(Y_test, y_pred, pos_label='positive', average='weighted', zero_division=0) * 100
        f1 = f1_score(Y_test, y_pred, pos_label='positive', average='weighted', zero_division=0) * 100
        acc = sum(1 for a, b in zip(Y_test, y_pred) if a == b) / len(Y_test) * 100
        cm = confusion_matrix(Y_test, y_pred, labels=['positive', 'negative'])
        
        metrics_data.append({
            'name': clf_name,
            'accuracy': round(acc, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1': round(f1, 2),
            'confusion_matrix': cm
        })
    
    # Hybrid metrics
    hybrid_preds = [ensemble.classify(f[0]) for f in test_set_formatted]
    h_precision = precision_score(Y_test, hybrid_preds, pos_label='positive', average='weighted', zero_division=0) * 100
    h_recall = recall_score(Y_test, hybrid_preds, pos_label='positive', average='weighted', zero_division=0) * 100
    h_f1 = f1_score(Y_test, hybrid_preds, pos_label='positive', average='weighted', zero_division=0) * 100
    h_acc = correct / len(preds) * 100
    h_cm = confusion_matrix(Y_test, hybrid_preds, labels=['positive', 'negative'])
    
    metrics_data.append({
        'name': 'Hybrid Model',
        'accuracy': round(h_acc, 2),
        'precision': round(h_precision, 2),
        'recall': round(h_recall, 2),
        'f1': round(h_f1, 2),
        'confusion_matrix': h_cm
    })

    return classifiers, accuracy, adjectives, word_features, metrics_data, Y_test

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.5rem 0;'>
        <div style='font-size:3.5rem;'>🧠</div>
        <div style='font-family:Poppins,sans-serif; font-weight:700; font-size:1.15rem; color:white; margin-top:0.5rem;'>Sentiment Analyzer</div>
        <div style='color:rgba(255,255,255,0.35); font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;'>Final Year NLP Project</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.08);'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#e94560;font-weight:600;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;'>📌 About Project</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sidebar-info'>Analyzes sentiment of <b>Hindi-English code-mixed</b> social media text using an ensemble of 6 ML classifiers combined with <b>VADER</b> for neutral detection.</div>""", unsafe_allow_html=True)

    st.markdown("<div style='color:#e94560;font-weight:600;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem;'>🤖 Classifiers</div>", unsafe_allow_html=True)
    for clf in ["Naive Bayes", "Multinomial NB", "Bernoulli NB", "Logistic Regression", "SGD Classifier", "SVM Classifier", "VADER (Neutral)"]:
        st.markdown(f"<div style='color:rgba(255,255,255,0.65);font-size:0.82rem;padding:0.25rem 0;border-bottom:1px solid rgba(255,255,255,0.05);'>▸ {clf}</div>", unsafe_allow_html=True)

    st.markdown("<div style='color:#e94560;font-weight:600;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem;'>💡 Sample Inputs</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sidebar-info'>
    <b>Positive:</b><br>"life is beautiful and wonderful"<br><br>
    <b>Negative:</b><br>"I am so sad and depressed"<br><br>
    <b>Neutral:</b><br>"I went to the market today"<br><br>
    <b>Hinglish:</b><br>"aaj ka din bahut amazing tha"
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='color:#e94560;font-weight:600;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem;'>📊 Dataset Info</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sidebar-info'>Source: <b>Sentiment140</b><br>Total tweets: <b>40,000</b><br>Train split: <b>85%</b><br>Test split: <b>15%</b><br>Labels: Positive / Negative / Neutral</div>""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class='main-header'>
    <h1>🧠 Bilingual Sentiment Analysis</h1>
    <p>Hindi · English · Code-Mixed · NLP · Machine Learning · Ensemble + VADER</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("⚙️ Loading and training all models.."):
    classifiers, accuracy, adjectives, word_features, metrics_data, Y_test = load_and_train()

st.success("✅ All 6 classifiers + VADER loaded and ready!")
st.markdown("<br>", unsafe_allow_html=True)

# Stats
c1, c2, c3, c4 = st.columns(4)
best_acc = round(max([a[0] for a in accuracy]), 1)
with c1: st.markdown("""<div class='stat-card'><div class='stat-number'>40K</div><div class='stat-label'>Training Tweets</div></div>""", unsafe_allow_html=True)
with c2: st.markdown("""<div class='stat-card'><div class='stat-number'>6+1</div><div class='stat-label'>ML + VADER</div></div>""", unsafe_allow_html=True)
with c3: st.markdown(f"""<div class='stat-card'><div class='stat-number'>{best_acc}%</div><div class='stat-label'>Best Accuracy</div></div>""", unsafe_allow_html=True)
with c4: st.markdown("""<div class='stat-card'><div class='stat-number'>3</div><div class='stat-label'>Sentiment Classes</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍  Sentiment Analysis", "📊  Model Performance", "ℹ️  How It Works"])

# TAB 1
with tab1:
    st.markdown("<div class='section-header'>Enter Text for Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:rgba(255,255,255,0.5);font-size:0.83rem;margin-bottom:0.5rem;'> — <b style='color:#e94560'></b> </div>", unsafe_allow_html=True)
   user_input = st.text_area("",height=160, label_visibility="collapsed")
    col1, col2, _ = st.columns([1.3, 0.8, 4])
    with col1: analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)
    with col2: clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if analyze_btn and user_input.strip():
        # Check karo kitni lines hain
        lines = [line.strip() for line in user_input.strip().split('\n') if line.strip()]
        is_multi = len(lines) > 1

        if is_multi:
            # Multiple tweets mode
            st.markdown("<div class='section-header'>📋 Multiple Tweet Results</div>", unsafe_allow_html=True)
            
            all_results = []
            with st.spinner(f"🔄 Analyzing {len(lines)} tweets..."):
                for tweet in lines:
                    try:
                        try:
                            translated = GoogleTranslator(source='auto', target='en').translate(tweet)
                        except:
                            translated = tweet

                        cleaned = start(translated)
                        feat = [i for i in cleaned.split() if i in adjectives]
                        feat_filtered = [w for w in feat if w in word_features]
                        test_data = list_to_dict(feat_filtered)

                        votes = []
                        for clf, name in classifiers:
                            votes.append(clf.classify(test_data))

                        from statistics import mode
                        try:
                            hybrid_res = mode(votes)
                        except:
                            hybrid_res = "positive"

                        sia = SentimentIntensityAnalyzer()
                        vscore = sia.polarity_scores(translated)
                        compound = vscore['compound']

                        if compound >= 0.05:
                            vader_res = "positive"
                        elif compound <= -0.05:
                            vader_res = "negative"
                        else:
                            vader_res = "neutral"

                        # VADER neutral → NEUTRAL
                        # VADER positive/negative → Hybrid result
                        if vader_res == "neutral":
                            final = "NEUTRAL"
                        else:
                            final = hybrid_res.upper()

                        conf = get_confidence(votes, hybrid_res)
                        all_results.append({"Tweet": tweet, "Sentiment": final, "Confidence": f"{conf}%", "Compound": round(compound, 3)})

                    except:
                        all_results.append({"Tweet": tweet, "Sentiment": "ERROR", "Confidence": "0%", "Compound": 0})

            # Summary cards
            pos_c = sum(1 for r in all_results if r['Sentiment'] == 'POSITIVE')
            neg_c = sum(1 for r in all_results if r['Sentiment'] == 'NEGATIVE')
            neu_c = sum(1 for r in all_results if r['Sentiment'] == 'NEUTRAL')

            s1, s2, s3, s4 = st.columns(4)
            with s1: st.markdown(f"<div class='stat-card'><div class='stat-number'>{len(lines)}</div><div class='stat-label'>Total Tweets</div></div>", unsafe_allow_html=True)
            with s2: st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#22c55e;'>{pos_c}</div><div class='stat-label'>😊 Positive</div></div>", unsafe_allow_html=True)
            with s3: st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#ef4444;'>{neg_c}</div><div class='stat-label'>😔 Negative</div></div>", unsafe_allow_html=True)
            with s4: st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#7b68ee;'>{neu_c}</div><div class='stat-label'>😐 Neutral</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Individual results
            for r in all_results:
                if r['Sentiment'] == 'POSITIVE':
                    color = "#22c55e"
                    icon = "😊"
                    bg = "#0a2e1a"
                    border = "#22c55e"
                elif r['Sentiment'] == 'NEGATIVE':
                    color = "#ef4444"
                    icon = "😔"
                    bg = "#2e0a0a"
                    border = "#ef4444"
                else:
                    color = "#7b68ee"
                    icon = "😐"
                    bg = "#1a1a2e"
                    border = "#7b68ee"

                st.markdown(
                    f"<div style='background:{bg};border:1px solid {border};border-radius:10px;padding:0.8rem 1rem;margin:0.4rem 0;'>"
                    f"<table style='width:100%;'><tr>"
                    f"<td style='color:white;font-size:0.87rem;width:55%;'>{r['Tweet']}</td>"
                    f"<td style='text-align:center;color:{color};font-weight:700;font-size:0.9rem;'>{icon} {r['Sentiment']}</td>"
                    f"<td style='text-align:right;color:#aaaaaa;font-size:0.82rem;'>{r['Confidence']}</td>"
                    f"</tr></table></div>",
                    unsafe_allow_html=True
                )

            # Download results
            st.markdown("<br>", unsafe_allow_html=True)
            df_multi = pd.DataFrame(all_results)
            csv_data = df_multi.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_data,
                file_name="multi_tweet_results.csv",
                mime="text/csv"
            )

        else:
            # Single tweet mode
            with st.spinner("🔄 Analyzing sentiment..."):
                try:
                    translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                    if translated.lower() != user_input.lower():
                        st.markdown(f"<div class='info-box'>🌐 <b>Auto-Translated:</b> {translated}</div>", unsafe_allow_html=True)
                    processed = translated
                except:
                    processed = user_input

                cleaned = start(processed)
                feat = [i for i in cleaned.split() if i in adjectives]
                feat_filtered = [w for w in feat if w in word_features]
                test_data = list_to_dict(feat_filtered)

                results = []
                votes = []
                for clf, name in classifiers:
                    label = clf.classify(test_data)
                    results.append({"Classifier": name, "Prediction": label})
                    votes.append(label)

                from statistics import mode
                try: hybrid_result = mode(votes)
                except: hybrid_result = classifiers[3][0].classify(test_data)

                vader_result, vader_score = get_vader_sentiment(processed)
                final_result = get_final_sentiment(hybrid_result, vader_result)
                confidence = get_confidence(votes, hybrid_result)
                conf_level, conf_color, conf_emoji = get_confidence_level(confidence)

            st.markdown("<br>", unsafe_allow_html=True)
            left, right = st.columns([1.1, 1])

            with left:
                if final_result == "positive":
                    st.markdown("""<div class='result-positive'><div style='font-size:4rem;'>😊</div><h1>POSITIVE</h1><p>Combined Hybrid + VADER Result</p></div>""", unsafe_allow_html=True)
                elif final_result == "negative":
                    st.markdown("""<div class='result-negative'><div style='font-size:4rem;'>😔</div><h1>NEGATIVE</h1><p>Combined Hybrid + VADER Result</p></div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class='result-neutral'><div style='font-size:4rem;'>😐</div><h1>NEUTRAL</h1><p>Combined Hybrid + VADER Result</p></div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class='confidence-box'>
                    <div style='color:#aaaaaa;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;'>🎯 Confidence Score</div>
                    <div style='font-family:Poppins,sans-serif;font-size:2.5rem;font-weight:800;color:{conf_color};'>{confidence}%</div>
                    <div style='color:{conf_color};font-size:0.85rem;font-weight:600;margin-top:0.3rem;'>{conf_emoji} {conf_level} Confidence</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:0.8rem;color:#aaaaaa;font-size:0.8rem;'>Confidence Level:</div>", unsafe_allow_html=True)
                st.progress(confidence / 100)

                pos_v = votes.count("positive")
                neg_v = votes.count("negative")
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:0.8rem 1rem;margin-top:0.8rem;text-align:center;border:1px solid rgba(255,255,255,0.06);'>
                    <span style='color:#22c55e;font-weight:600;'>✅ Positive Votes: {pos_v}</span>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <span style='color:#ef4444;font-weight:600;'>❌ Negative Votes: {neg_v}</span>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class='vader-box'>
                    <div style='color:#7b68ee;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:0.8rem;'>🔬 VADER Analysis</div>
                    <table style='width:100%;'>
                        <tr><td style='color:rgba(255,255,255,0.6);font-size:0.82rem;'>😊 Positive</td><td style='text-align:right;color:#22c55e;font-weight:600;'>{round(vader_score['pos']*100,1)}%</td></tr>
                        <tr><td style='color:rgba(255,255,255,0.6);font-size:0.82rem;'>😔 Negative</td><td style='text-align:right;color:#ef4444;font-weight:600;'>{round(vader_score['neg']*100,1)}%</td></tr>
                        <tr><td style='color:rgba(255,255,255,0.6);font-size:0.82rem;'>😐 Neutral</td><td style='text-align:right;color:#7b68ee;font-weight:600;'>{round(vader_score['neu']*100,1)}%</td></tr>
                        <tr><td style='color:rgba(255,255,255,0.6);font-size:0.82rem;'>📊 Compound</td><td style='text-align:right;color:white;font-weight:600;'>{round(vader_score['compound'],3)}</td></tr>
                        <tr><td style='color:rgba(255,255,255,0.6);font-size:0.82rem;'>🏷️ VADER Label</td><td style='text-align:right;color:#7b68ee;font-weight:600;'>{vader_result.upper()}</td></tr>
                    </table>
                </div>""", unsafe_allow_html=True)

            with right:
                st.markdown("<div class='section-header'>All Classifier Results</div>", unsafe_allow_html=True)
                for r in results:
                    is_pos = r["Prediction"] == "positive"
                    icon = "✅" if is_pos else "❌"
                    color = "#22c55e" if is_pos else "#ef4444"
                    label = "POSITIVE" if is_pos else "NEGATIVE"
                    clf_name = str(r["Classifier"])
                    st.markdown(
                        f"<div style='background:#1a1a2e;border:1px solid #2a2a4e;border-radius:10px;padding:0.65rem 1rem;margin:0.3rem 0;'>"
                        f"<table style='width:100%;'><tr>"
                        f"<td style='color:white;font-size:0.87rem;font-weight:500;width:70%;'>{clf_name}</td>"
                        f"<td style='text-align:right;color:{color};font-weight:700;font-size:0.84rem;'>{icon} {label}</td>"
                        f"</tr></table></div>",
                        unsafe_allow_html=True
                    )

                hybrid_color = "#22c55e" if hybrid_result == "positive" else "#ef4444"
                hybrid_icon = "✅" if hybrid_result == "positive" else "❌"
                st.markdown(
                    f"<div style='background:#2a0a1a;border:1px solid #e94560;border-radius:10px;padding:0.65rem 1rem;margin:0.3rem 0;'>"
                    f"<table style='width:100%;'><tr>"
                    f"<td style='color:white;font-size:0.87rem;font-weight:600;width:70%;'>🏆 Hybrid Ensemble Model</td>"
                    f"<td style='text-align:right;color:{hybrid_color};font-weight:700;font-size:0.84rem;'>{hybrid_icon} {hybrid_result.upper()}</td>"
                    f"</tr></table></div>",
                    unsafe_allow_html=True
                )

                vader_color = "#22c55e" if vader_result == "positive" else ("#ef4444" if vader_result == "negative" else "#7b68ee")
                vader_icon = "✅" if vader_result == "positive" else ("❌" if vader_result == "negative" else "😐")
                st.markdown(
                    f"<div style='background:#0f1a3e;border:1px solid #7b68ee;border-radius:10px;padding:0.65rem 1rem;margin:0.3rem 0;'>"
                    f"<table style='width:100%;'><tr>"
                    f"<td style='color:white;font-size:0.87rem;font-weight:600;width:70%;'>🔬 VADER Analyzer</td>"
                    f"<td style='text-align:right;color:{vader_color};font-weight:700;font-size:0.84rem;'>{vader_icon} {vader_result.upper()}</td>"
                    f"</tr></table></div>",
                    unsafe_allow_html=True
                )


            # Clause Analysis
            clauses = analyze_clauses(user_input)
            if len(clauses) > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class=\'section-header\'>🔀 Clause-wise Sentiment Breakdown</div>", unsafe_allow_html=True)
                st.markdown("<div style=\'color:rgba(255,255,255,0.5);font-size:0.83rem;margin-bottom:0.8rem;\'>Your sentence contains multiple parts — each part analyzed separately!</div>", unsafe_allow_html=True)

                clause_results = get_clause_sentiments(clauses)
                pos_count = sum(1 for r in clause_results if r["sentiment"] == "POSITIVE")
                neg_count = sum(1 for r in clause_results if r["sentiment"] == "NEGATIVE")
                neu_count = sum(1 for r in clause_results if r["sentiment"] == "NEUTRAL")

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    st.markdown(f"<div class=\'stat-card\'><div class=\'stat-number\' style=\'color:#22c55e;font-size:1.5rem;\'>{pos_count}</div><div class=\'stat-label\'>Positive Parts</div></div>", unsafe_allow_html=True)
                with cc2:
                    st.markdown(f"<div class=\'stat-card\'><div class=\'stat-number\' style=\'color:#ef4444;font-size:1.5rem;\'>{neg_count}</div><div class=\'stat-label\'>Negative Parts</div></div>", unsafe_allow_html=True)
                with cc3:
                    st.markdown(f"<div class=\'stat-card\'><div class=\'stat-number\' style=\'color:#7b68ee;font-size:1.5rem;\'>{neu_count}</div><div class=\'stat-label\'>Neutral Parts</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                for idx, r in enumerate(clause_results):
                    clause_html = (
                        "<div style='background:#1a1a2e;border:1px solid " + r["color"] + ";border-radius:10px;padding:0.8rem 1.2rem;margin:0.4rem 0;'>"
                        "<table style='width:100%;'><tr>"
                        "<td style='color:#aaaaaa;font-size:0.75rem;width:8%;'>Part " + str(idx+1) + "</td>"
                        "<td style='color:white;font-size:0.87rem;width:55%;font-style:italic;'>" + r["clause"] + "</td>"
                        "<td style='text-align:center;color:" + r["color"] + ";font-weight:700;font-size:0.9rem;'>" + r["icon"] + " " + r["sentiment"] + "</td>"
                        "<td style='text-align:right;color:#aaaaaa;font-size:0.8rem;'>Score: " + str(r["compound"]) + "</td>"
                        "</tr></table></div>"
                    )
                    st.markdown(clause_html, unsafe_allow_html=True)

                if pos_count > 0 and neg_count > 0:
                    st.warning("Mixed Sentiment Detected! This sentence contains both Positive and Negative parts. The overall result shows the dominant sentiment.")

    elif analyze_btn:
        st.warning("⚠️ Please enter some text to analyze!")

    # Live Demo Section
    st.markdown("---")
    st.markdown("<div class='section-header'>🔴 Live Tweet Analysis Demo</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-bottom:1rem;'>Watch tweets being analyzed in real-time — one by one, just like a live Twitter feed!</div>", unsafe_allow_html=True)

    sample_tweets = [
        "I am very happy and excited today!",
        "This is so sad and depressing, everything is wrong",
        "Life is beautiful yaar, bahut maza aa raha hai",
        "I hate everything, this is terrible and horrible",
        "aaj ka din bahut amazing tha, loved it!",
        "I went to the market today",
        "This movie was absolutely wonderful and amazing",
        "I am so frustrated and angry right now",
        "yaar ye bahut achha hai, I am loving it",
        "Everything is going wrong today, feeling terrible"
    ]

    speed = st.select_slider(
        "Analysis Speed:",
        options=["Slow (2s)", "Normal (1s)", "Fast (0.5s)"],
        value="Normal (1s)"
    )
    speed_map = {"Slow (2s)": 2.0, "Normal (1s)": 1.0, "Fast (0.5s)": 0.5}
    delay = speed_map[speed]

    if st.button("▶️ Start Live Demo", use_container_width=False):
        st.markdown("<div style='color:#e94560;font-weight:600;font-size:0.9rem;margin:0.5rem 0;'>🔴 LIVE — Analyzing tweets...</div>", unsafe_allow_html=True)

        live_container = st.empty()
        summary_container = st.empty()

        import time
        live_results = []

        for i, tweet in enumerate(sample_tweets):
            try:
                # Translation
                try:
                    translated = GoogleTranslator(source='auto', target='en').translate(tweet)
                except:
                    translated = tweet

                # Preprocessing
                cleaned = start(translated)
                feat = [w for w in cleaned.split() if w in adjectives]
                feat_filtered = [w for w in feat if w in word_features]
                test_data = list_to_dict(feat_filtered)

                # Hybrid votes
                votes = []
                for clf, name in classifiers:
                    votes.append(clf.classify(test_data))

                from statistics import mode
                try:
                    hybrid_res = mode(votes)
                except:
                    hybrid_res = "positive"

                # VADER
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                vscore = sia.polarity_scores(translated)
                compound = vscore['compound']

                if compound >= 0.05:
                    vader_res = "positive"
                elif compound <= -0.05:
                    vader_res = "negative"
                else:
                    vader_res = "neutral"

                if vader_res == "neutral":
                    final = "NEUTRAL"
                else:
                    final = hybrid_res.upper()

                conf = get_confidence(votes, hybrid_res)
                live_results.append({"tweet": tweet, "sentiment": final, "confidence": conf})

            except:
                final = "ERROR"
                conf = 0
                live_results.append({"tweet": tweet, "sentiment": "ERROR", "confidence": 0})

            # Live display update
            with live_container.container():
                st.markdown(f"<div style='color:#aaaaaa;font-size:0.78rem;margin-bottom:0.5rem;'>Analyzed: {i+1}/{len(sample_tweets)} tweets</div>", unsafe_allow_html=True)
                for j, r in enumerate(live_results):
                    if r['sentiment'] == 'POSITIVE':
                        color = "#22c55e"; icon = "😊"; bg = "#0a2e1a"; border = "#22c55e"
                    elif r['sentiment'] == 'NEGATIVE':
                        color = "#ef4444"; icon = "😔"; bg = "#2e0a0a"; border = "#ef4444"
                    elif r['sentiment'] == 'NEUTRAL':
                        color = "#7b68ee"; icon = "😐"; bg = "#1a1a2e"; border = "#7b68ee"
                    else:
                        color = "#aaaaaa"; icon = "❓"; bg = "#1a1a2e"; border = "#aaaaaa"

                    # Highlight latest tweet
                    opacity = "1" if j == len(live_results)-1 else "0.6"
                    new_badge = "<span style='background:#e94560;color:white;font-size:0.65rem;padding:0.1rem 0.4rem;border-radius:4px;margin-left:0.5rem;'>NEW</span>" if j == len(live_results)-1 else ""

                    st.markdown(
                        f"<div style='background:{bg};border:1px solid {border};border-radius:10px;padding:0.6rem 1rem;margin:0.25rem 0;opacity:{opacity};'>"
                        f"<table style='width:100%;'><tr>"
                        f"<td style='color:white;font-size:0.84rem;width:60%;'>{r['tweet'][:50]}...{new_badge}</td>"
                        f"<td style='text-align:center;color:{color};font-weight:700;font-size:0.88rem;'>{icon} {r['sentiment']}</td>"
                        f"<td style='text-align:right;color:#aaaaaa;font-size:0.8rem;'>{r['confidence']}%</td>"
                        f"</tr></table></div>",
                        unsafe_allow_html=True
                    )

            time.sleep(delay)

        # Final summary
        pos_c = sum(1 for r in live_results if r['sentiment'] == 'POSITIVE')
        neg_c = sum(1 for r in live_results if r['sentiment'] == 'NEGATIVE')
        neu_c = sum(1 for r in live_results if r['sentiment'] == 'NEUTRAL')

        with summary_container.container():
            st.markdown("<div class='section-header'>📊 Live Demo Summary</div>", unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            with s1: st.markdown(f"<div class='stat-card'><div class='stat-number'>{len(sample_tweets)}</div><div class='stat-label'>Total Analyzed</div></div>", unsafe_allow_html=True)
            with s2: st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#22c55e;'>{pos_c}</div><div class='stat-label'>😊 Positive</div></div>", unsafe_allow_html=True)
            with s3: st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#ef4444;'>{neg_c}</div><div class='stat-label'>😔 Negative</div></div>", unsafe_allow_html=True)
            with s4: st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#7b68ee;'>{neu_c}</div><div class='stat-label'>😐 Neutral</div></div>", unsafe_allow_html=True)

        st.success("✅ Live Demo Complete!")

    # CSV Upload Section
    st.markdown("---")
    st.markdown("<div class='section-header'>📂 Bulk CSV Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-bottom:1rem;'>Upload a CSV file with a column named <b style='color:#e94560'>tweet</b> to analyze multiple tweets at once!</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], help="CSV mein ek column hona chahiye jiska naam 'tweet' ho")

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)

        if 'tweet' not in df_upload.columns:
            st.error("❌ CSV mein 'tweet' column nahi mila! Column ka naam 'tweet' hona chahiye.")
        else:
            st.success(f"✅ {len(df_upload)} tweets loaded successfully!")
            st.dataframe(df_upload.head(5), use_container_width=True, hide_index=True)

            if st.button("🚀 Analyze All Tweets"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                results_list = []
                confidence_list = []
                compound_list = []
                vader_list = []
                total = len(df_upload)

                for i, tweet in enumerate(df_upload['tweet']):
                    try:
                        try:
                            translated = GoogleTranslator(source='auto', target='en').translate(str(tweet))
                        except:
                            translated = str(tweet)

                        cleaned = start(translated)
                        feat = [w for w in cleaned.split() if w in adjectives]
                        feat_filtered = [w for w in feat if w in word_features]
                        test_data = list_to_dict(feat_filtered)

                        votes = []
                        for clf, name in classifiers:
                            votes.append(clf.classify(test_data))

                        from statistics import mode
                        try:
                            hybrid_res = mode(votes)
                        except:
                            hybrid_res = "positive"

                        sia = SentimentIntensityAnalyzer()
                        vscore = sia.polarity_scores(translated)
                        compound = vscore['compound']

                        if compound >= 0.05:
                            vader_res = "positive"
                        elif compound <= -0.05:
                            vader_res = "negative"
                        else:
                            vader_res = "neutral"

                        # VADER neutral → NEUTRAL
                        # VADER positive/negative → Hybrid result
                        if vader_res == "neutral":
                            final = "NEUTRAL"
                        else:
                            final = hybrid_res.upper()

                        conf = get_confidence(votes, hybrid_res)
                        results_list.append(final)
                        confidence_list.append(f"{conf}%")
                        compound_list.append(round(compound, 3))
                        vader_list.append(vader_res.upper())

                    except:
                        results_list.append("ERROR")
                        confidence_list.append("0%")
                        compound_list.append(0)
                        vader_list.append("ERROR")

                    progress_bar.progress((i + 1) / total)
                    status_text.text(f"Analyzing: {i+1}/{total} tweets...")

                status_text.text("✅ Analysis Complete!")
                df_upload['Sentiment'] = results_list
                df_upload['Confidence'] = confidence_list
                df_upload['Compound_Score'] = compound_list
                df_upload['VADER_Label'] = vader_list

                st.markdown("<div class='section-header'>📊 Analysis Summary</div>", unsafe_allow_html=True)
                pos_count = results_list.count('POSITIVE')
                neg_count = results_list.count('NEGATIVE')
                neu_count = results_list.count('NEUTRAL')

                s1, s2, s3, s4 = st.columns(4)
                with s1: st.markdown(f"<div class='stat-card'><div class='stat-number'>{total}</div><div class='stat-label'>Total Tweets</div></div>", unsafe_allow_html=True)
                with s2: st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#22c55e;'>{pos_count}</div><div class='stat-label'>Positive</div></div>", unsafe_allow_html=True)
                with s3: st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#ef4444;'>{neg_count}</div><div class='stat-label'>Negative</div></div>", unsafe_allow_html=True)
                with s4: st.markdown(f"<div class='stat-card'><div class='stat-number' style='color:#7b68ee;'>{neu_count}</div><div class='stat-label'>Neutral</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                fig_pie.patch.set_facecolor('#0d0d1a')
                ax_pie.set_facecolor('#0d0d1a')
                pie_labels, pie_values, pie_colors = [], [], []
                if pos_count > 0:
                    pie_labels.append(f'Positive ({pos_count})')
                    pie_values.append(pos_count)
                    pie_colors.append('#22c55e')
                if neg_count > 0:
                    pie_labels.append(f'Negative ({neg_count})')
                    pie_values.append(neg_count)
                    pie_colors.append('#ef4444')
                if neu_count > 0:
                    pie_labels.append(f'Neutral ({neu_count})')
                    pie_values.append(neu_count)
                    pie_colors.append('#7b68ee')
                ax_pie.pie(pie_values, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', textprops={'color':'white'}, startangle=90)
                ax_pie.set_title("Sentiment Distribution", color='white', fontsize=12, fontweight='bold')
                st.pyplot(fig_pie)

                st.markdown("<div class='section-header'>📋 Detailed Results</div>", unsafe_allow_html=True)
                st.dataframe(df_upload, use_container_width=True, hide_index=True)

                csv_result = df_upload.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_result,
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# TAB 2
with tab2:
    perf_tab1, perf_tab2, perf_tab3, perf_tab4 = st.tabs([
        "📊 Accuracy Chart",
        "📋 Detailed Metrics",
        "🔲 Confusion Matrix",
        "📈 Metrics Comparison"
    ])

    names = [a[1] for a in accuracy]
    values = [round(a[0], 2) for a in accuracy]
    max_val = max(values)

    with perf_tab1:
        st.markdown("<div class=\'section-header\'>Classifier Accuracy Comparison</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#1a1a2e")
        colors = ["#e94560" if v == max_val else "#0f3460" for v in values]
        bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="none", zorder=3)
        ax.set_ylim([55, 70])
        ax.set_xlabel("Classifier", fontsize=11, color="#aaaaaa", labelpad=10)
        ax.set_ylabel("Accuracy (%)", fontsize=11, color="#aaaaaa", labelpad=10)
        ax.set_title("Model Accuracy Comparison", fontsize=13, color="white", pad=15, fontweight="bold")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        ax.grid(axis="y", color="#2a2a3e", linestyle="--", zorder=0)
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for spine in ["bottom","left"]: ax.spines[spine].set_color("#2a2a3e")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold", color="white")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class=\'section-header\'>Accuracy Table</div>", unsafe_allow_html=True)
            df_acc = pd.DataFrame({"Classifier": names, "Accuracy (%)": values, "Rank": ["Best" if v == max_val else "Good" for v in values]})
            st.dataframe(df_acc, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("<div class=\'section-header\'>Key Observations</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:1.5rem;border:1px solid rgba(255,255,255,0.07);'>
                <div style='color:rgba(255,255,255,0.75);font-size:0.85rem;line-height:2;'>
                    All classifiers achieve <b style='color:#e94560'>~62-65%</b> accuracy<br>
                    Hybrid model uses <b style='color:#e94560'>majority voting</b> strategy<br>
                    <b style='color:#7b68ee'>VADER</b> added for Neutral detection<br>
                    Dataset: <b style='color:#e94560'>40,000 tweets</b> from Sentiment140<br>
                    Train/Test split: <b style='color:#e94560'>85% / 15%</b><br>
                    Supports <b style='color:#e94560'>3 classes</b>: Positive, Negative, Neutral
                </div>
            </div>""", unsafe_allow_html=True)

    with perf_tab2:
        st.markdown("<div class=\'section-header\'>Precision · Recall · F1 Score</div>", unsafe_allow_html=True)
        st.markdown("<div style=\'color:rgba(255,255,255,0.5);font-size:0.83rem;margin-bottom:1rem;\'>Research-level evaluation metrics for all classifiers</div>", unsafe_allow_html=True)

        metrics_df = pd.DataFrame([{
            "Classifier": m["name"],
            "Accuracy (%)": m["accuracy"],
            "Precision (%)": m["precision"],
            "Recall (%)": m["recall"],
            "F1 Score (%)": m["f1"]
        } for m in metrics_data])

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class=\'section-header\'>Best Performer Per Metric</div>", unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        best_acc_v = metrics_df["Accuracy (%)"].max()
        best_prec_v = metrics_df["Precision (%)"].max()
        best_rec_v = metrics_df["Recall (%)"].max()
        best_f1_v = metrics_df["F1 Score (%)"].max()
        best_acc_clf = metrics_df.loc[metrics_df["Accuracy (%)"].idxmax(), "Classifier"]
        best_prec_clf = metrics_df.loc[metrics_df["Precision (%)"].idxmax(), "Classifier"]
        best_rec_clf = metrics_df.loc[metrics_df["Recall (%)"].idxmax(), "Classifier"]
        best_f1_clf = metrics_df.loc[metrics_df["F1 Score (%)"].idxmax(), "Classifier"]

        with b1:
            st.markdown(f"<div class=\'stat-card\'><div class=\'stat-number\' style=\'font-size:1.3rem;color:#22c55e;\'>{best_acc_v}%</div><div style=\'color:#22c55e;font-size:0.7rem;font-weight:600;\'>Best Accuracy</div><div style=\'color:#aaaaaa;font-size:0.72rem;margin-top:0.3rem;\'>{best_acc_clf}</div></div>", unsafe_allow_html=True)
        with b2:
            st.markdown(f"<div class=\'stat-card\'><div class=\'stat-number\' style=\'font-size:1.3rem;color:#e94560;\'>{best_prec_v}%</div><div style=\'color:#e94560;font-size:0.7rem;font-weight:600;\'>Best Precision</div><div style=\'color:#aaaaaa;font-size:0.72rem;margin-top:0.3rem;\'>{best_prec_clf}</div></div>", unsafe_allow_html=True)
        with b3:
            st.markdown(f"<div class=\'stat-card\'><div class=\'stat-number\' style=\'font-size:1.3rem;color:#7b68ee;\'>{best_rec_v}%</div><div style=\'color:#7b68ee;font-size:0.7rem;font-weight:600;\'>Best Recall</div><div style=\'color:#aaaaaa;font-size:0.72rem;margin-top:0.3rem;\'>{best_rec_clf}</div></div>", unsafe_allow_html=True)
        with b4:
            st.markdown(f"<div class=\'stat-card\'><div class=\'stat-number\' style=\'font-size:1.3rem;color:#eab308;\'>{best_f1_v}%</div><div style=\'color:#eab308;font-size:0.7rem;font-weight:600;\'>Best F1 Score</div><div style=\'color:#aaaaaa;font-size:0.72rem;margin-top:0.3rem;\'>{best_f1_clf}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class=\'section-header\'>Metric Definitions</div>", unsafe_allow_html=True)
        defs = [
            ("Accuracy", "Overall % of correct predictions out of total predictions"),
            ("Precision", "Out of all Positive predictions, how many were actually Positive?"),
            ("Recall", "Out of all actual Positives, how many did the model correctly find?"),
            ("F1 Score", "Harmonic mean of Precision and Recall — best overall metric"),
        ]
        d1, d2 = st.columns(2)
        for i, (title, desc) in enumerate(defs):
            with d1 if i % 2 == 0 else d2:
                st.markdown(f"<div style=\'background:#1a1a2e;border-radius:10px;padding:0.8rem 1rem;margin:0.3rem 0;border-left:3px solid #e94560;\'><div style=\'color:white;font-weight:600;font-size:0.9rem;\'>{title}</div><div style=\'color:rgba(255,255,255,0.55);font-size:0.82rem;margin-top:0.2rem;\'>{desc}</div></div>", unsafe_allow_html=True)

    with perf_tab3:
        st.markdown("<div class=\'section-header\'>Confusion Matrix</div>", unsafe_allow_html=True)
        st.markdown("<div style=\'color:rgba(255,255,255,0.5);font-size:0.83rem;margin-bottom:1rem;\'>Shows actual vs predicted results for each classifier</div>", unsafe_allow_html=True)

        clf_names_list = [m["name"] for m in metrics_data]
        selected_clf = st.selectbox("Select Classifier:", clf_names_list)
        selected_metrics = next(m for m in metrics_data if m["name"] == selected_clf)
        cm = selected_metrics["confusion_matrix"]

        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        fig_cm.patch.set_facecolor("#0d0d1a")
        ax_cm.set_facecolor("#1a1a2e")
        ax_cm.imshow(cm, interpolation="nearest", cmap="RdYlGn")
        ax_cm.set_title(f"Confusion Matrix — {selected_clf}", color="white", fontsize=13, fontweight="bold", pad=15)
        classes = ["Positive", "Negative"]
        ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(classes, color="white", fontsize=11)
        ax_cm.set_yticklabels(classes, color="white", fontsize=11)
        ax_cm.set_xlabel("Predicted Label", color="#aaaaaa", fontsize=11)
        ax_cm.set_ylabel("Actual Label", color="#aaaaaa", fontsize=11)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                          color="white" if cm[i, j] < thresh else "black", fontsize=16, fontweight="bold")
        plt.tight_layout()

        col_cm1, col_cm2 = st.columns([1.2, 1])
        with col_cm1:
            st.pyplot(fig_cm)
        with col_cm2:
            TP = int(cm[0][0]); FN = int(cm[0][1]); FP = int(cm[1][0]); TN = int(cm[1][1])
            st.markdown(f"""
            <div style='background:#1a1a2e;border-radius:12px;padding:1.2rem;margin-top:2rem;border:1px solid rgba(255,255,255,0.07);'>
                <div style='color:#e94560;font-weight:600;font-size:0.85rem;margin-bottom:1rem;'>Matrix Breakdown</div>
                <table style='width:100%;'>
                    <tr><td style='color:#22c55e;font-size:0.85rem;padding:0.3rem 0;'>True Positive (TP)</td><td style='text-align:right;color:white;font-weight:700;'>{TP}</td></tr>
                    <tr><td style='color:#ef4444;font-size:0.85rem;padding:0.3rem 0;'>False Negative (FN)</td><td style='text-align:right;color:white;font-weight:700;'>{FN}</td></tr>
                    <tr><td style='color:#f97316;font-size:0.85rem;padding:0.3rem 0;'>False Positive (FP)</td><td style='text-align:right;color:white;font-weight:700;'>{FP}</td></tr>
                    <tr><td style='color:#7b68ee;font-size:0.85rem;padding:0.3rem 0;'>True Negative (TN)</td><td style='text-align:right;color:white;font-weight:700;'>{TN}</td></tr>
                </table>
                <hr style='border-color:rgba(255,255,255,0.08);margin:0.8rem 0;'>
                <div style='color:rgba(255,255,255,0.75);font-size:0.83rem;line-height:1.8;'>
                    Accuracy: <b style='color:#22c55e;'>{selected_metrics["accuracy"]}%</b><br>
                    Precision: <b style='color:#e94560;'>{selected_metrics["precision"]}%</b><br>
                    Recall: <b style='color:#7b68ee;'>{selected_metrics["recall"]}%</b><br>
                    F1 Score: <b style='color:#eab308;'>{selected_metrics["f1"]}%</b>
                </div>
            </div>""", unsafe_allow_html=True)

    with perf_tab4:
        st.markdown("<div class=\'section-header\'>All Metrics Comparison Chart</div>", unsafe_allow_html=True)
        clf_names_chart = [m["name"] for m in metrics_data]
        acc_vals = [m["accuracy"] for m in metrics_data]
        prec_vals = [m["precision"] for m in metrics_data]
        rec_vals = [m["recall"] for m in metrics_data]
        f1_vals = [m["f1"] for m in metrics_data]
        x = np.arange(len(clf_names_chart))
        width = 0.2

        fig_comp, ax_comp = plt.subplots(figsize=(14, 6))
        fig_comp.patch.set_facecolor("#0d0d1a")
        ax_comp.set_facecolor("#1a1a2e")
        ax_comp.bar(x - 1.5*width, acc_vals, width, label="Accuracy", color="#0f3460", zorder=3)
        ax_comp.bar(x - 0.5*width, prec_vals, width, label="Precision", color="#e94560", zorder=3)
        ax_comp.bar(x + 0.5*width, rec_vals, width, label="Recall", color="#7b68ee", zorder=3)
        ax_comp.bar(x + 1.5*width, f1_vals, width, label="F1 Score", color="#eab308", zorder=3)
        ax_comp.set_ylim([50, 75])
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(clf_names_chart, rotation=15, ha="right", color="#aaaaaa", fontsize=9)
        ax_comp.set_ylabel("Score (%)", color="#aaaaaa", fontsize=11)
        ax_comp.set_title("Accuracy vs Precision vs Recall vs F1", color="white", fontsize=13, fontweight="bold", pad=15)
        ax_comp.tick_params(colors="#aaaaaa")
        ax_comp.grid(axis="y", color="#2a2a3e", linestyle="--", zorder=0)
        ax_comp.legend(facecolor="#1a1a2e", edgecolor="#2a2a4e", labelcolor="white", fontsize=9)
        for spine in ["top","right"]: ax_comp.spines[spine].set_visible(False)
        for spine in ["bottom","left"]: ax_comp.spines[spine].set_color("#2a2a3e")
        plt.tight_layout()
        st.pyplot(fig_comp)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:1.5rem;border:1px solid rgba(255,255,255,0.07);'>
            <div style='color:rgba(255,255,255,0.75);font-size:0.85rem;line-height:2;'>
                <b style='color:#0f3460'>Accuracy</b> — Overall correct predictions<br>
                <b style='color:#e94560'>Precision</b> — Minimize false positives<br>
                <b style='color:#7b68ee'>Recall</b> — Minimize false negatives<br>
                <b style='color:#eab308'>F1 Score</b> — Best overall metric for sentiment analysis!
            </div>
        </div>""", unsafe_allow_html=True)

# TAB 3
with tab3:
    st.markdown("<div class='section-header'>Processing Pipeline</div>", unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Input Text", "User enters English, Hindi, or Hinglish text into the system"),
        ("2️⃣", "Language Detection & Translation", "Hindi/Hinglish words auto-detected and translated to English using Google Translate API"),
        ("3️⃣", "Text Preprocessing", "HTML tags removed → URLs removed → Emojis removed → Contractions expanded → Stopwords removed → Lemmatization applied"),
        ("4️⃣", "Feature Extraction", "Adjectives and negation words extracted as sentiment features using Bag-of-Words approach"),
        ("5️⃣", "ML Classification", "6 classifiers independently predict: Naive Bayes, Multinomial NB, Bernoulli NB, Logistic Regression, SGD, SVM"),
        ("6️⃣", "Hybrid Ensemble Voting", "All 6 predictions combined using majority voting — tie handled by Logistic Regression"),
        ("7️⃣", "VADER Analysis", "VADER scores Positive, Negative, Neutral using compound score (-1 to +1)"),
        ("8️⃣", "Final Combined Result", "If VADER = Neutral → Final is Neutral. Else Hybrid + VADER combine for final decision"),
        ("9️⃣", "Confidence Score", "Calculated from voting ratio — shows how sure the model is about prediction"),
        ("🔟", "Result Display", "Final sentiment (POSITIVE/NEGATIVE/NEUTRAL) with confidence score and all classifier results"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;
             padding:1rem 1.2rem;margin:0.5rem 0;border:1px solid rgba(255,255,255,0.06);border-left:3px solid #e94560;'>
            <div style='display:flex;align-items:flex-start;gap:0.8rem;'>
                <span style='font-size:1.3rem;'>{icon}</span>
                <div>
                    <div style='color:white;font-weight:600;font-size:0.95rem;'>{title}</div>
                    <div style='color:rgba(255,255,255,0.55);font-size:0.84rem;margin-top:0.25rem;'>{desc}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Technologies & Libraries</div>", unsafe_allow_html=True)
    techs = [
        ("🐍", "Python", "Core programming language for entire project"),
        ("📚", "NLTK", "Tokenization, POS tagging, lemmatization, Naive Bayes, VADER"),
        ("🤖", "Scikit-learn", "ML classifiers — Multinomial NB, Bernoulli NB, Logistic Regression, SGD, SVM"),
        ("🔬", "VADER", "Sentiment scoring for Neutral detection — compound score based"),
        ("🌐", "Deep Translator", "Google Translate API for Hindi to English translation"),
        ("🍜", "BeautifulSoup", "HTML tag removal from tweet text"),
        ("📊", "Matplotlib", "Accuracy comparison bar charts and visualizations"),
        ("🎨", "Streamlit", "Web UI framework for Python ML applications"),
    ]
    col1, col2 = st.columns(2)
    for i, (icon, name, desc) in enumerate(techs):
        with col1 if i % 2 == 0 else col2:
            with st.container():
                st.markdown(f"**{icon} {name}**")
                st.caption(desc)
                st.markdown("---")
