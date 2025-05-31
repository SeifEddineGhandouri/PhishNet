# PhishNet - Real-Time Phishing URL Detection System with Flask Dashboard

import re
import whois
import socket
import requests
import tldextract
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

# Example URL dataset (phishing and legitimate)
data = pd.read_csv("urls_dataset.csv")  # Columns: url, label (0: safe, 1: phishing)

# --- Feature Extraction Functions ---
def get_url_length(url):
    return len(url)

def count_dots(url):
    return url.count('.')

def get_domain_entropy(url):
    from collections import Counter
    import math
    domain = tldextract.extract(url).domain
    prob = [v / len(domain) for v in Counter(domain).values()]
    entropy = -sum(p * math.log(p) for p in prob)
    return entropy

def is_https(url):
    return 1 if url.startswith("https") else 0

# --- Feature Engineering ---
def extract_features(df):
    df['url_length'] = df['url'].apply(get_url_length)
    df['dots'] = df['url'].apply(count_dots)
    df['has_ip'] = df['url'].apply(lambda u: 1 if re.search(r"\d+\.\d+\.\d+\.\d+", u) else 0)
    df['entropy'] = df['url'].apply(get_domain_entropy)
    df['https'] = df['url'].apply(is_https)
    return df

data = extract_features(data)
X = data[['url_length', 'dots', 'has_ip', 'entropy', 'https']]
y = data['label']

# --- Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Flask Web Interface ---
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    url_checked = None
    if request.method == 'POST':
        url = request.form['url']
        url_checked = url
        features = pd.DataFrame([{
            'url_length': get_url_length(url),
            'dots': count_dots(url),
            'has_ip': 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0,
            'entropy': get_domain_entropy(url),
            'https': is_https(url)
        }])
        prediction = model.predict(features)[0]
        result = "⚠️ Phishing detected!" if prediction == 1 else "✅ Safe URL"
    return render_template('index.html', result=result, url=url_checked)

if __name__ == '__main__':
    app.run(debug=True)
