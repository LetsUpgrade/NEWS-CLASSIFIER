from flask import Flask, render_template, request, url_for, redirect
import pickle
import numpy as np
from flask_cors import CORS
import newspaper
from newspaper import Article
import urllib

app = Flask(__name__, template_folder='venv/templates', static_folder="venv/static")
CORS(app)

with open('model.pickle', 'rb') as target:
    model = pickle.load(target)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        url = request.get_data(as_text=True)[5:]
        url = urllib.parse.unquote(url)
        article = Article(str(url))
        article.download()
        article.parse()
        article.nlp()
        news = article.summary
        pred = model.predict([news])
        if pred == 0:
            return render_template("index1.html")

        elif pred == 1:
            return render_template("index2.html")


@app.route('/close', methods = ['GET', 'POST'])
def close():
    return  render_template("index.html")


if __name__ == "__main__":
    app.debug = True
    app.run()
