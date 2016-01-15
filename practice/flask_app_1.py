from collections import Counter
from flask import Flask, request
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import cPickle as pickle

app = Flask(__name__)



def dict_to_html(d):
    return '<br>'.join('{0}: {1}'.format(k, d[k]) for k in sorted(d))

def recommender(input_string, tfidfed_matrix, links):
    stop = set(stopwords.words('english'))
    stop.update(punctuation)

    user_input = word_tokenize(input_string)
    snowball = SnowballStemmer('english')
    user_snowball = [snowball.stem(word) for word in user_input if word not in stop]
    # remove useless words
    #lowercase words, keeps only root of words
    user = [str(' '.join(user_snowball))]
    # converts list of words into string
    recommend = cosine_similarity(tfidf_vectorizer.transform(user).todense(), tfidfed_matrix.todense())
    # x-axis is the original data, y-axis is the query (raw_input) you put in
    # docs must be list of strings
    title_index = np.argmax(recommend)
    # find max similarity
    return links[title_index].split('/')[-1]
    # recommendation!



# Form page to submit text
@app.route('/')
def submission_page():
    return '''
        <form action="/word_counter" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''


# My word counter app
@app.route('/word_counter', methods=['POST'] )
def word_counter():
    text = str(request.form['user_input'])
    recommend = recommender(text, tfidfed_matrix, links)
    return str(recommend)
    """
    X = vectorizer.transform(np.array([text]))
    prediction = model.predict(X)
    word_counts = Counter(text.lower().split())
    page = 'There are {0} words.<br><br> And Your topic is most likely {1}\
    <br><br> Individual word counts:<br> {2}'
    """

#    page = 'There are {0} words.<br><br> And Your topic is most likely {1}\
#    <br><br> Individual word counts:<br> {1}'
#    return page.format(len(word_counts), prediction, dict_to_html(word_counts))

if __name__ == '__main__':
    with open('links.pkl') as f:
        links = pickle.load(f)
    with open('vectorizer.pkl') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('matrix.pkl') as f:
        tfidfed_matrix = pickle.load(f)
    app.run(host='0.0.0.0', port=8080, debug=True)
