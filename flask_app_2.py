from recommendation_models import *
from flask import Flask, session, redirect, url_for, escape, request, render_template
import time


app = Flask(__name__)

"""

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
"""


def title_cleaner(lst_o_titles):
    def punctuation_cleaner(title):
        return ''.join([c if c not in [':'] else "" for c in title])
        # remove colon from comic book titles
    comics = [comic.split('/')[-1] for comic in lst_o_titles]
    comics = [punctuation_cleaner(comic) + '.jpg' for comic in comics]
    return comics

@app.route('/')
def submission_page():
    return '''
        <form action="/login" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['user_input']

        return redirect(url_for('recommender_input'))
    return '''
        <form action="" method="post">
            <p><input type=text name=username>
            <br>
            <input type=submit value=Login></p>
        </form>
    '''

@app.route('/recommender-input', methods=['GET', 'POST'])
def recommender_input():
    if request.method == 'POST':
        session['input_string'] = request.form['input_string']
        return redirect(url_for('recommender_output'))
    return '''
        <form action="" method="post">
            <p>Type your idea of a good comic book: <br>
            <input type=text name=input_string><br>
            <input type=submit value=Enter_Now></p>
        </form>
    '''

@app.route('/all-comics')
def show_all():
    return render_template('all_comics.html', all_comic_pics=title_cleaner(links))

"""
<button type='button' class='btn btn-lg btn-danger' onclick="window.location='/openfire'; return false;">Open Fire!</button> 
"""

@app.route('/recommender-output', methods=['GET', 'POST'])
def recommender_output():
#    text = str(request.form['user_input'])
    recommend = cos_sim_recommender(str(session['input_string']), 
        tfidf_vectorizer, tfidfed_matrix, links)
    image_name = recommend + '.jpg'
#    return ''' <img src = /static/ ''' + image_name + ''' >'''

    return render_template('results.html', picture=image_name, input_text=str(session['input_string']))
    return str(recommend)










    """
    X = vectorizer.transform(np.array([text]))
    prediction = model.predict(X)
    word_counts = Counter(text.lower().split())
    page = 'There are {0} words.<br><br> And Your topic is most likely {1}\
    <br><br> Individual word counts:<br> {2}'
    """

# set the secret key.  keep this really secret:
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

if __name__ == '__main__':
    links, tfidf_vectorizer, tfidfed_matrix = load_data()
    kmeans = make_kclusters(tfidf_vectorizer, tfidfed_matrix)
    print_kclusters(kmeans, links)
    cos_sim_rc2c(links, tfidfed_matrix)
    nmf = NMF(n_components=10)
    W_sklearn = nmf.fit_transform(tfidfed_matrix)
    H_sklearn = nmf.components_
    describe_nmf_results(tfidfed_matrix, H_sklearn, W_sklearn, tfidf_vectorizer)
    print_nmf_clusters(H_sklearn, links, W_sklearn)


    app.run(host='0.0.0.0', port=8080, debug=True)
