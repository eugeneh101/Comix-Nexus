import os.path
from recommendation_engines import *
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


@app.route('/recommender-output', methods=['GET', 'POST'])
def recommender_output():
#    text = str(request.form['user_input'])
    recommend = cos_sim_recommender(str(session['input_string']), 
        tfidf_vectorizer, tfidfed_matrix, links)
    image_name = recommend + '.jpg'
#    return ''' <img src = /static/ ''' + image_name + ''' >'''

    return render_template('results.html', picture=image_name, input_text=str(session['input_string']))
    return str(recommend)


@app.route('/nmf')
def show_all():
    return render_template('nmf.html', all_comic_pics=title_to_jpg(links))


@app.route('/nmf/<comic_title>')
def show_user_profile(comic_title):
    # show the user profile for that user
    #return 'User {{username}}'

    comic = [link for link in title_cleaner(links) if link == comic_title][0]

    """
    nmf1 = nmf_recommender_1(comic, H_sklearn, links, tfidf_vectorizer, 
        tfidfed_matrix, W_sklearn)
    img_name1 = title_cleaner([nmf2])[0] + '.jpg'
    """
    nmf2 = nmf_recommender_2(comic, H_sklearn, links, tfidf_vectorizer, 
        tfidfed_matrix, W_sklearn)
    img_name2 = title_cleaner([nmf2])[0] + '.jpg'
    return render_template('nmf_recommendations.html', comic_title=comic_title, 
        recommend2=nmf2, pic2=img_name2)


    """
    nmf2 = nmf_recommender_2(comic, H_sklearn, links, tfidf_vectorizer, 
        tfidfed_matrix, W_sklearn)
    img_name2 = title_cleaner([nmf2])[0] + '.jpg'
    return render_template('nmf_recommendations.html', comic_title=comic_title, recommend1=nmf1, 
        pic1=img_name1, recommend2=nmf2, pic2=img_name2)
    """






# set the secret key.  keep this really secret:
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

if __name__ == '__main__':
    # checks to see if you have web scrapped previously
    if not os.path.isfile('links.pkl'):
        from tfidf_corpus import *

    # create our TF-IDF variables
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





"""
cos_sim_recommender(raw_input('type what you want> '), tfidf_vectorizer, tfidfed_matrix, links)
bob fighting stone my patience is stone and my will is of the stars
#> 'All-Star_Superman'

draw_dendrogram(links, tfidfed_matrix)

good_comic = raw_input('What comic do you want a similar one? ')
Wanted_(comics)
bad_comics = raw_input('Comics you hate? Separate by commas ').split(',')
Watchmen, The_Dark_Knight_Returns, Kingdom_Come_(comics)
bad_comics = [comic.strip() for comic in bad_comics]
cos_sim_c2c(links, tfidfed_matrix, good_comic, bad_comics)
#> ['All-Star_Superman', 'The_Authority', 'The_Sandman_(Vertigo)']

nmf_recommender_1(raw_input('type what you want> '), H_sklearn, links, tfidf_vectorizer, tfidfed_matrix, W_sklearn)
bob fighting stone my patience is stone and my will is of the stars
#> 'Wanted_(comics)'

nmf_recommender_2(raw_input('type in for nmf recommendation> '), H_sklearn, 
    links, tfidf_vectorizer, tfidfed_matrix, W_sklearn)
bob fighting stone my patience is stone and my will is of the stars
#> 'Wanted_(comics)'


from recommendation_models import *
links, tfidf_vectorizer, tfidfed_matrix = load_data()
kmeans = make_kclusters(tfidf_vectorizer, tfidfed_matrix)
print_kclusters(kmeans, links)
cos_sim_rc2c(links, tfidfed_matrix)
nmf = NMF(n_components=10)
W_sklearn = nmf.fit_transform(tfidfed_matrix)
H_sklearn = nmf.components_
describe_nmf_results(tfidfed_matrix, H_sklearn, W_sklearn, tfidf_vectorizer)
print_nmf_clusters(H_sklearn, links, W_sklearn)


"""
