import os.path
from recommendation_engines import *
from flask import Flask, session, redirect, url_for, escape, request, render_template
import time


app = Flask(__name__)

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
            <p>Type your idea of a good comic book: 
            <br>
            <input type=text name=input_string>
            <br>
            <input type=submit value=Enter_Now></p>
        </form>
    '''


@app.route('/recommender-output', methods=['GET', 'POST'])
def recommender_output():
    text = session['input_string']

    cos_recommendation = cos_sim_recommender(text, tfidf_vectorizer,
        tfidfed_matrix, links)
    cos_recommendation = title_cleaner([cos_recommendation])[0]

    nmf1 = nmf_recommender_1(text, H_sklearn, links, tfidf_vectorizer, 
        tfidfed_matrix, W_sklearn)
    nmf1_recommendation = title_cleaner([nmf1])[0]
    nmf2 = nmf_recommender_2(text, H_sklearn, links, tfidf_vectorizer, 
        tfidfed_matrix, W_sklearn)
    nmf2_recommendation = title_cleaner([nmf2])[0]
    return render_template('comic_recommendations.html', input_text=
        str(session['input_string']), cos_recommendation=cos_recommendation,
        nmf1_recommendation=nmf1_recommendation, nmf2_recommendation=
        nmf2_recommendation)
#    return ''' <img src = /static/ ''' + image_name + ''' >'''



@app.route('/c2c')
def show_all():
    return render_template('comic_2_comic_recommender.html', all_comics=
        title_cleaner(links))


@app.route('/c2c/<comic_title>')
def comic_2_comic_recommendations(comic_title):
    comic = title_to_link(comic_title, links)
    cos = cos_sim_c2c(links, tfidfed_matrix, comic, rejected_comics=[], 
        how_many=3)
    cos_recommendations = title_cleaner(cos)

    nmf = nmf_c2c_in(comic, links, W_sklearn, how_many=3, rejected_comics=[])
    nmf_recommendations = title_cleaner(nmf)

    return render_template('comic_2_comic_recommendations.html', comic_title=
        comic_title, cos_recommendations=cos_recommendations,  
        nmf_recommendations=nmf_recommendations)

"""
@app.route('/random', methods=['GET', 'POST'])
def random():
    def show_page():
        cos_comic, cos_similar_comics = cos_sim_rc2c(links, tfidfed_matrix)
        nmf_comic, nmf_similar_comics = nmf_sim_rc2c(links, tfidfed_matrix, 
            W_sklearn)
        return '''
            <h1 style="color:red">Random Comics!</h1>
            <form action="" method="post">
                <p> Some More Random Comics? 
                <br>
                <input type=submit value=Yes></p>
                <input type=submit value="No?"></p>
            </form>
            <p> Cosine Similarity Random Comic: ''' + cos_comic + ''' </p>
            <img src = '''"/static/" + title_cleaner([cos_comic])[0] + ".jpg"'''>

        '''
    if request.method == 'POST':
        return show_page()
    return show_page()
"""

@app.route('/random', methods=['GET', 'POST'])
def random():
    cos_comic, cos_similar_comics = cos_sim_rc2c(links, tfidfed_matrix)
    nmf_comic, nmf_similar_comics = nmf_sim_rc2c(links, tfidfed_matrix, 
        W_sklearn)
    if request.method == 'POST':
        return render_template('random.html', 
            cos_comic=title_cleaner([cos_comic])[0], 
            cos_similar_comics=title_cleaner(cos_similar_comics),
            nmf_comic=title_cleaner([nmf_comic])[0],
            nmf_similar_comics= title_cleaner(nmf_similar_comics))
    return render_template('random.html', 
        cos_comic=title_cleaner([cos_comic])[0], 
        cos_similar_comics=title_cleaner(cos_similar_comics),
        nmf_comic=title_cleaner([nmf_comic])[0],
        nmf_similar_comics= title_cleaner(nmf_similar_comics))


#    return ''' <img src = /static/ ''' + image_name + ''' >'''
#    return '''<img src='''"/static/" + title_cleaner([cos_comic])[0] + ".jpg"'''>'''




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
