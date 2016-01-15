from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import cPickle as pickle
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import NMF

def load_data():
	with open('links.pkl') as f:
	    links = pickle.load(f)
	with open('vectorizer.pkl') as f:
	    tfidf_vectorizer = pickle.load(f)
	with open('matrix.pkl') as f:
	    tfidfed_matrix = pickle.load(f)
	return links, tfidf_vectorizer, tfidfed_matrix

def string_cleaner(input_string):
    """Tokenizes input string annd outputs a list of 1 string
    """
    stop = set(stopwords.words('english'))
    stop.update(punctuation)

    user_input = word_tokenize(input_string)
    snowball = SnowballStemmer('english')
    user_snowball = [snowball.stem(word) for word in user_input if word
                     not in stop]
    # remove useless words; lowercase words, keeps only root of words
    user = [str(' '.join(user_snowball))]
    # converts list of words into string
    return user


def cos_sim_recommender(input_string, tfidf_vectorizer, tfidfed_matrix, links):
    """Makes recommendation based on cosine similarity of TF-IDF matrix
    """ 
    user = string_cleaner(input_string)
    recommend = cosine_similarity(tfidf_vectorizer.transform(user).todense(), 
                                  tfidfed_matrix.todense())
    # x-axis is the original data, y-axis is the query (raw_input) you put in
    # docs must be list of strings
    title_index = np.argmax(recommend)
    # find max similarity
    return links[title_index].split('/')[-1]
    # recommendation!

def make_kclusters(tfidf_vectorizer, tfidfed_matrix, n_clusters = 8):
    """Apply k-means clustering to the articles
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(tfidfed_matrix)
    features = tfidf_vectorizer.get_feature_names()
    # features is list of words

    # 2. Print out the centroids.
    #print "cluster centers:"
    #print kmeans.cluster_centers_

    # 3. Find the top 10 features for each cluster.
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-19:-1]
    print "top features for each cluster:"
    for num, centroid in enumerate(top_centroids):
        print "%d: %s" % (num, ", ".join(features[i] for i in centroid))
    return kmeans

def print_kclusters(kmeans, links):
    """Print KMean Clusters
    """
    titles = np.array([link.split('/')[-1] for link in links])
    for index_num, label in enumerate(set(kmeans.labels_)): 
    	#index_num isn't true label
        indices = np.where(kmeans.labels_ == label)[0]
        print index_num
        for index in indices:
            print titles[index]
        print ""



def draw_dendrogram(links, tfidf_matrix):
    """Hierarchical clustering plot
    """
	# distxy = squareform(pdist(tfidf_matrix.todense(), 
	#	metric='cosine'))
    link = linkage(tfidf_matrix.todense(), method='complete', metric='cosine')
    dendro = dendrogram(link, color_threshold=1.5, leaf_font_size=9, labels=
                    [link.split('/')[-1] for link in links], leaf_rotation=90)
    plt.show()
    


def cos_sim_c2c(links, tfidfed_matrix, input_string, rejected_comics=[], 
	how_many = 3):
    """Comic to comic book recommendation using cosine similarity
    Given a comic book, recommend a similar comic book
    """
    titles = np.array([link.split('/')[-1] for link in links])
    try:
        which_comic = np.where(titles == input_string)[0][0]
    except:
        return 'Your preferred comic title is not in this database'
    distxy = squareform(pdist(tfidfed_matrix.todense(), metric='cosine'))
    closest_comics = titles[np.argsort(distxy[which_comic])][1:]
    best_n_comics = []
    for comic in closest_comics:
        if comic in rejected_comics:
            continue
        else:
            best_n_comics.append(comic)
        if len(best_n_comics) == how_many:
            return best_n_comics
    return best_n_comics


def cos_sim_rc2c(links, tfidfed_matrix):
    """Random comic to comic using cosine similarity
    Given a comic book, find a similar comic book
    """
    titles = np.array([link.split('/')[-1] for link in links])
    random_comic = random.choice(titles)
    # over_recommended_comics = 'The_Sandman_(Vertigo), Watchmen, Saga_(comic_book)'
    output = "A random comic: " + random_comic + "; Similar comics: "
    return output + ', '.join(cos_sim_c2c(links, tfidfed_matrix, random_comic,
        '', how_many = 3))   


# Non-negative Matrix Facttorization
def reconst_mse(target, left, right):
    return (np.array(target - left.dot(right))**2).mean()

def describe_nmf_results(document_term_mat, H, W, vectorizer, n_top_words = 15):
    """Prints top tokenized words for each cluster
    """
    feature_words = vectorizer.get_feature_names()
    print("Reconstruction error: %f") %(reconst_mse(document_term_mat, W, H))
    for topic_num, topic in enumerate(H):
        print("Topic %d:" % topic_num)
        print(" ".join([feature_words[i] \
                for i in topic.argsort()[:-n_top_words - 1:-1]]))
    return 

def print_nmf_clusters(H_sklearn, links, W_sklearn):
    """Prints comic books in each cluster
    """
    for cluster_index in range(len(H_sklearn)):
        titles = np.array([link.split('/')[-1] for link in links])
        comics_in_cluster = []
        print cluster_index
        for ith, comic in enumerate(W_sklearn):
            if cluster_index == np.argmax(comic):
                print titles[ith]
        print ""

def word_to_index_in_vectorizer(lst_o_words, tfidf_vectorizer):
    """Convert a list of words into a list of word indices from vectorizer
    """ 
    word_indices = []
    for word in lst_o_words[0].split():
        try:
            word_indices.append(tfidf_vectorizer.vocabulary_[word])
        except:
            continue
    return word_indices

def nmf_recommender_1(input_string, H_sklearn, links, tfidf_vectorizer, 
	tfidfed_matrix, W_sklearn):
    """User inputs a string and outputs comic book recommendation
    This is a NMF comic book recommender that competes against cosine similarity
    recommender above. NMF recommender is 'weaker' than cosine similarity in 
    that a string copied from comic book wiki page and pasted into recommender
    will have correct comic book recommenedation by cosine similarity but often
    not NMF because of how NMF works. Comments below 
    """
    user = string_cleaner(input_string)
    # get tokenized words from input string
    word_indices = word_to_index_in_vectorizer(user, tfidf_vectorizer)
    # for each word, get the index from vectorizer
    average_topics = [0] * H_sklearn.shape[0]
    for index in range(len(average_topics)):
        average_topics[index] = H_sklearn[index][word_indices].mean()
    # for each word, get the "average" topic that the word would appear in
    guess = np.argmax(cosine_similarity(average_topics, W_sklearn))
    # look at the average topics generated from words and perform cosine
    # similarity to recommend the closest comic book
    return np.array([link.split('/')[-1] for link in links])[guess]
    
def nmf_recommender_2(input_string, H_sklearn, links, tfidf_vectorizer, 
	tfidfed_matrix, W_sklearn):
    user = string_cleaner(input_string)
    # get tokenized words from input string
    word_indices = word_to_index_in_vectorizer(user, tfidf_vectorizer)
    # for each word, get the index from vectorizer    
    
    top_topics = []
    for index in word_indices:
        top_topics.append(np.argmax(H_sklearn[:, index]))
    # get top topics for each word
    
    ranked_comics = []
    for index in top_topics:
        order = W_sklearn[:, index].copy().argsort()[::-1]
        ranks = order.argsort()
        ranked_comics.append(list(ranks))
    # rank all comic books for each topic; lower is better
    
    guess = np.argmin(np.mean(np.array(ranked_comics), axis=0))
    # find the comic that is relatively low (closest) on each topic
    return np.array([link.split('/')[-1] for link in links])[guess]






if __name__ == "__main__":
	pass

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