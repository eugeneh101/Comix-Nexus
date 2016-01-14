from web_scrapping import * 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



def clean_text(list_o_text):
    docs = [''.join([char if char not in punctuation else ' ' for char in 
                     comic]) for comic in list_o_text]

    # remove punctuation from string
    docs = [word_tokenize(comic) for comic in docs]
    # make string into list of words

    # 3. Strip out stop words from each tokenized document.
    stop = set(stopwords.words('english'))
    stop.update(punctuation)
    other_words = ['cite', 'cite_note', 'cite_ref', 'class', 'href', 'id', 
                   'redirect', 'ref', 'refer', 'span', 'sup', 'title', 'wiki']
    stop.update(other_words)
    docs = [[word for word in words if word.strip(punctuation) not in stop] 
            for words in docs]
    # remove stop words
    
    # Stemming / Lemmatization
    # 1. Stem using both stemmers and the lemmatizer
    #porter = PorterStemmer()
    snowball = SnowballStemmer('english')
    #wordnet = WordNetLemmatizer()
    #docs_porter = [[porter.stem(word) for word in words] for words in docs]
    docs_snowball = [[snowball.stem(word) for word in words] for words in docs]
    #docs_wordnet = [[wordnet.lemmatize(word) for word in words] for words in docs]
    docs = [' '.join(doc) for doc in docs_snowball]
    # for each document, it becomes a long string
    return docs

docs = clean_text(comic_text)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidfed_matrix = tfidf_vectorizer.fit_transform(docs)
# docs must be list of strings
tfidf_vectorizer.vocabulary_

cosine_similarities = cosine_similarity(tfidfed_matrix.todense(), 
	tfidfed_matrix.todense())

for i, link in enumerate(links):
    for j, link in enumerate(links):
        print i, j, cosine_similarities[i, j]

print cosine_similarities.shape
print len(tfidf_vectorizer.vocabulary_)


# save the data
import cPickle as pickle
with open('links.pkl', 'w') as f:
    pickle.dump(links, f)
with open('vectorizer.pkl', 'w') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('matrix.pkl', 'w') as f:
    pickle.dump(tfidfed_matrix, f)
with open('comic_text.pkl', 'w') as f:
    pickle.dump(comic_text, f)