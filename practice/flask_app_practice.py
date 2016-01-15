"""
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
	return 'Hello World!!!!'

@app.route('/user/username')
def show_user_profile(username):
    # show the user profile for that user
    return 'User bob'

@app.route('/post')
def show_post():
    # show the post with the given id, the id is an integer
    return 'Post 1'

@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')
def about():
    return 'The about page'


if __name__ == '__main__':
	app.run()
"""
from flask import Flask, session, redirect, url_for, escape, request
import time
app = Flask(__name__)

@app.route('/')
def index():
    if 'username' in session:
        return 'Logged in as %s' % escape(session['username'])
    return 'You are not logged in'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
    	#print "hi this line is working"
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    #print 'line 2 is working'
    return '''
        <form action="" method="post">
            <p><input type=text name=username><br>
            <input type=submit value=Login></p>
        </form>
    '''

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    #return 'User {{username}}'
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id


@app.route('/tester')
def tester():
	time.sleep(2)
	return redirect(url_for('show_user_profile', username=session['username']))

@app.route('/tester_two', methods=['GET', 'POST'])
def tester_two():
    if request.method == 'POST':
    	print "line 1 is working"
        session['number_from_tester'] = request.form['post_number']
        return redirect(url_for('show_post', post_id=session['number_from_tester']))
    print "line 2 is working"
    return '''
        <form action="" method="post">
            <p><input type=text name=post_number>
            <p><input type=submit value=Number__>
        </form>
    '''


@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('bye'))
#    return redirect(url_for('index'))

@app.route('/bye')
def bye():
    # remove the username from the session if it's there
    return 'bye bye!'
    #redirect(url_for('index'))


# set the secret key.  keep this really secret:
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

#go to localhost:5000



"""
def dict_to_html(d):
    return '<br>'.join('{0}: {1}'.format(k, d[k]) for k in sorted(d))


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
    X = vectorizer.transform(np.array([text]))
    prediction = model.predict(X)
    word_counts = Counter(text.lower().split())
    page = 'There are {0} words.<br><br> And Your topic is most likely {1}\
    <br><br> Individual word counts:<br> {2}'
    return page.format(len(word_counts), prediction, dict_to_html(word_counts))#.format(len(word_counts), prediction, dict_to_html(word_counts))
"""



"""
def more():
    return render_template('starter_template.html')
"""