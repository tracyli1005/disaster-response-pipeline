import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('merged', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs=[]
    data_one=[]
    data_two=[]
    data_three=[]
    data_four=[]
    
    data_one.append(Bar(
              x=genre_names,
              y=genre_counts
             ))
    
    layout_one=dict(title= 'Distribution of Message Genres',
                yaxis=dict(title= "Count"),
                xaxis=dict(title="Genre")
                   )
    
    direct_result_counts=df[df['genre']=='direct'].iloc[:,4:].sum().values.tolist()
    news_result_counts=df[df['genre']=='news'].iloc[:,4:].sum().values.tolist()
    social_result_counts=df[df['genre']=='social'].iloc[:,4:].sum().values.tolist()
    result_names=df.columns[4:].tolist()
    
    data_two.append(Bar(
              x=result_names,
              y=direct_result_counts
             ))
    
    layout_two=dict(title= 'Distribution of Categories of Direct Messages',
                yaxis=dict(title= "Count"),
                xaxis=dict(title="Categories")
                   )
    
    data_three.append(Bar(
              x=result_names,
              y=news_result_counts
             ))
    
    layout_three=dict(title= 'Distribution of Categories of News Messages',
                yaxis=dict(title= "Count"),
                xaxis=dict(title="Categories")
                   )
    
    data_four.append(Bar(
              x=result_names,
              y=social_result_counts
             ))
    
    layout_four=dict(title= 'Distribution of Categories of Social Messages',
                yaxis=dict(title= "Count"),
                xaxis=dict(title="Categories")
                   )
    graphs.append(dict(data=data_one, layout=layout_one))
    graphs.append(dict(data=data_two, layout=layout_two))
    graphs.append(dict(data=data_three, layout=layout_three))
    graphs.append(dict(data=data_four, layout=layout_four))
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
