import sys
from sqlalchemy import create_engine
import re
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    '''
    load data from database
    Args:
    database_filepath=file path of cleaned database
    Return:
    X:id,message,original,genre
    Y:categories
    category_names 
   
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('merged',engine)
    X = df.message.values
    Y=df.iloc[:,4:].values
    category_names= df.columns[4:]
    return X,Y,category_names


def tokenize(text):
    '''
    tokenize text in messages,remove stop words and non-text
    '''
    stop_words = stopwords.words("english")
    lemmatizer=WordNetLemmatizer()

    text=re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    tokens=word_tokenize(text)
    
    tokens=[lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
    return (tokens)


def build_model():
    '''
    build machine learning pipeline with CountVectorizer and TdidfTransformer, 
    choose random forest classifer as default classification model to fit training data
    use gridsearch CV to optmize the model performance
    '''
    model_pipeline = Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer()),
            ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=10),n_jobs=-1))             
        ]) 
    
    parameters={
    'clf__estimator__n_estimators':[25,50]        
    }

    model=GridSearchCV(model_pipeline,param_grid=parameters,n_jobs=-1)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    select 1 column to test accuracy score of the model
    '''
    Y_pred=model.predict(X_test)
    print(classification_report(Y_test[:,35], Y_pred[:,35], target_names=category_names))
    print(accuracy_score(Y_test[:,35], Y_pred[:,35]))


def save_model(model, model_filepath):
    '''
    save the optimized model as pkl file
    
    Arg:model_filepath=file path to save the model to
    '''
    with open(model_filepath,'wb') as f:
       pickle.dump(model,f)

def main():
    '''
    take the input file path of the database to fit,train and evaluate model, and save optimized model to assigned path
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()