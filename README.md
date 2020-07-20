# Disaster response pipeline

## Installation 
Dependencies:
Python, NumPy, Pandas,Json,Flask,Plotly,Sklearn,Nltk,SQLAlchemy,sys,re,pickle

## File Descriptions
There're 2 disaster data(messages and relevant categories) provided by Figure8. We need to build a machine learning pipeline to categorize the messages 
and send it to appropriate organinzations to support
We also need to build a web app which enables the staff to input new messages and get classification result.The web app will display some visualization of the data

## File structure

- app
   - template

      - master.html  # main page of web app

      - go.html  # classification result page of web app

   - run.py  # Flask file that runs app

- data
   - disaster_categories.csv  # data to process 

   - disaster_messages.csv  # data to process

   - process_data.py

- models
   - train_classifier.py

- README.md 

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements
Must give credit to Figure 8 for the data. You can find the Licensing for the data and other descriptive information at Udacity.
