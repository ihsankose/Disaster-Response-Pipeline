# Disaster Response Pipeline Project

This project uses machine learning to classify messages based on their categories total of 26, such as food-water-aid related or information or request etc.
First it cleans and wrangle training datasets.
Then it uses CountVectorizer, TfidfTransformer to transform and RandomForestClassifier to classify. Furthermore GridSearchCV is used to optimized parameters.
Finally, a web app is launched where the user can view training messages analysis and an option to input a new message to be classified.

### Instructions:

1. If the model going to be trained with a new dataset,      
   run the following commands in the project's root directory to set up your database and model.      
        - To run ETL pipeline that cleans data and stores in database      
            `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`      
        - To run ML pipeline that trains classifier and saves     
            `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`      

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. To see the web app go to http://0.0.0.0:3001/
