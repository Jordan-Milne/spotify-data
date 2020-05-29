# spotify-data
## Work-In-Progress

---

### The motivation behind this project is that my girlfriend and I (*~~argue~~*) have conversations  sometimes about who gets to control the music. This got me thinking about how similar is our taste in music. I started playing around with the Spotify API and decided to see how well a ML model could classify our top 200 songs (each) of the last two years using data provided from the Spotify API.  To make it a *real* challange I did not use lyrics, genre, artist, or song title as features in the model."

## The work flow of the project follows these steps:

### Step 1. Credentials
* Store secret credentials in a .env file so public (github) cannot see
### Step 2. Creating the DataFrame
* Create a function that uses the spotify API to get data from each song in playlist and create a DataFrame
### Step 3. Minor Preprocessing
* Splitting the data into train and test sets
* Standard scaling the quantitive features and label encoding categorical features
### Step 4. Modeling
* Analyzing multiple baseline ML model_selection
* Optimizing promising ML model using grid search over the hyperparameters
### Step 5. Feature Importances
### Step 5. ROC AUC
* to do
