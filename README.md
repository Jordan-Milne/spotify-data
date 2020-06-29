# spotify-data
## Work-In-Progress

---

### The motivation behind this project is that my girlfriend and I have very different tastes in music. The main assumption I used is that all the songs on my girlfriends playlists  are songs I do not like (which is typically very true). Using songs from her playlists as songs I do not like and songs from my playlists as songs I like, I trained a classification algorithim learn what songs I do and do not like. Now I can feed the model playlists from all over spotify and it will screen them for songs I would like.

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
### Step 6. ROC AUC
### Step 7. Personal Validation
### Step 6. R
