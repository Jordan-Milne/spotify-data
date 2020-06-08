from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv, find_dotenv

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold,ShuffleSplit
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve
from catboost import CatBoostClassifier

load_dotenv(find_dotenv())
cid = os.environ['CLIENT_ID']
secret = os.environ['CLIENT_SECRET']
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET','POST'])
def index():
    rocauc = ''
    args = request.form
    def spot_model(pl_1,pl_2):

        # Function to make dataframe from playlist
        def analyze_playlist(creator, playlist_id,person):

            # Create empty dataframe
            column_list = ["artist","album","track_name",  "track_id","explicit","popularity", "danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]

            main_df = pd.DataFrame(columns = column_list)

            # Looping through playlist and appending features

            playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
            for track in playlist:        # Create empty dict
                features = {}        # Get metadata
                features["artist"] = track["track"]["album"]["artists"][0]["name"]
                features["album"] = track["track"]["album"]["name"]
                features["track_name"] = track["track"]["name"]
                features["track_id"] = track["track"]["id"]
                features["explicit"] = track['track']['explicit']
                features["popularity"] = track['track']['popularity']

                # collecting meta data (audio features)
                audio_features = sp.audio_features(features["track_id"])[0]
                for feature in column_list[6:]:
                    features[feature] = audio_features[feature]

                # merge
                track_df = pd.DataFrame(features, index = [0])
                main_df = pd.concat([main_df, track_df], ignore_index = True)
            main_df['person'] = person

            return main_df


        df1 = analyze_playlist("spotify", str(pl_1),'user1')
        df2 = analyze_playlist("spotify", str(pl_2),'user2')


        # merge
        df = pd.concat([df1,df2], ignore_index = True)


        # Manually Label Encoding the person column
        df['person'] = df['person'].map({'user1': 1, 'user2': 0})

        # Seperating target out for train-test-split
        target = 'person'
        y = df[target]
        X = df.drop(target, axis=1)

        # Train-test-split and KFold
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        kf = KFold(n_splits=5, shuffle=True)

        # From Sklearn-pandas package I use the DataFrame mapper to significantly simplify preprocessing. This selects
        # the features to include and any preprocessing that needs to be done to the data.
        mapper = DataFrameMapper([
             (['danceability'], StandardScaler()),
             ('explicit', LabelEncoder()),
             (['energy'], [StandardScaler()]),
             (['popularity'], [StandardScaler()]),
             (['key'], [StandardScaler()]),
             (['loudness'],  [StandardScaler()]),
             (['mode'], StandardScaler()),
             (['speechiness'], StandardScaler()),
             (['instrumentalness'], StandardScaler()),
             (['liveness'],  StandardScaler()),
             (['valence'], StandardScaler()),
             (['tempo'],  StandardScaler()),
             (['duration_ms'],  StandardScaler()),
             ], df_out=True)

        Z_train = mapper.fit_transform(X_train)
        Z_test = mapper.transform(X_test)

        # Running the tuned model to see how the loss graph looks
        model = CatBoostClassifier(
            iterations=1000,
            early_stopping_rounds=10,
            depth = 4,
            l2_leaf_reg = 5,
            learning_rate = 0.03,
            custom_loss=['AUC', 'Accuracy']
        )

        model.fit(
            Z_train,
            y_train,
            eval_set=(Z_train, y_train),
            verbose=False,
            plot=False)

        # calculate the fpr and tpr for all thresholds of the classification
        probs = model.predict_proba(Z_test)
        preds = probs[:,1]
        fpr, tpr, threshold = roc_curve(y_test, preds)
        roc_auc = round(auc(fpr, tpr),2)
        return roc_auc
    if request.method == 'POST':
        rocauc = str(spot_model(args.get('playinglist1'), args.get('playinglist2')))
        # rocauc = '88.5'

    return render_template('index.html', rocauc=rocauc)



# @app.route('/result', methods=['GET','POST'])
# def result():
#
#
#
#     return render_template('result.html',rocauc=rocauc)



if __name__ == '__main__':
    app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', debug=True, port=80)
