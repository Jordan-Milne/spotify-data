import os

from dotenv import load_dotenv, find_dotenv
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# Loading API keys
load_dotenv(find_dotenv())

cid = os.environ['CLIENT_ID']
secret = os.environ['CLIENT_SECRET']

# Authenticating API Credentials
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

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

# Loading my personal playlist and my gf's personal playlists (top songs for 2018 & 2019 curated by spotify)
play_list1_id = os.environ['pl1_id']
play_list2_id = os.environ['pl2_id']
play_list3_id = os.environ['pl3_id']
play_list4_id = os.environ['pl4_id']

df1 = analyze_playlist("spotify", play_list1_id,'jordan')
df2 = analyze_playlist("spotify", play_list2_id,'kathryn')
df3 = analyze_playlist("spotify", play_list3_id,'kathryn')
df4 = analyze_playlist("spotify", play_list4_id,'jordan')

# merge
df = pd.concat([df1, df2,df3,df4], ignore_index = True)

# export

df.to_csv('playlists.csv', index=False)
