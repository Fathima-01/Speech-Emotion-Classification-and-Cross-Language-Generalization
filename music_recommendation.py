import webbrowser

def recommend_song(emotion):
    song_database = {
        "happy": "https://open.spotify.com/track/1lCRw5FEZ1gPDNPzy1K4zW",  # Example: Happy Song
        "sad": "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",    # Example: Sad Song
        "angry": "https://open.spotify.com/track/0x7RSxYdxOuyRMALNwNQq6",  # Example: Angry Song
        "neutral": "https://open.spotify.com/track/7B5Npv3XBHzy42L6Vilo5p", # Example: Neutral Song
        "fear": "https://open.spotify.com/track/2TpxZ7JUBn3uw46aR7qd6V"    # Example: Fear Song
    }

    song_link = song_database.get(emotion, None)

    if song_link:
        webbrowser.open(song_link)  # Open the song in the default browser
        return f"Recommended Song: {song_link}"
    else:
        return "No song recommendation available."

# Example usage:
if __name__ == "__main__":
    emotion = "happy"  # Change this for testing
    print(recommend_song(emotion))
