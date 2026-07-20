import csv
import re
import time
import os
import json
import requests
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Define folder structures to keep the workspace clean
INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

# Create the directories if they don't already exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map file paths to their respective folders
DIARY_CSV = os.path.join(INPUT_DIR, "diary.csv")
REVIEWS_CSV = os.path.join(INPUT_DIR, "reviews.csv")
RATINGS_CSV = os.path.join(INPUT_DIR, "ratings.csv")
WATCHLIST_CSV = os.path.join(INPUT_DIR, "watchlist.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ai_movie_profile.csv")

# Cache file to store API responses and prevent redundant network calls
CACHE_FILE = os.path.join(OUTPUT_DIR, "tmdb_cache.json")

STOP_WORDS = {"the", "and", "a", "to", "of", "in", "i", "is", "that", "it", "on", "you", "this", "for", "but", "with",
              "are", "have", "be", "at", "or", "as", "was", "so", "if", "out", "not", "my", "film", "movie"}


def load_cache():
    # Load existing cache or initialize a fresh structure if it's the first run
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Cache file is corrupted. Starting fresh.")
    return {"genre_map": {}, "movies": {}}


def save_cache(cache_data):
    # Save the dictionary back to disk as JSON
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=4)


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return polarity, "Positive"
    elif polarity < -0.1:
        return polarity, "Negative"
    else:
        return polarity, "Neutral"


def extract_keywords(text):
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    keywords = [word for word in words if word not in STOP_WORDS and len(word) > 3]
    word_counts = {}
    for w in keywords:
        word_counts[w] = word_counts.get(w, 0) + 1
    sorted_keywords = sorted(word_counts, key=word_counts.get, reverse=True)[:5]
    return ", ".join(sorted_keywords)


def process_text_data(review_text):
    if review_text.strip():
        _, sentiment_label = analyze_sentiment(review_text)
        keywords = extract_keywords(review_text)
        return sentiment_label, keywords
    return "No Review", ""


def get_tmdb_genre_map(cache_data):
    if not TMDB_API_KEY:
        return {}

    # Check if we already fetched and saved the genre map previously
    if cache_data.get("genre_map"):
        return cache_data["genre_map"]

    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        genres = response.json().get("genres", [])

        # We need to convert the keys to strings because JSON dictionary keys must be strings
        genre_map = {str(genre["id"]): genre["name"] for genre in genres}
        cache_data["genre_map"] = genre_map
        return genre_map
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch genre map from TMDB: {e}")
        return {}


def fetch_movie_genres(title, year, unique_key, genre_map, cache_data):
    # Skip the network request entirely if we've looked this movie up before
    if unique_key in cache_data["movies"]:
        return cache_data["movies"][unique_key]

    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}&year={year}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            genre_ids = results[0].get("genre_ids", [])
            # Map the IDs to names, ensuring we check against string keys
            genre_names = [genre_map[str(gid)] for gid in genre_ids if str(gid) in genre_map]
            final_genres = ", ".join(genre_names)

            # Save the result to our in-memory cache
            cache_data["movies"][unique_key] = final_genres
            return final_genres

    except requests.exceptions.RequestException:
        pass

    # If the lookup fails or finds nothing, cache it as Unknown so we don't keep retrying it
    cache_data["movies"][unique_key] = "Unknown"
    return "Unknown"


# Initialize our state
movie_database = {}
MAX_ITERATIONS = 50000
api_cache = load_cache()

print("Aggregating and deduplicating data from inputs/ folder...")

# 1. Process diary
try:
    with open(DIARY_CSV, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        loop_count = 0
        for row in reader:
            if loop_count >= MAX_ITERATIONS: break
            title = row.get("Name", "Unknown")
            year = row.get("Year", "")
            unique_key = f"{title}_{year}"

            sentiment_label, keywords = process_text_data(row.get("Review", ""))
            movie_database[unique_key] = {
                "Title": title, "Year": year, "Status": "Watched",
                "Rating": row.get("Rating", ""), "Sentiment": sentiment_label, "Keywords": keywords
            }
            loop_count += 1
except FileNotFoundError:
    print(f"Warning: {DIARY_CSV} not found.")

# 2. Process reviews
try:
    with open(REVIEWS_CSV, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        loop_count = 0
        for row in reader:
            if loop_count >= MAX_ITERATIONS: break
            title = row.get("Name", "Unknown")
            year = row.get("Year", "")
            unique_key = f"{title}_{year}"
            review_text = row.get("Review", "")

            if review_text.strip():
                sentiment_label, keywords = process_text_data(review_text)
                if unique_key in movie_database:
                    movie_database[unique_key]["Sentiment"] = sentiment_label
                    movie_database[unique_key]["Keywords"] = keywords
                else:
                    movie_database[unique_key] = {
                        "Title": title, "Year": year, "Status": "Watched",
                        "Rating": row.get("Rating", ""), "Sentiment": sentiment_label, "Keywords": keywords
                    }
            loop_count += 1
except FileNotFoundError:
    print(f"Warning: {REVIEWS_CSV} not found.")

# 3. Process standalone ratings
try:
    with open(RATINGS_CSV, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        loop_count = 0
        for row in reader:
            if loop_count >= MAX_ITERATIONS: break
            title = row.get("Name", "Unknown")
            year = row.get("Year", "")
            unique_key = f"{title}_{year}"

            if unique_key not in movie_database:
                movie_database[unique_key] = {
                    "Title": title, "Year": year, "Status": "Watched",
                    "Rating": row.get("Rating", ""), "Sentiment": "No Review", "Keywords": ""
                }
            loop_count += 1
except FileNotFoundError:
    print(f"Warning: {RATINGS_CSV} not found.")

# 4. Process watchlist
try:
    with open(WATCHLIST_CSV, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        loop_count = 0
        for row in reader:
            if loop_count >= MAX_ITERATIONS: break
            title = row.get("Name", "Unknown")
            year = row.get("Year", "")
            unique_key = f"{title}_{year}"

            if unique_key not in movie_database:
                movie_database[unique_key] = {
                    "Title": title, "Year": year, "Status": "Watchlist",
                    "Rating": "", "Sentiment": "", "Keywords": ""
                }
            loop_count += 1
except FileNotFoundError:
    print(f"Warning: {WATCHLIST_CSV} not found.")

# Fetch TMDB genre map (will load from cache if available)
genre_map = get_tmdb_genre_map(api_cache)

if movie_database:
    print(f"Writing dataset to {OUTPUT_CSV}...")

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as outfile:
        fieldnames = ["Title", "Year", "Status", "Rating", "Sentiment", "Keywords", "Genres"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        write_count = 0

        for unique_key, row_data in movie_database.items():
            if write_count >= (MAX_ITERATIONS * 4): break

            if TMDB_API_KEY:
                # Pass the unique_key and cache_data into the fetcher
                genres = fetch_movie_genres(row_data["Title"], row_data["Year"], unique_key, genre_map, api_cache)
                row_data["Genres"] = genres

                # Only sleep if we actually made a network call (i.e. if it wasn't in cache initially)
                # A quick hack to check if we just added it to the cache is beyond this scope, but
                # sleeping 0.05 seconds even on cache hits won't hurt execution time significantly.
                time.sleep(0.01)
            else:
                row_data["Genres"] = "No API Key"

            writer.writerow(row_data)
            write_count += 1

            if write_count % 100 == 0:
                print(f"Processed {write_count} movies...")

    # Save the cache file to disk once all lookups are complete
    save_cache(api_cache)
    print(f"Export complete. Deduplicated {write_count} total unique movies.")
    print("API cache updated.")
else:
    print("No data found. Ensure the CSV files are inside the 'inputs' folder.")