import csv
import re
import time
import os
import json
import requests
from collections import defaultdict
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
MAX_ITERATIONS = 50000


# ==========================================
# PART 1: DATA AGGREGATION & TMDB FETCHING
# ==========================================

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Cache file is corrupted. Starting fresh.")
    return {"genre_map": {}, "movies": {}}


def save_cache(cache_data):
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

    if cache_data.get("genre_map"):
        return cache_data["genre_map"]

    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        genres = response.json().get("genres", [])
        genre_map = {str(genre["id"]): genre["name"] for genre in genres}
        cache_data["genre_map"] = genre_map
        return genre_map
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch genre map from TMDB: {e}")
        return {}


def fetch_movie_genres(title, year, unique_key, genre_map, cache_data):
    if unique_key in cache_data["movies"]:
        return cache_data["movies"][unique_key]

    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}&year={year}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            genre_ids = results[0].get("genre_ids", [])
            genre_names = [genre_map[str(gid)] for gid in genre_ids if str(gid) in genre_map]
            final_genres = ", ".join(genre_names)
            cache_data["movies"][unique_key] = final_genres
            return final_genres

    except requests.exceptions.RequestException:
        pass

    cache_data["movies"][unique_key] = "Unknown"
    return "Unknown"


def run_data_aggregation():
    movie_database = {}
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
                    genres = fetch_movie_genres(row_data["Title"], row_data["Year"], unique_key, genre_map, api_cache)
                    row_data["Genres"] = genres
                    time.sleep(0.01)
                else:
                    row_data["Genres"] = "No API Key"

                writer.writerow(row_data)
                write_count += 1

                if write_count % 100 == 0:
                    print(f"Processed {write_count} movies...")

        save_cache(api_cache)
        print(f"\n[Success] Export complete. Deduplicated {write_count} total unique movies.")
        print("API cache updated.")
    else:
        print("\n[Error] No data found. Ensure the CSV files are inside the 'inputs' folder.")


# ==========================================
# PART 2: ANALYSIS
# ==========================================

def analyze_genre_preferences(filepath):
    genre_totals = defaultdict(float)
    genre_counts = defaultdict(int)

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row_count = 0
        for row in reader:
            if row_count >= MAX_ITERATIONS: break
            row_count += 1

            if row.get("Status") == "Watched" and row.get("Rating"):
                try:
                    rating = float(row["Rating"])
                except ValueError:
                    continue

                genres = [g.strip() for g in row.get("Genres", "").split(",") if g.strip()]

                genre_iterations = 0
                for genre in genres:
                    if genre_iterations >= 20: break
                    genre_iterations += 1

                    genre_totals[genre] += rating
                    genre_counts[genre] += 1

    genre_averages = {}
    dict_iterations = 0
    for genre, total in genre_totals.items():
        if dict_iterations >= 1000: break
        dict_iterations += 1
        if genre_counts[genre] >= 3:
            genre_averages[genre] = round(total / genre_counts[genre], 2)

    return dict(sorted(genre_averages.items(), key=lambda item: item[1], reverse=True))


def analyze_decade_preferences(filepath):
    decade_totals = defaultdict(float)
    decade_counts = defaultdict(int)

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row_count = 0
        for row in reader:
            if row_count >= MAX_ITERATIONS: break
            row_count += 1

            if row.get("Status") == "Watched" and row.get("Rating") and row.get("Year"):
                try:
                    rating = float(row["Rating"])
                    year = int(row["Year"])
                except ValueError:
                    continue

                decade = (year // 10) * 10
                decade_totals[decade] += rating
                decade_counts[decade] += 1

    decade_averages = {}
    dict_iterations = 0
    for dec, total in decade_totals.items():
        if dict_iterations >= 100: break
        dict_iterations += 1
        if decade_counts[dec] >= 3:
            decade_averages[f"{dec}s"] = round(total / decade_counts[dec], 2)

    return dict(sorted(decade_averages.items(), key=lambda item: item[1], reverse=True))


def run_analysis():
    if not os.path.exists(OUTPUT_CSV):
        print(f"\n[Error] Could not find {OUTPUT_CSV}.")
        print("Please run Option 1 (Build AI Profile) first so there is data to analyze.")
        return

    print(f"\nReading data from {OUTPUT_CSV}...")
    genre_stats = analyze_genre_preferences(OUTPUT_CSV)
    decade_stats = analyze_decade_preferences(OUTPUT_CSV)

    print("\n==============================")
    print("   YOUR MOVIE PREFERENCES     ")
    print("==============================\n")

    print("Top Genres by Average Rating (Min. 3 movies watched):")
    genre_print_count = 0
    for genre, avg in genre_stats.items():
        if genre_print_count >= 5: break
        if genre != "Unknown":  # Filter out failed API lookups from the rankings
            print(f"  - {genre}: {avg}/5.0")
            genre_print_count += 1

    print("\nAverage Rating by Release Decade (Min. 3 movies watched):")
    decade_print_count = 0
    for decade, avg in decade_stats.items():
        if decade_print_count >= 10: break
        print(f"  - {decade}: {avg}/5.0")
        decade_print_count += 1

    print("\n==============================\n")


# ==========================================
# CLI INTERFACE
# ==========================================

def main():
    while True:
        print("\n--- Letterboxd AI Processing Tool ---")
        print("1. Build AI Profile (Aggregate CSVs & Fetch TMDB Data)")
        print("2. Run Local Analysis (Generate Stats Report)")
        print("3. Exit")

        choice = input("\nSelect an option (1-3): ").strip()

        if choice == '1':
            run_data_aggregation()
        elif choice == '2':
            run_analysis()
        elif choice == '3':
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()