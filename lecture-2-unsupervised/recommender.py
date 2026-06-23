"""
Simple Recommender System
--------------------------
Lecture 2 covered two approaches:
- Content-based filtering: recommend by genre, year, description
- Collaborative filtering: recommend by what similar people liked

This implements a tiny version of both using a fake movie dataset.
No external libraries — just dictionaries and math.
"""

import math


# small fake dataset — each movie has features and user ratings
# in a real system this would be a database with millions of rows
MOVIES = {
    "Inception":       {"genre": "sci-fi",  "year": 2010, "rating": 8.8},
    "Interstellar":    {"genre": "sci-fi",  "year": 2014, "rating": 8.6},
    "The Matrix":      {"genre": "sci-fi",  "year": 1999, "rating": 8.7},
    "Parasite":        {"genre": "thriller","year": 2019, "rating": 8.5},
    "Get Out":         {"genre": "thriller","year": 2017, "rating": 7.7},
    "Hereditary":      {"genre": "horror",  "year": 2018, "rating": 7.3},
    "The Shining":     {"genre": "horror",  "year": 1980, "rating": 8.4},
    "Pulp Fiction":    {"genre": "crime",   "year": 1994, "rating": 8.9},
    "No Country":      {"genre": "crime",   "year": 2007, "rating": 8.1},
}

# what different users have rated (scale 1-10)
USER_RATINGS = {
    "alice": {"Inception": 9, "Interstellar": 8, "The Matrix": 9, "Parasite": 6},
    "bob":   {"Parasite": 9, "Get Out": 8, "Hereditary": 7, "No Country": 8},
    "carol": {"Inception": 7, "The Shining": 9, "Hereditary": 8, "Get Out": 7},
    "dave":  {"Pulp Fiction": 10, "No Country": 9, "Parasite": 8, "Get Out": 7},
}


# ── Content-Based Filtering ──────────────────────────────────────────────────

def content_based(liked_movie, n=3):
    """
    recommend movies similar to one you already liked.
    similarity = same genre gets a big boost, similar year helps a bit.
    
    from lecture: this is the "knowing genre, year, description" approach.
    """
    liked = MOVIES.get(liked_movie)
    if not liked:
        print(f"  don't know '{liked_movie}'")
        return []

    scores = {}
    for title, info in MOVIES.items():
        if title == liked_movie:
            continue

        score = 0

        # same genre = strong signal
        if info["genre"] == liked["genre"]:
            score += 10

        # similar release year = weak signal
        year_diff = abs(info["year"] - liked["year"])
        score += max(0, 5 - year_diff * 0.1)

        # slightly prefer higher-rated movies (but don't let it dominate)
        score += info["rating"] * 0.3

        scores[title] = score

    ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
    return ranked[:n]


# ── Collaborative Filtering ──────────────────────────────────────────────────

def cosine_similarity(user_a, user_b):
    """
    measure how similar two users' tastes are.
    
    cosine similarity: 1 = identical taste, 0 = no overlap, -1 = opposite
    we use this to find users "like you" — the core of collaborative filtering.
    """
    # only compare movies both users have rated
    common = set(user_a) & set(user_b)
    if not common:
        return 0.0

    dot = sum(user_a[m] * user_b[m] for m in common)
    mag_a = math.sqrt(sum(user_a[m] ** 2 for m in common))
    mag_b = math.sqrt(sum(user_b[m] ** 2 for m in common))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def collaborative(user, n=3):
    """
    recommend based on what people with similar taste liked.
    
    step 1: find users most similar to this user
    step 2: look at what they liked that this user hasn't seen
    step 3: rank those movies by how much similar users liked them
    """
    if user not in USER_RATINGS:
        print(f"  no data for user '{user}'")
        return []

    my_ratings = USER_RATINGS[user]

    # find similarity to every other user
    similarities = {}
    for other, their_ratings in USER_RATINGS.items():
        if other == user:
            continue
        similarities[other] = cosine_similarity(my_ratings, their_ratings)

    # weight each unseen movie by how similar the person who liked it is
    scores = {}
    for other, sim in similarities.items():
        if sim <= 0:
            continue
        for movie, rating in USER_RATINGS[other].items():
            if movie in my_ratings:
                continue  # already seen
            if movie not in scores:
                scores[movie] = 0
            scores[movie] += sim * rating

    ranked = sorted(scores, key=lambda m: scores[m], reverse=True)
    return ranked[:n]


if __name__ == "__main__":
    print("=" * 45)
    print("  Recommender System — Lecture 2 | CS50x AI")
    print("=" * 45)
    print()

    # content-based demo
    seed = "Inception"
    print(f"  Content-based: you liked '{seed}'")
    recs = content_based(seed)
    for i, movie in enumerate(recs, 1):
        info = MOVIES[movie]
        print(f"    {i}. {movie} ({info['year']}, {info['genre']})")

    print()

    # collaborative filtering demo
    user = "alice"
    print(f"  Collaborative: recommending for '{user}'")
    print(f"  Alice has rated: {list(USER_RATINGS[user].keys())}")
    recs = collaborative(user)
    for i, movie in enumerate(recs, 1):
        info = MOVIES[movie]
        print(f"    {i}. {movie} ({info['genre']})")

    print()

    # show similarity matrix — interesting to look at
    print("  User similarity matrix:")
    users = list(USER_RATINGS.keys())
    header = "         " + "".join(f"{u:>10}" for u in users)
    print(f"  {header}")
    for u1 in users:
        row = f"  {u1:<8} "
        for u2 in users:
            if u1 == u2:
                row += f"{'1.00':>10}"
            else:
                sim = cosine_similarity(USER_RATINGS[u1], USER_RATINGS[u2])
                row += f"{sim:>10.2f}"
        print(row)

