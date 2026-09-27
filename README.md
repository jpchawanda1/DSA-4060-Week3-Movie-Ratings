# DSA 4060 - Week 2 Practical Task
## Recommendation Data and Popularity Baselines

Cleaning, exploration and a popularity-based recommender built on a dataset of users'
movie ratings.

---

## The task

Given a raw `ratings.csv`-style dataset, the brief asks for seven things:

1. **Load and inspect** the dataset - first 10 records, dimensions, data types, missing
   values and duplicates.
2. **Clean** it - remove duplicates, remove records with missing user or movie IDs, and
   confirm ratings fall within the expected scale.
3. **Explore** the interaction data - total ratings, unique users, unique movies, most
   active user, most-rated movie, average rating.
4. **Build a user-item interaction matrix** - users as rows, movies as columns, ratings
   in the cells.
5. **Calculate sparsity** and explain what the result means.
6. **Build a popularity-based recommender** - average rating and rating count per film,
   then the top 10 films with at least 10 ratings.
7. **Interpret the results** - why those films rank top, one advantage and two
   limitations of the approach.

---

## Repository contents

| File | What it is |
|---|---|
| `DSA_4060_Week2_Movie_Ratings.ipynb` | The completed notebook - all seven tasks, with explanations in Markdown cells |
| `DSA_4060_Week2_Movie_Ratings.csv` | The dataset supplied by the lecturer |
| `top_10_recommended_movies.csv` | The ten recommended films, produced by Task 6 |
| `README.md` | This file |

The dataset has five columns: `user_id`, `movie_id`, `movie_title`, `rating` (1-5 stars)
and `timestamp`.

---

## Running the notebook

```bash
pip install pandas numpy matplotlib jupyter
jupyter notebook DSA_4060_Week2_Movie_Ratings.ipynb
```

Run all cells from the top. The notebook reads the CSV from the repository root and
writes `top_10_recommended_movies.csv` back to it.

---

## Findings

### Cleaning

The raw file held **509 rows**, of which **9 were unusable**:

| Problem | Rows removed | Why |
|---|---|---|
| Exact duplicate records | 3 | The same rating recorded twice inflates rating counts, which drive the popularity ranking |
| Missing `user_id` or `movie_id` | 3 | Cannot be placed in the user-item matrix - we do not know which cell they belong to |
| Ratings outside the 1-5 scale (`-1`, `0`, `6`) | 3 | Impossible on this scale, so the value is untrustworthy |

That leaves **500 clean ratings**.

One thing deliberately *not* removed: some users rated the same film more than once on
different dates - 70 such repeat ratings. These are genuine repeat opinions rather than
duplicate records, so they are kept in the dataset and averaged per user when the
user-item matrix is built.

### Exploratory analysis

| Measure | Value |
|---|---|
| Total ratings (after cleaning) | 500 |
| Unique users | 50 |
| Unique movies | 24 |
| Distinct user-movie pairs | 430 |
| Most active user | **U010** - 17 ratings |
| Most-rated movie | **Nairobi Nights** and **Mountain Echoes** - tied on 37 ratings each |
| Average rating | **3.45** out of 5 |

Ratings cluster on 3 and 4 stars with very few 1s, so the mean sits above the scale
midpoint - the mild positive skew typical of ratings data, since people mostly watch
films they already expect to like.

### User-item matrix and sparsity

The matrix is **50 users x 24 movies = 1,200 possible cells**, of which **430 are
filled**.

```
Sparsity = 1 - 430/1200 = 0.6417  ->  64.17% empty
```

**What this means:** roughly one cell in three holds a rating, so the average user has
rated about 9 of the 24 films and has no recorded opinion on the other 15. Predicting
what belongs in those empty cells is exactly the recommender's job.

This is unusually *dense* for recommendation data. Real systems typically sit at
95-99.9% sparsity - the MovieLens 100K benchmark is about 93.7% empty - because their
catalogues hold thousands of items any one person will never touch. Sparsity grows with
the catalogue: the denominator rises with every film added while a user's viewing habits
do not. That is what makes recommendation hard, and it is why collaborative filtering,
which depends on users having rated films in common, gets weaker as a service grows.

### Popularity-based recommender

Two numbers per film: the **average rating** (how much viewers liked it) and the
**number of ratings** (how much evidence stands behind that average). The minimum of 10
ratings is the guard on the average - a 5.0 from one viewer is not a safer bet than a
4.3 from thirty. Eligible films are then ranked by average rating, with ties broken on
rating count.

**Top 10 recommended movies** (also in `top_10_recommended_movies.csv`):

| Rank | Movie | Average rating | Number of ratings |
|---:|---|---:|---:|
| 1 | Market Day | 4.47 | 15 |
| 2 | Midnight in Mombasa | 4.35 | 17 |
| 3 | Code Breakers | 4.25 | 16 |
| 4 | Home Again | 4.13 | 15 |
| 5 | Future Africa | 4.13 | 15 |
| 6 | Hidden Lake | 4.12 | 16 |
| 7 | City of Tomorrow | 3.91 | 34 |
| 8 | Mountain Echoes | 3.86 | 37 |
| 9 | The Final Goal | 3.76 | 17 |
| 10 | Coastal Dreams | 3.67 | 27 |

---

## Interpretation

### Why these movies appear at the top

The `>= 10 ratings` filter runs first, then the survivors are sorted by average rating -
so the top films are those whose viewers scored them highest on average. In this
particular dataset every one of the 24 films clears the 10-rating bar (the least-rated
has 13), so the filter changes nothing here; on a real catalogue, with its long tail of
films holding one or two ratings each, it would be the most consequential part of the
rule.

Two things are worth noticing:

- **The spread is narrow.** First and tenth place are separated by about 0.8 of a star.
  The ranking is real, but neighbouring films sit close enough that a handful of extra
  ratings could reorder the middle of the list.
- **Popularity here means "well liked", not "most watched".** *Nairobi Nights* received
  the most ratings of any film (37) and still misses the top 10 - plenty of people rated
  it, and they were lukewarm. Ranking by rating count instead of average would produce a
  visibly different list.

### Advantage

**It works immediately for a brand-new user, and it costs almost nothing to run.** The
recommender needs no history, no profile and no similar users, which solves the *user
cold-start* problem - what to show someone who has just signed up. It is a single
`groupby` over the ratings table, cheap to compute and easy to explain to a
non-technical stakeholder. That is why it is the standard baseline any fancier model has
to beat.

### Limitations

**1. It is identical for every user - there is no personalisation.** Everyone sees the
same ten films, so the recommender cannot tell a documentary enthusiast from a comedy
fan. The 64% of the matrix measured as empty is precisely the individual preference
information this method discards. It also has no idea which of these films a user has
already seen, so part of every list is wasted.

**2. It creates a popularity feedback loop that buries the catalogue.** Films that are
already widely rated get recommended, which earns them more views and more ratings,
which keeps them at the top. Films with few ratings never surface, so they never
accumulate the ratings that would let them qualify - the `>= 10` threshold, necessary as
it is, locks them out. New releases suffer the same fate, the *item cold-start* problem.
The long tail becomes effectively invisible and recommendations grow less diverse over
time.

### Where this leads next

The popularity model is the floor, not the ceiling. The natural next step is
**collaborative filtering** over the user-item matrix built in Task 4: find users with
similar rating patterns and recommend what those neighbours liked. That fixes the
personalisation gap directly - at the cost of depending on rating overlap between users,
which is exactly what sparsity erodes.
