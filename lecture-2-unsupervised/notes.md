# Lecture 2 — Unsupervised Learning

> My notes from CS50x AI Week — unsupervised learning section.
> The big idea: no labels. The model finds patterns on its own.

---

## Supervised vs. Unsupervised

| | Supervised | Unsupervised |
|---|---|---|
| Data | Labeled (cat / dog) | No labels |
| Goal | Learn from correct answers | Find hidden patterns |
| Examples | Classification, Regression | Clustering, Dimensionality Reduction |

---

## Clustering

Grouping data points together based on similarity.
The model decides the groups — we don't tell it what to look for.

---

### K-Means Clustering

The most popular clustering algorithm.

**How it works:**

1. Pick K random points as starting centroids
2. Assign every data point to the nearest centroid
3. Move each centroid to the center (mean) of its cluster
4. Repeat steps 2-3 until centroids stop moving

**Convergence** = when centroids barely move anymore. We've found stable clusters.

```
Before:                After:
  · · · × × ×           ● ● ● ◆ ◆ ◆
  · · ×   × ×    →      ● ● ◆   ◆ ◆
  · ·   ×            +  ● ●   ◆
                      (centroids marked with +)
```

*Key vocab: centroid = center of a cluster, variety = گون*

---

### DBSCAN

Different approach — doesn't need K upfront. Finds clusters based on density.

**Two parameters:**
- **EPS** = maximum distance to look for nearby points
- **MinPts** = minimum number of nearby points to form a dense region

**Steps:**

1. For each point: count how many points are within EPS distance
2. If count ≥ MinPts → it becomes a **Core Point**
3. Connect this core point to its nearby points
4. If those nearby points are also core points → expand the cluster further
5. Points near a core point but without enough neighbors → **Border Points**
6. Points not part of any dense region → **Noise**

**Noise points = Anomaly Detection**
Points that don't belong to any cluster are outliers — useful for fraud detection, network intrusion detection, etc.

---

## Dimensionality Reduction

Reducing the number of input features while keeping the important information.

```
Before: (n₁, n₂, n₃, n₄, n₅, n₆, n₇)  ← 7 features
         ↓ Dimensionality Reduction
After:  (n₁, n₂, n₃)                    ← 3 features
```

Why? Fewer features = faster training, less noise, easier to visualize.

---

## Association Rules (Lecture 2 — Digital Marketing / E-Commerce)

Finding patterns in what items appear together.

### Support
Measure of how frequently a set of items appears in a dataset.

### Confidence
Measure of the strength of association between two sets of items.
*"If you buy A, how likely are you to also buy B?"*

### Apriori Algorithm
Uses support and confidence to find useful association rules efficiently.

---

## Recommender Systems

Making suggestions based on data.

### Content-Based Filtering
Recommend based on properties of the item itself:
- Genre, year, description, actors
- "You liked sci-fi from 2010 → here's more sci-fi"

**One-Hot Encoding** → used to convert categories into numbers

### Collaborative Filtering
Recommend based on what similar people liked:
- Find users with similar taste
- Recommend what they liked that you haven't seen
- "People like you also enjoyed..."

### A/B Testing
1. Suggest option A to the user
2. If they like it → suggest more like A
3. If not → suggest option B
4. Keep testing until you find what works

---

## Key Takeaways

1. **Unsupervised learning finds structure without labels.** The model groups, reduces, or finds patterns in data with no human guidance on what's "correct."

2. **K-Means and DBSCAN solve the same problem differently.** K-Means needs you to pick K upfront; DBSCAN finds clusters automatically but needs EPS and MinPts tuned carefully.

3. **Noise isn't always bad.** In DBSCAN, noise points = outliers = anomalies. Anomaly detection is one of the most useful applications in fraud detection and security.

---

## My Questions

1. How do you choose the right K in K-Means? Is there a mathematical way to find the optimal number of clusters?

2. DBSCAN handles arbitrary shapes well, but K-Means assumes spherical clusters. When would each be better?

3. In collaborative filtering, what happens with new users who have no rating history? (This is called the "cold start problem")

---

## Projects Built from This Lecture

- `kmeans.py` — K-Means from scratch with ASCII visualization showing clusters forming
- `recommender.py` — Content-based + collaborative filtering on a movie dataset
