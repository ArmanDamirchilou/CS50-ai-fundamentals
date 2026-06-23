"""
K-Means Clustering
------------------
Lecture 2 showed this diagram of colored dots moving toward
their cluster centers. This is that, step by step.

Points get assigned to the nearest centroid, centroids move
to the middle of their cluster, repeat until nothing changes.
That's it. That's K-Means.
"""

import random
import math


def distance(a, b):
    # standard euclidean distance between two points
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def assign_clusters(points, centroids):
    # for each point, find which centroid it's closest to
    clusters = [[] for _ in centroids]
    for point in points:
        nearest = min(range(len(centroids)),
                      key=lambda i: distance(point, centroids[i]))
        clusters[nearest].append(point)
    return clusters


def move_centroids(clusters):
    # move each centroid to the mean position of its cluster
    # if a cluster is empty (rare but possible), keep it where it is
    new_centroids = []
    for cluster in clusters:
        if not cluster:
            # TODO: handle empty clusters more gracefully
            # for now just skip — hasn't been a problem in practice
            continue
        dims = len(cluster[0])
        mean = [sum(p[d] for p in cluster) / len(cluster) for d in range(dims)]
        new_centroids.append(mean)
    return new_centroids


def converged(old, new, threshold=0.001):
    # stop when centroids barely move anymore
    return all(distance(o, n) < threshold for o, n in zip(old, new))


def kmeans(points, k, max_iter=100):
    # pick k random starting centroids from the actual data points
    # (smarter than random positions — avoids empty clusters)
    centroids = random.sample(points, k)

    print(f"  Starting K-Means with k={k}, {len(points)} points")
    print()

    for iteration in range(max_iter):
        clusters = assign_clusters(points, centroids)
        new_centroids = move_centroids(clusters)

        sizes = [len(c) for c in clusters]
        print(f"  iter {iteration+1:3d} | cluster sizes: {sizes}")

        if converged(centroids, new_centroids):
            print(f"\n  Converged after {iteration+1} iterations.")
            break

        centroids = new_centroids

    return clusters, centroids


def make_blobs(n_per_cluster=30, centers=None, noise=0.8):
    # generate 2D point clouds around fixed centers
    # this mimics the colored dot diagrams from lecture
    if centers is None:
        centers = [(2, 2), (7, 3), (5, 8)]

    points = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, noise)
            y = cy + random.gauss(0, noise)
            points.append((x, y))
            true_labels.append(label)

    combined = list(zip(points, true_labels))
    random.shuffle(combined)
    return [p for p, _ in combined], [l for _, l in combined]


def render_ascii(clusters, centroids, width=50, height=20):
    """
    draw the clusters in the terminal.
    not pretty, but you can actually see the clusters form.
    """
    symbols = ["●", "◆", "▲", "■", "★"]
    centroid_sym = "+"

    # find bounds
    all_points = [p for c in clusters for p in c]
    if not all_points:
        return

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
    y_min, y_max = min(ys) - 0.5, max(ys) + 0.5

    def to_grid(x, y):
        col = int((x - x_min) / (x_max - x_min) * (width - 1))
        row = int((y - y_min) / (y_max - y_min) * (height - 1))
        return row, col

    grid = [[" " for _ in range(width)] for _ in range(height)]

    for i, cluster in enumerate(clusters):
        sym = symbols[i % len(symbols)]
        for point in cluster:
            r, c = to_grid(*point)
            if 0 <= r < height and 0 <= c < width:
                grid[r][c] = sym

    for i, centroid in enumerate(centroids):
        r, c = to_grid(*centroid)
        if 0 <= r < height and 0 <= c < width:
            grid[r][c] = centroid_sym

    print()
    border = "─" * (width + 2)
    print(f"  ┌{border}┐")
    for row in reversed(grid):  # flip so y increases upward
        print(f"  │ {''.join(row)} │")
    print(f"  └{border}┘")
    print(f"  Legend: {' '.join(symbols[:len(clusters)])} = clusters,  + = centroid")
    print()


if __name__ == "__main__":
    print("=" * 45)
    print("  K-Means Clustering — Lecture 2 | CS50x AI")
    print("=" * 45)
    print()

    points, true_labels = make_blobs(n_per_cluster=40)

    k = 3
    clusters, centroids = kmeans(points, k)

    print()
    print("  Final clusters:")
    render_ascii(clusters, centroids)

    # show some stats
    print("  Cluster summary:")
    for i, (cluster, centroid) in enumerate(zip(clusters, centroids)):
        avg_dist = sum(distance(p, centroid) for p in cluster) / len(cluster)
        print(f"    Cluster {i+1}: {len(cluster)} points | "
              f"centroid ({centroid[0]:.2f}, {centroid[1]:.2f}) | "
              f"avg dist: {avg_dist:.2f}")

