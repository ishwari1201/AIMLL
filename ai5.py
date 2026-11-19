
# Disjoint Set (Union-Find) class
class DisjointSet:

    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}

    # Find the parent of a vertex
    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])

        return self.parent[item]

    # Join two sets
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            self.parent[rootY] = rootX


# Kruskal's algorithm
def kruskal(vertices, edges):

    # Sort edges by weight
    edges = sorted(edges, key=lambda x: x[2])

    ds = DisjointSet(vertices)

    mst = []
    total_cost = 0

    for u, v, weight in edges:

        # Check whether vertices belong to different sets
        if ds.find(u) != ds.find(v):

            ds.union(u, v)

            mst.append((u, v, weight))
            total_cost += weight

        if len(mst) == len(vertices) - 1:
            break

    return mst, total_cost


# Take graph input from user
n = int(input("Enter number of vertices: "))

vertices = input("Enter vertex names separated by space: ").split()

e = int(input("Enter number of edges: "))

edges = []

for i in range(e):
    u, v, w = input("Enter edge (u v weight): ").split()
    edges.append((u, v, int(w)))

# Apply Kruskal's algorithm
mst, cost = kruskal(vertices, edges)

# Display MST
print("\nEdges in Minimum Spanning Tree:")

for u, v, weight in mst:
    print(u, "-", v, ":", weight)

print("\nTotal Minimum Cost =", cost)

if len(mst) != len(vertices) - 1:
    print("Warning: Graph is disconnected. MST does not exist.")
