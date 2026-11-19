
import heapq

def prim(graph, start):
    visited = set()
    min_heap = []

    visited.add(start)

    # Insert starting vertex edges into min-heap
    for neighbor, weight in graph[start]:
        heapq.heappush(min_heap, (weight, start, neighbor))

    mst = []
    total_cost = 0

    while min_heap:
        weight, u, v = heapq.heappop(min_heap)

        # Skip if vertex is already visited
        if v not in visited:
            visited.add(v)

            mst.append((u, v, weight))
            total_cost += weight

            # Add edges of newly visited vertex
            for neighbor, w in graph[v]:
                if neighbor not in visited:
                    heapq.heappush(min_heap, (w, v, neighbor))

    return mst, total_cost


# Take graph input from user
n = int(input("Enter number of vertices: "))

vertices = input("Enter vertex names separated by space: ").split()

graph = {v: [] for v in vertices}

e = int(input("Enter number of edges: "))

for i in range(e):
    u, v, w = input("Enter edge (u v weight): ").split()
    w = int(w)

    graph[u].append((v, w))
    graph[v].append((u, w))

start = input("Enter starting vertex: ")

mst, cost = prim(graph, start)

print("\nEdges in Minimum Spanning Tree:")

for u, v, weight in mst:
    print(u, "-", v, ":", weight)

print("\nTotal Minimum Cost =", cost)
