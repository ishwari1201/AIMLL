
import heapq

# Dijkstra's algorithm
def dijkstra(graph, start):

    distances = {v: float('inf') for v in graph}
    paths = {v: [] for v in graph}

    distances[start] = 0
    paths[start] = [start]

    pq = [(0, start)]

    while pq:

        current_distance, current_node = heapq.heappop(pq)

        if current_distance > distances[current_node]:
            continue

        for neighbor, latency in graph[current_node]:

            new_distance = current_distance + latency

            if new_distance < distances[neighbor]:

                distances[neighbor] = new_distance
                paths[neighbor] = paths[current_node] + [neighbor]

                heapq.heappush(pq, (new_distance, neighbor))

    return distances, paths


# Take network input from user
n = int(input("Enter number of network nodes: "))

nodes = input("Enter node names separated by space: ").split()

graph = {v: [] for v in nodes}

e = int(input("Enter number of network connections: "))

for i in range(e):

    u, v, latency = input(
        "Enter connection (source destination latency): "
    ).split()

    latency = int(latency)

    graph[u].append((v, latency))
    graph[v].append((u, latency))

# Select source node
start = input("Enter source node: ")

# Calculate shortest paths
distances, paths = dijkstra(graph, start)

# Display results
print("\nMINIMUM LATENCY ROUTES FROM", start)

for node in nodes:

    print("\nDestination:", node)
    print("Minimum Latency:", distances[node])

    if paths[node]:
        print("Shortest Path:", " -> ".join(paths[node]))
    else:
        print("Shortest Path: No route available")

print("\nDijkstra Algorithm Completed Successfully!")
