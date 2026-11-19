
import heapq

def dijkstra(graph, start):
    distances = {v: float('inf') for v in graph}
    paths = {v: [] for v in graph}

    distances[start] = 0
    paths[start] = [start]

    pq = [(0, start)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_vertex] + [neighbor]

                heapq.heappush(pq, (distance, neighbor))

    return distances, paths


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

start = input("Enter source vertex: ")

distances, paths = dijkstra(graph, start)

print("\nShortest Distances and Paths from", start)

for vertex in vertices:
    print("Vertex:", vertex)
    print("Distance:", distances[vertex])

    if paths[vertex]:
        print("Path:", " -> ".join(paths[vertex]))
    else:
        print("Path: No path available")

    print()
