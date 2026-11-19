
import heapq

# 1. Take input
n = int(input("Enter number of vertices: "))

nodes = input("Enter vertex names: ").split()

graph = {node: [] for node in nodes}

m = int(input("Enter number of edges: "))

# 2. Create weighted graph
for i in range(m):
    u, v, w = input("Enter edge (source destination weight): ").split()

    w = int(w)

    graph[u].append((v, w))
    graph[v].append((u, w))

# 3. Take heuristic values
h = {}

for node in nodes:
    h[node] = int(input("Enter heuristic for " + node + ": "))

start = input("Enter start node: ")
goal = input("Enter goal node: ")

# 4. A* Search
pq = [(h[start], 0, start)]
cost = {node: float('inf') for node in nodes}
cost[start] = 0

parent = {}

while pq:
    f, g, current = heapq.heappop(pq)

    if g != cost[current]:
        continue

    if current == goal:
        break

    for neighbor, weight in graph[current]:
        new_cost = g + weight

        if new_cost < cost[neighbor]:
            cost[neighbor] = new_cost
            parent[neighbor] = current

            f = new_cost + h[neighbor]
            heapq.heappush(pq, (f, new_cost, neighbor))

# 5. Display shortest path
if cost[goal] == float('inf'):
    print("No path found!")
else:
    path = []
    current = goal

    while current != start:
        path.append(current)
        current = parent[current]

    path.append(start)
    path.reverse()

    print("\nOptimal Path:")
    print(" -> ".join(path))

    print("Total Path Cost:", cost[goal])
