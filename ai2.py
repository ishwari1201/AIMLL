
import heapq

# 1. Take input
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

grid = []

print("Enter grid (0 = empty, 1 = obstacle):")

for i in range(rows):
    row = list(map(int, input().split()))
    grid.append(row)

start = tuple(map(int, input("Enter start row and column: ").split()))
goal = tuple(map(int, input("Enter goal row and column: ").split()))

# 2. Manhattan Distance Heuristic
def heuristic(node):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

# 3. A* Search
pq = [(heuristic(start), 0, start)]
cost = {start: 0}
parent = {}

while pq:
    f, g, current = heapq.heappop(pq)

    if g != cost[current]:
        continue

    if current == goal:
        break

    # Possible movements
    moves = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    for dr, dc in moves:
        nr = current[0] + dr
        nc = current[1] + dc

        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbor = (nr, nc)
            new_cost = g + 1

            if new_cost < cost.get(neighbor, float('inf')):
                cost[neighbor] = new_cost
                parent[neighbor] = current

                f = new_cost + heuristic(neighbor)
                heapq.heappush(pq, (f, new_cost, neighbor))

# 4. Display shortest path
if goal not in cost:
    print("No path found!")
else:
    path = []
    current = goal

    while current != start:
        path.append(current)
        current = parent[current]

    path.append(start)
    path.reverse()

    print("\nShortest Path:")
    print(path)

    print("Total Cost:", cost[goal])

    # Display path in grid
    for r, c in path:
        if (r, c) != start and (r, c) != goal:
            grid[r][c] = '*'

    for row in grid:
        print(*row)
