
import heapq

# 1. Take input
start = tuple(map(int, input("Enter start state (9 numbers, use 0 for blank): ").split()))
goal = tuple(map(int, input("Enter goal state (9 numbers, use 0 for blank): ").split()))

if len(start) != 9 or len(goal) != 9 or sorted(start) != list(range(9)) or sorted(goal) != list(range(9)):
    print("Invalid input! Enter numbers 0 to 8 exactly once.")
else:

    # 2. Manhattan Distance Heuristic
    def heuristic(state):
        distance = 0
      
        for i in range(9):
            if state[i] != 0:
                j = goal.index(state[i])

                r1, c1 = divmod(i, 3)
                r2, c2 = divmod(j, 3)

                distance += abs(r1 - r2) + abs(c1 - c2)

        return distance

    # 3. A* Search
    pq = [(heuristic(start), 0, start)]
    cost: dict[tuple[int, ...], int] = {start: 0}
    parent = {}

    while pq:
        f, g, current = heapq.heappop(pq)

        if g != cost[current]:
            continue

        if current == goal:
            break

        # Find blank position
        pos = current.index(0)
        r, c = divmod(pos, 3)

        # Possible moves
        moves = [
            (-1, 0, "Up"),
            (1, 0, "Down"),
            (0, -1, "Left"),
            (0, 1, "Right")
        ]

        for dr, dc, move in moves:
            nr, nc = r + dr, c + dc

            if 0 <= nr < 3 and 0 <= nc < 3:
                new_pos = nr * 3 + nc

                new_state = list(current)
                new_state[pos], new_state[new_pos] = new_state[new_pos], new_state[pos]
                new_state = tuple(new_state)

                new_cost = g + 1

                if new_cost < cost.get(new_state, float('inf')):
                    cost[new_state] = new_cost
                    parent[new_state] = (current, move)

                    f = new_cost + heuristic(new_state)
                    heapq.heappush(pq, (f, new_cost, new_state))

    # 4. Display solution
    if goal not in cost:
        print("No solution found!")
    else:
        path = []
        current = goal

        while current != start:
            previous, move = parent[current]
            path.append((current, move))
            current = previous

        path.reverse()

        print("\nInitial State:")
        print(start[:3])
        print(start[3:6])
        print(start[6:9])

        for state, move in path:
            print("\nBlank moves:", move)
            print(state[:3])
            print(state[3:6])
            print(state[6:9])

        print("\nGoal Reached!")
        print("Total Moves:", cost[goal])
