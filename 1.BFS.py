# 1.Best First Search

def best_first_search(graph,start,goal,heuristic, path=[]):
    open_list = [(0,start)]
    closed_list = set()
    closed_list.add(start)

    while open_list:
        open_list.sort(key = lambda x: heuristic[x[1]], reverse=True)
        cost, node = open_list.pop()
        path.append(node)

        if node==goal:
            return cost, path

        closed_list.add(node)
        for neighbour, neighbour_cost in graph[node]:
            if neighbour not in closed_list:
                closed_list.add(node)
                open_list.append((cost+neighbour_cost, neighbour))

    return None

# Define a smaller graph
graph = {
    'S': [('A', 1), ('B', 4)],
    'A': [('C', 2), ('D', 5)],
    'B': [('D', 1)],
    'C': [('G', 3)],
    'D': [('G', 2)],
    'G': []  # Goal node
}

# Define the heuristic values for each node
heuristic = {
    'S': 7,
    'A': 6,
    'B': 2,
    'C': 1,
    'D': 3,
    'G': 0  # Heuristic value for goal node is always 0
}

# Start and goal nodes
start = 'S'
goal = 'G'

# Perform the best first search
result = best_first_search(graph, start, goal, heuristic)

# Output the result
if result:
    print(f"Minimum cost path from {start} to {goal} is {result[1]}")
    print(f"Cost: {result[0]}")
else:
    print(f"No path from {start} to {goal}")
