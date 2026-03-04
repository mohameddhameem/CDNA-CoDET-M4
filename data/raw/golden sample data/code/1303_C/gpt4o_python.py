def perfect_keyboard(test_cases):
    from collections import defaultdict, deque
    
    results = []
    
    for s in test_cases:
        graph = defaultdict(set)
        for i in range(len(s) - 1):
            graph[s[i]].add(s[i + 1])
            graph[s[i + 1]].add(s[i])
        
        visited = {}
        layout = []
        possible = True

        def bfs(start):
            queue = deque([start])
            visited[start] = True
            component = [start]
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        component.append(neighbor)
            return component

        for char in graph:
            if char not in visited:
                component = bfs(char)
                layout.append(component)

        if len(layout) > 1:
            possible = False
        else:
            # Check for degree constraints
            for key in graph:
                if len(graph[key]) > 2:
                    possible = False
                    break

        if not possible:
            results.append("NO")
            continue

        # Create perfect layout
        used = set()
        final_layout = []

        def dfs(node):
            used.add(node)
            final_layout.append(node)
            for neighbor in graph[node]:
                if neighbor not in used:
                    dfs(neighbor)

        for char in graph:
            if char not in used:
                dfs(char)

        remaining_letters = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) not in used]
        final_layout.extend(remaining_letters)

        results.append("YES")
        results.append("".join(final_layout))

    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
T = int(data[0])
test_cases = data[1:T + 1]

# Solve and print the results
results = perfect_keyboard(test_cases)
for result in results:
    print(result)