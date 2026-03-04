from collections import deque

def frog_traveler(n, a, b):
    visited = [False] * (n + 1)
    queue = deque([(n, 0)])  # (current depth, jumps)
    visited[n] = True
    prev = [-1] * (n + 1)  # to reconstruct the path

    while queue:
        current_depth, jumps = queue.popleft()
        if current_depth == 0:
            path = []
            while current_depth != n:
                path.append(current_depth)
                current_depth = prev[current_depth]
            path.reverse()
            return (jumps, path)
        
        max_jump = a[current_depth - 1]
        for jump in range(max_jump + 1):
            next_depth = current_depth - jump + b[current_depth - 1]
            if next_depth >= 0 and not visited[next_depth]:
                visited[next_depth] = True
                queue.append((next_depth, jumps + 1))
                prev[next_depth] = current_depth

    return -1

n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

result = frog_traveler(n, a, b)
if result == -1:
    print(-1)
else:
    jumps, path = result
    print(jumps)
    print(' '.join(map(str, path)))