import sys

def tricolor(n, m, edges):
    # Check if it is possible to color the graph with tricolors
    # and return the corresponding coloring or -1 if it is not possible
    if not is_tricolorable(n, m, edges):
        return -1
    
    # Find the number of vertices with even degree
    num_odd = 0
    for i in range(n):
        if edges[i].degree % 2 == 1:
            num_odd += 1
    
    # Initialize the colors
    colors1 = [-1] * n
    colors2 = [-1] * n
    colors3 = [-1] * n
    
    # Color the vertices with even degree
    for i in range(n):
        if edges[i].degree % 2 == 0:
            if colors1[i] == -1:
                colors1[i] = 1
            elif colors2[i] == -1:
                colors2[i] = 2
            else:
                colors3[i] = 3
    
    # Color the remaining vertices
    for i in range(n):
        if colors1[i] == -1:
            colors1[i] = 1
        elif colors2[i] == -1:
            colors2[i] = 2
        else:
            colors3[i] = 3
    
    return colors

def is_tricolorable(n, m, edges):
    # Check if it is possible to color the graph with tricolors
    # Returns True if it is possible, False otherwise
    # This function is based on the theory that a tricolorable graph
    # must have an even number of vertices with even degree
    # and no more than one vertex with odd degree
    num_odd = 0
    for i in range(n):
        if edges[i].degree % 2 == 1:
            num_odd += 1
    if num_odd > 1:
        return False
    return True

n, m = map(int, sys.stdin.readline().split())
edges = []
for _ in range(m):
    edges.append(list(map(int, sys.stdin.readline().split())))
ans = tricolor(n, m, edges)
for edge in ans:
    print(edge, end=" ")