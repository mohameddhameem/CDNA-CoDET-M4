def check(table, m, n):
    # Check if there is a 1 in the table
    for i in range(m):
        for j in range(n):
            if table[i][j] == 1:
                return True
    return False

def paint(table, m, n):
    # Paint the table with a chess coloring
    for i in range(m):
        for j in range(n):
            if table[i][j] == 1:
                table[i][j] = 0
            else:
                table[i][j] = 1

def solve(m, n):
    # Check if it is possible to get the desired picture
    if check(table, m, n) == False:
        return -1

    # Initialize the table with zeros
    table = [[0 for _ in range(n)] for _ in range(m)]

    # Paint the table with a chess coloring
    paint(table, m, n)

    # Calculate the number of operations needed to get the desired picture
    q = 0
    for i in range(m):
        for j in range(n):
            if table[i][j] == 1:
                q += 1

    return q

# Read input
t = int(input())
for _ in range(t):
    m, n = map(int, input().split())
    table = []
    for _ in range(m):
        table.append(list(map(int, input().split())))

    # Solve the problem
    q = solve(m, n)
    if q == -1:
        print("-1")
    else:
        print(q)
        for i in range(m):
            for j in range(n):
                if table[i][j] == 1:
                    print(i+1, j+1, i+1, j+1)