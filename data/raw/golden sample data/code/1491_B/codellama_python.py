# Read input
n_cases = int(input())
cases = []
for i in range(n_cases):
    cases.append(list(map(int, input().split())))

# Define function to calculate min cost
def min_cost(n, m, k):
    # Calculate the minimum cost to add k edges
    cost = k * (k + 1) // 2

    # Add the cost of adding edges with weight k + 1
    if k > 1:
        cost += (k - 1) * (k + 1)

    # Return the minimum cost
    return cost

# Iterate through cases and output the minimum cost
for case in cases:
    n, m = case[0], case[1]
    if n * (n - 1) // 2 < m:
        print(-1)
    else:
        k = 1
        while m > k:
            k += 1
            m -= k
        print(min_cost(n, m, k))