def game_of_life(t, test_cases):
    results = []
    for n, m, initial_state in test_cases:
        state = list(initial_state)
        if m > 0:
            for _ in range(min(m, 2)):  # We only need to simulate up to 2 iterations
                new_state = state[:]
                for i in range(n):
                    if state[i] == '0':
                        alive_neighbors = (state[i-1] == '1') + (state[i+1] == '1' if i + 1 < n else 0)
                        if alive_neighbors == 1:
                            new_state[i] = '1'
                state = new_state
        results.append(''.join(state))
    return results

# Reading input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []
index = 1
for _ in range(t):
    n, m = map(int, data[index].split())
    initial_state = data[index + 1]
    test_cases.append((n, m, initial_state))
    index += 2

# Getting the results
results = game_of_life(t, test_cases)

# Printing the results
print('\n'.join(results))