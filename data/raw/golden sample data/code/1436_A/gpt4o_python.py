def can_reorder(t, test_cases):
    results = []
    for n, s in test_cases:
        # We can always reorder the string to avoid palindromic substrings
        # of length greater than 1 if there are at most one type of character.
        if s.count('0') > 1 and s.count('1') > 1:
            results.append("NO")
        else:
            results.append("YES")
    return results

# Read input
t = int(input())
test_cases = [(int(input()), input().strip()) for _ in range(t)]

# Get results
results = can_reorder(t, test_cases)

# Print results
print("\n".join(results))