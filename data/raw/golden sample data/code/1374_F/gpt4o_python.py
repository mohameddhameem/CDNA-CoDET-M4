def cyclic_shifts_sorting(t, test_cases):
    results = []
    
    for n, a in test_cases:
        # Check for impossible cases first
        if sorted(a) == a:
            results.append("0")
            continue
        
        # Identify the sorted version of the list
        sorted_a = sorted(a)
        operations = []
        
        for i in range(n - 2):
            while a != sorted_a:
                # Find the first mismatch
                if a[i] == sorted_a[i]:
                    i += 1
                    continue
                
                # Perform a cyclic shift
                a[i], a[i + 1], a[i + 2] = a[i + 2], a[i], a[i + 1]
                operations.append(i + 1)  # Store 1-based index
                
                if a == sorted_a:
                    break
        
        if a == sorted_a:
            results.append(f"{len(operations)}")
            results.append(" ".join(map(str, operations)))
        else:
            results.append("-1")
    
    return "\n".join(results)

# Reading input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []
index = 1
for _ in range(t):
    n = int(data[index])
    a = list(map(int, data[index + 1].split()))
    test_cases.append((n, a))
    index += 2

# Get the results
output = cyclic_shifts_sorting(t, test_cases)
print(output)