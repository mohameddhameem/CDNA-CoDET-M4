def dominant_character(t, test_cases):
    results = []
    for n, s in test_cases:
        min_length = float('inf')
        count_a = count_b = count_c = 0
        
        for i in range(n):
            if s[i] == 'a':
                count_a += 1
            elif s[i] == 'b':
                count_b += 1
            elif s[i] == 'c':
                count_c += 1
            
            if count_a > count_b and count_a > count_c:
                # Try to find the smallest valid substring
                for j in range(i + 1):
                    if s[j] == 'a':
                        count_a -= 1
                    elif s[j] == 'b':
                        count_b -= 1
                    elif s[j] == 'c':
                        count_c -= 1
                    
                    if count_a > count_b and count_a > count_c:
                        min_length = min(min_length, i - j + 1)
                        break  # No need to continue reducing the start

                # Restore counts for the next iteration
                count_a = count_b = count_c = 0
                for k in range(i + 1):
                    if s[k] == 'a':
                        count_a += 1
                    elif s[k] == 'b':
                        count_b += 1
                    elif s[k] == 'c':
                        count_c += 1
        
        results.append(min_length if min_length != float('inf') else -1)
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = [(int(data[i * 2 + 1]), data[i * 2 + 2]) for i in range(t)]
results = dominant_character(t, test_cases)

# Output results
print('\n'.join(map(str, results)))