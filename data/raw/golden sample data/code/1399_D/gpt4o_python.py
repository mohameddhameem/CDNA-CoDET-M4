def binary_string_to_subsequences(test_cases):
    results = []
    
    for s in test_cases:
        n = len(s)
        subsequence_indices = [0] * n
        current_subsequence = 1
        
        last_char = ''
        for i in range(n):
            if s[i] != last_char:
                subsequence_indices[i] = current_subsequence
                last_char = s[i]
            else:
                current_subsequence += 1
                subsequence_indices[i] = current_subsequence
            
            if current_subsequence > 2:
                current_subsequence = 1

        # Count how many unique subsequences we used
        k = max(subsequence_indices)
        results.append(f"{k}\n{' '.join(map(str, subsequence_indices))}")

    print("\n".join(results))

# Reading input
import sys
input = sys.stdin.read

data = input().splitlines()
t = int(data[0])
test_cases = [data[i * 2 + 2] for i in range(t)]

binary_string_to_subsequences(test_cases)