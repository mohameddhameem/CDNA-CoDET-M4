def max_three_blocks_palindrome(t, test_cases):
    results = []
    
    for _ in range(t):
        n = test_cases[_][0]
        a = test_cases[_][1]
        
        count = [0] * 201
        for num in a:
            count[num] += 1
        
        max_length = 0
        
        for i in range(1, 201):
            if count[i] == 0:
                continue
            for j in range(1, 201):
                if i == j or count[j] == 0:
                    continue
                
                left_count = 0
                right_count = 0
                
                # Count left side
                for k in range(n):
                    if a[k] == i:
                        left_count += 1
                    elif a[k] == j and left_count > 0:
                        break
                
                # Count right side
                for k in range(n-1, -1, -1):
                    if a[k] == i:
                        right_count += 1
                    elif a[k] == j and right_count > 0:
                        break
                
                max_length = max(max_length, left_count + right_count)
        
        for i in range(1, 201):
            max_length = max(max_length, count[i])
        
        results.append(max_length)
    
    return results

# Input processing
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

# Getting results
results = max_three_blocks_palindrome(t, test_cases)

# Output results
for result in results:
    print(result)