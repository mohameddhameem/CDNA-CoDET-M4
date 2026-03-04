from collections import Counter
import sys
input = sys.stdin.read

def longest_strike(test_cases):
    results = []
    for n, k, arr in test_cases:
        count = Counter(arr)
        valid_numbers = [num for num in count if count[num] >= k]
        
        if not valid_numbers:
            results.append("-1")
            continue
        
        l, r = min(valid_numbers), max(valid_numbers)
        results.append(f"{l} {r}")
    
    return results

def main():
    data = input().split()
    t = int(data[0])
    idx = 1
    test_cases = []
    
    for _ in range(t):
        n, k = map(int, data[idx:idx+2])
        arr = list(map(int, data[idx+2:idx+2+n]))
        test_cases.append((n, k, arr))
        idx += 2 + n
    
    results = longest_strike(test_cases)
    print("\n".join(results))

if __name__ == "__main__":
    main()