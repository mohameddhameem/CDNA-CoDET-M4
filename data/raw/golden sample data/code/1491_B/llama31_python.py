def min_cost(a, k):
    n = len(a)
    ans = float('inf')
    for i in range(n):
        max_val = 0
        min_val = float('inf')
        for j in range(n):
            if a[j] == a[i]:
                p = min(k, a[j] // (a[j] // a[i]))
                max_val = max(max_val, a[j] // p)
                min_val = min(min_val, a[j] // p)
        ans = min(ans, max_val - min_val)
    return ans

def main():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        a = list(map(int, input().split()))
        print(min_cost(a, k))

if __name__ == "__main__":
    main()