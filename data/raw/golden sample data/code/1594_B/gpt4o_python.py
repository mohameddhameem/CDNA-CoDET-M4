def special_number(n, k):
    mod = 10**9 + 7
    result = 0
    power = 1
    
    while k > 0:
        if k & 1:
            result = (result + power) % mod
        power = (power * n) % mod
        k >>= 1
    
    return result

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    print(special_number(n, k))