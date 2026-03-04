def find_spy_string(n, m, strings):
    for i in range(n):
        for j in range(m):
            candidate = list(strings[i])
            for ch in 'abcdefghijklmnopqrstuvwxyz':
                if ch != candidate[j]:
                    candidate[j] = ch
                    candidate_str = ''.join(candidate)
                    if all(sum(c1 != c2 for c1, c2 in zip(candidate_str, s)) <= 1 for s in strings):
                        return candidate_str
                    candidate[j] = strings[i][j]
    return "-1"

t = int(input())
results = []
for _ in range(t):
    n, m = map(int, input().split())
    strings = [input().strip() for _ in range(n)]
    results.append(find_spy_string(n, m, strings))

print("\n".join(results))