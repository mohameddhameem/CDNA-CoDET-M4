def game_of_life(n, m, s):
    s = list(map(int, s))
    for _ in range(m % 4):
        for i in range(n):
            if i == 0 or i == n - 1:
                l, r = 0, 1
            elif i == 1 or i == n - 2:
                l, r = 1, 2
            else:
                l, r = 1, 3
            alive = s[i - l] + s[i] + s[i + r]
            if s[i] == 0 and alive == 1:
                s[i] = 1
            elif s[i] == 1 and (alive < 2 or alive > 3):
                s[i] = 0
    return ''.join(map(str, s))

t = int(input())
for _ in range(t):
    n, m = map(int, input().split())
    s = input()
    print(game_of_life(n, m, s))