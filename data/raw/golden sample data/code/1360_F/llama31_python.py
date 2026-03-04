def spy_string(n, m, strings):
    def check_string(s):
        for string in strings:
            diff = sum(1 for i in range(m) if s[i]!= string[i])
            if diff > 1:
                return False
        return True

    for string in strings:
        for i in range(m):
            new_string = string[:i] + '_' + string[i+1:]
            if check_string(new_string):
                return new_string

    return -1

t = int(input())
for _ in range(t):
    n, m = map(int, input().split())
    strings = [input() for _ in range(n)]
    print(spy_string(n, m, strings))