import collections

def count_substrings(s, t):
    count = 0
    for i in range(len(s) - 1):
        if s[i:i+2] == t:
            count += 1
    return count

def optimal_replacements(s, t, k):
    if k > len(s):
        return -1
    cnt = 0
    for i in range(len(s)):
        c = s[i]
        s = s.replace(c, 'a', 1)
        cnt += count_substrings(s, t)
        if cnt > k:
            break
    return count_substrings(s, t)

if __name__ == '__main__':
    s = input()
    t = input()
    k = int(input())
    print(optimal_replacements(s, t, k))