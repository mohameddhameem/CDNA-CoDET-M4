def max_paint(n, a):
    from collections import defaultdict
    
    positions = defaultdict(list)
    
    for idx, value in enumerate(a):
        positions[value].append(idx)
    
    count = 0
    
    for indices in positions.values():
        m = len(indices)
        if m >= 3:
            count += m // 3
    
    return count

n = int(input())
a = list(map(int, input().split()))
print(max_paint(n, a))