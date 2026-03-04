def solve():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    idx = 0
    q = int(data[idx])
    idx += 1
    results = []
    
    for _ in range(q):
        t = data[idx]
        idx += 1
        n = int(data[idx])
        idx += 1
        words = [data[idx + i] for i in range(n)]
        idx += n

        colored = [False] * len(t)
        steps = []

        def color(s, start):
            for i in range(len(s)):
                colored[start + i] = True
            steps.append((words.index(s) + 1, start + 1))

        # A greedy approach to color all characters
        for i in range(len(t)):
            if colored[i]:
                continue
            found = False
            for w in words:
                if t.startswith(w, i):
                    color(w, i)
                    found = True
                    break
            if not found:
                results.append("-1")
                break
        else:
            results.append(str(len(steps)))
            for step in steps:
                results.append(f"{step[0]} {step[1]}")

    sys.stdout.write("\n".join(results) + "\n")

solve()