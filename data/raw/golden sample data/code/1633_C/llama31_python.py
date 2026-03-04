def kill_the_monster():
    t = int(input())
    for _ in range(t):
        hc, dc = map(int, input().split())
        hm, dm = map(int, input().split())
        k, w, a = map(int, input().split())

        if (hm + dc - 1) // dc <= (hc + (k * a) - 1) // (dc + (k * w)):
            print("YES")
        else:
            print("NO")

kill_the_monster()