def round_down_price(t, prices):
    results = []
    for m in prices:
        m = int(m)
        power_of_ten = 1
        while power_of_ten * 10 <= m:
            power_of_ten *= 10
        d = m - power_of_ten
        results.append(d)
    return results

t = int(input())
prices = [input().strip() for _ in range(t)]
results = round_down_price(t, prices)
print('\n'.join(map(str, results)))