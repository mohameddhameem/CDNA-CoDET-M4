def round_down_price(m):
    k = int(math.log10(m))
    d = 10**k - m
    return d