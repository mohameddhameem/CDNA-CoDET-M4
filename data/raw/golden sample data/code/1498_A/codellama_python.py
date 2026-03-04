def gcdSum(n):
    sum_of_digits = sum([int(digit) for digit in str(n)])
    gcd = math.gcd(n, sum_of_digits)
    return gcd


def get_next_number(n):
    while True:
        n += 1
        gcd = gcdSum(n)
        if gcd > 1:
            return n


t = int(input())

for _ in range(t):
    n = int(input())
    result = get_next_number(n)
    print(result)