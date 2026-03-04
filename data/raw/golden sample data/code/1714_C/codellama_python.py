import math

def find_min_varied_number(s):
    # Find the minimum number with the given sum of digits
    # such that all digits are distinct (i.e. all digits are unique)
    min_varied_number = 1
    while True:
        # If the number is greater than or equal to the given sum, break
        if min_varied_number >= s:
            break
        # If the number is prime, return it
        if is_prime(min_varied_number):
            return min_varied_number
        # Otherwise, increase the number by 1
        min_varied_number += 1

def is_prime(n):
    # If the number is less than or equal to 1, return False
    if n <= 1:
        return False
    # If the number is 2 or 3, return True
    if n == 2 or n == 3:
        return True
    # If the number is even, return False
    if n % 2 == 0:
        return False
    # If the number is a multiple of 3, return False
    if n % 3 == 0:
        return False
    # Test the number n against the first 50 possible divisors
    for i in range(5, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    # If the number passes all tests, return True
    return True

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        s = int(input())
        print(find_min_varied_number(s))