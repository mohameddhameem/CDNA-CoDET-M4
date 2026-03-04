def find_unique_number(x):
    # Generate all possible numbers with the given number of digits
    num_digits = len(str(x))
    all_numbers = [str(i) for i in range(10**num_digits)]
    # Check if any of the numbers have the required sum of digits and are unique
    for number in all_numbers:
        sum_digits = sum(int(digit) for digit in number)
        if sum_digits == x:
            digits = set(number)
            if len(digits) == x:
                return int(''.join(sorted(digits)))
    return -1