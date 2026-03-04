import itertools

def count_binary_strings(n, a):
    # Initialize the count to 0
    count = 0

    # Loop through each row of the matrix
    for row in a:
        # Generate all possible binary strings of length n
        for s in itertools.product('01', repeat=n):
            # If the string meets the constraints, increment the count
            if meet_constraints(s, row):
                count += 1

    # Return the count modulo 998244353
    return count % 998244353

def meet_constraints(s, row):
    # Loop through each character in the string
    for i in range(len(s)):
        # If the character is not equal to the corresponding value in the row, return False
        if s[i]!= str(row[i]):
            return False

    # If the code reaches this point, all characters are equal, so return True
    return True

# Read the input
n = int(input())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))

# Call the count_binary_strings function and print the result
print(count_binary_strings(n, a))