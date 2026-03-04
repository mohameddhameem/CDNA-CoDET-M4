import sys

def query_and(i, j):
    print(f"AND {i} {j}")
    sys.stdout.flush()
    return int(input().strip())

def query_or(i, j):
    print(f"OR {i} {j}")
    sys.stdout.flush()
    return int(input().strip())

def query_xor(i, j):
    print(f"XOR {i} {j}")
    sys.stdout.flush()
    return int(input().strip())

def main():
    n = int(input().strip())
    
    # Prepare storage for results
    a = [0] * n

    # Step 1: Gather initial values using OR and AND
    or_1_2 = query_or(1, 2)
    or_1_3 = query_or(1, 3)
    or_2_3 = query_or(2, 3)

    and_1_2 = query_and(1, 2)
    and_1_3 = query_and(1, 3)
    and_2_3 = query_and(2, 3)

    # Step 2: Calculate a[1], a[2], a[3]
    a[0] = (and_1_2 + and_1_3 - and_2_3)  # a[1]
    a[1] = (or_1_2 - a[0])  # a[2]
    a[2] = (or_1_3 - a[0])  # a[3]

    # Step 3: Calculate a[4] to a[n]
    for i in range(4, n + 1):
        a[i - 1] = query_xor(1, i) ^ a[0]

    # Output the result
    print("! " + " ".join(map(str, a)))
    sys.stdout.flush()

if __name__ == "__main__":
    main()