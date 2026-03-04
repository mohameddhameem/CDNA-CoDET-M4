from collections import defaultdict
import sys
import math

input = sys.stdin.read
data = input().split()

n = int(data[0])
k = list(map(int, data[1:n + 1]))

# Precompute factorials and their distances
max_k = max(k)
factorials = [math.factorial(i) for i in range(max_k + 1)]
distances = defaultdict(int)

# Calculate distances based on the given rules
def lowest_prime_divisor(x):
    if x <= 1:
        return x
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return i
    return x

def get_distance(x):
    if x in distances:
        return distances[x]
    
    if x == 1:
        return 0
    
    f = lowest_prime_divisor(x)
    distance = 1 + get_distance(x // f)
    distances[x] = distance
    return distance

total_distance = defaultdict(int)

# Count how many fragments are at each factorial node
for ki in k:
    fact = factorials[ki]
    total_distance[fact] += 1

# Calculate the minimum sum of distances
min_sum_distance = float('inf')

for node in total_distance.keys():
    current_distance_sum = 0
    for fact, count in total_distance.items():
        current_distance_sum += get_distance(fact) * count
    min_sum_distance = min(min_sum_distance, current_distance_sum)

print(min_sum_distance)