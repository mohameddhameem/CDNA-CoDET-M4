import sys
import math

def query(S):
    print(f"? {len(S)} " + " ".join(map(str, S)))
    sys.stdout.flush()
    return input().strip()

def guess(x):
    print(f"! {x}")
    sys.stdout.flush()
    return input().strip()

n = int(input().strip())
low, high = 1, n
answers = []
prev_answer = None

for _ in range(53):
    mid = (low + high) // 2
    S = list(range(low, mid + 1))
    
    answer = query(S)
    answers.append(answer)
    
    if answer == "YES":
        low = mid + 1
    else:
        high = mid

# Now we have a list of answers and we need to determine the actual number
# We need to analyze the responses we received.
possible = set(range(1, n + 1))

for i in range(len(answers)):
    if answers[i] == "YES":
        possible.intersection_update(set(range(low, n + 1)))
    else:
        possible.intersection_update(set(range(1, low)))

# We can make two guesses based on the remaining possible numbers
for num in possible:
    if guess(num) == ":)":
        break
    possible.remove(num)

if len(possible) > 0:
    guess(possible.pop())