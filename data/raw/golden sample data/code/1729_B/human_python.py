from string import ascii_lowercase 

# print(ascii_lowercase)

def solve():

    input()

    n = list(input())

    ans = []

    i = len(n) - 1

    while i >= 0:

        if n[i] == "0":

            ans.append(ascii_lowercase[int(n[i - 2] + n[i - 1]) - 1])

            i -= 2  

        else:

            ans.append(ascii_lowercase[int(n[i]) - 1])

        i -= 1 

    print("".join(ans[::-1]))



for i in range(int(input())):

    solve()