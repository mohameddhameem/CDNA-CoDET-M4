n = int(input())



color = "blue"

a = "unlock"



for i in range(n):

    s = input()

    if s != "unlock" and s != "lock" and a != "lock":

        color = s

    elif s == "lock":

        a = "lock"

    elif s == "unlock":

        a = "unlock"

print(color)