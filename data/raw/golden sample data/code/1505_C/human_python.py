T=input()

if len(T)<=2:

    print("YES")

else:

    flag=False

    for i in range(2,len(T)):

        if ord(T[i])-64!=((ord(T[i-1])+ord(T[i-2])-129)%26):

            flag=True

            break

    if flag:

        print("NO")

    else:

        print("YES")