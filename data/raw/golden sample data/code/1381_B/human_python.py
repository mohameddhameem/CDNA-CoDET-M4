t = int(input())

for i in range(t):

  

    n = int(input())

    arr = list(map(int,input().split()))

    list_temp = []

    count = 1

    max_num = arr[0]

  

    for i in range(1, 2*n):

        if arr[i] > max_num:

            max_num = arr[i]

            list_temp.append(count)

            count = 1

        else:

            count += 1

    if count:

        list_temp.append(count)

    

    dp = 1

    for i in list_temp:

        dp |= dp << i

    if dp & (1 << n):

        print("YES")

    else:

        print("NO")