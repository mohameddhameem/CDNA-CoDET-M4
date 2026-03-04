t = int(input())

while t > 0:

    t = t - 1

    n = int(input())

    arr_input = input().split()

    arr = []

    isUgly = False

    sum = 0

    for i in range(n):

        arr.append(int(arr_input[i]))

    for num in arr:

        if num == sum:

            isUgly = True

            break

        else:

            sum += num

    output_arr = []

    left = 0

    right = len(arr) - 1

    output = ""

    if not isUgly:

        for i in range(len(arr) - 1):

            output += str(arr[i]) + " "

        output += str(arr[-1])

        print("YES")

        print(output)

    else:

        arr.sort()

        while right >= left:

            if left == right:

                output_arr.append(arr[left])

            else:

                output_arr.append(arr[left])

                output_arr.append(arr[right])

            right -= 1

            left += 1

        sum = 0

        isUgly = False

        for num in output_arr:

            if num == sum:

                isUgly = True

                break

            else:

                sum += num

        if not isUgly:

            for i in range(len(output_arr) - 1):

                output += str(output_arr[i]) + " "

            output += str(output_arr[-1])

            print("YES")

            print(output)

        else:

            print("NO")