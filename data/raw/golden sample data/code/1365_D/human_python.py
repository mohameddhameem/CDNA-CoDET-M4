import sys

sys.setrecursionlimit(3000)

for _ in range(int(input())):

    n,m = map(int,input().split())

    maze = []

    for i in range(n):

        s = list(input())

        maze.append(s)

    cnt = 0

    for i in range(n):

        for j in range(m):

            if maze[i][j]=='G':

                cnt+=1

    for i in range(n):

        for j in range(m):

            if maze[i][j]=='B':

                if i+1!=n and maze[i+1][j]!='B': maze[i+1][j] = '#'

                if i-1!=-1 and maze[i-1][j]!='B': maze[i-1][j] = '#'

                if j+1!=m and maze[i][j+1]!='B': maze[i][j+1] = '#'

                if j-1!=-1 and maze[i][j-1]!='B': maze[i][j-1] = '#'

    count = 0

    for i in range(n):

        for j in range(m):

            if maze[i][j]=='G':

                count+=1

    if cnt!=count :

        print("No")

    else:

        for i in range(n):

            for j in range(m):

                if maze[i][j]=='B': maze[i][j]='#'

        mat = [[False for i in range(m)] for j in range(n)]

        visited = set()

        def dfs(i,j):

            visited.add((i,j))

            mat[i][j] = True

            if i+1!=n and maze[i+1][j]!='#' and (i+1,j) not in visited: dfs(i+1,j)

            if i - 1 != -1 and maze[i-1][j] != '#' and (i-1,j) not in visited: dfs(i - 1, j)

            if j + 1 != m and maze[i][j+1] != '#' and (i,j+1) not in visited: dfs(i, j+1)

            if j - 1 != -1 and maze[i][j-1] != '#' and (i,j-1) not in visited: dfs(i, j-1)

        dfs(n-1,m-1)

        ans = "Yes"

        for i in range(n):

            for j in range(m):

                if maze[i][j]=='G':

                    if mat[i][j]!=True:

                        ans = "No"

                        break

            if ans=="No":

                break

        if count>0 and maze[n-1][m-1]=='#':

            print("No")

        else:

            print(ans)