import sys

n, m = list(map(int, input().split()))
mat = []
for i in range(n):
mat.append(list(map(int, input().split())))

def make_even_length(mat):
for i in range(n):
for j in range(m):
if mat[i][j] == 1 and (i-i//2)%2 == 0 and (j-j//2)%2 == 0:
return False
return True

def count_ones(mat):
count = 0
for i in range(n):
for j in range(m):
if mat[i][j] == 1:
count += 1
return count

def make_good(mat):
if make_even_length(mat):
return True
count = count_ones(mat)
for i in range(n):
for j in range(m):
if mat[i][j] == 1:
count -= 1
if count == 0:
return True
return False

if make_good(mat):
print(count)
else:
print(-1)

```