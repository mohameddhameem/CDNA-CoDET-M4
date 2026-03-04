m,n=map(int,input().split())

A=*map(input,['']*m),

S=A[0][0]<'.'

i=j=0

while i<m-1 or j<n-1:

 if j>n-2or i<m-1and'.'>A[i+1][j]:i+=1

 else:j+=1

 S+=A[i][j]<'.'

print(+S)