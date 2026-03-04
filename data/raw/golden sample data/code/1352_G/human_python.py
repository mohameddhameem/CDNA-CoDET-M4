I=input

for i in ' '*int(I()):

	n=int(input())

	if n<4:print(-1)

	else:print(*list(range(1,n+1,2))[::-1]+[4,2]+list(range(6,n+1,2)))