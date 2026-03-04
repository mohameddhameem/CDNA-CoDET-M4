import sys, os.path

if(os.path.exists('input.txt')):

    sys.stdin = open("input.txt","r")

    sys.stdout = open("outpput.txt","w")



t=int(input())

for i in range(t):

	n,d=map(int,input().split())

	l=list(map(int,input().split()))

	l.sort()

	if l[n-1]<=d:

		print("YES")

	else:

		if l[0]+l[1]<=d:

			print("YES")

		else:

			print("NO")







