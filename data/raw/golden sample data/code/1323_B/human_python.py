n,m,k=map(int,input().split())

a=[len(i)for i in input().replace(' ','').split('0')if len(i)>0]

b=[len(i)for i in input().replace(' ','').split('0')if len(i)>0]

d=[(i,k//i)for i in range(1,int(k**0.5)+1)if k%i==0]

d+=[(j,i)for i,j in d if i!=j]

c=0

for x,y in d:c+=sum(i-x+1 for i in a if x<=i)*sum(j-y+1 for j in b if y<=j)

print(c)