// LUOGU_RID: 101032166
#include<bits/stdc++.h>

using namespace std;

int a[10000],t,n,cnt,Max,x;

bitset<10000> b;

int main()

{

	scanf("%d",&t);

	while(t--)

	{

		scanf("%d",&n);

		Max=0;

		cnt=0;

		for(int i=1;i<=2*n;i++)

		{

			scanf("%d",&x);

			if(x<Max) a[cnt]++;

			else a[++cnt]=1,Max=x;

		}

		b&=0;

		b[0]=1;

		for(int i=1;i<=cnt;i++)

			b|=b<<a[i];

		if(b[n]) printf("YES\n");

		else printf("NO\n");

	}

	

	return 0;

}