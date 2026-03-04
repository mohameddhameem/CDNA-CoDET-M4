#include<bits/stdc++.h>

using namespace std;

#define int long long

const int maxn=2e6+10,mod=1e9+7;



signed main()

{

	int t;

	cin >> t;

	while(t--)

	{

		int a,b;

		cin >> a >> b;

		int sum=a+b;

		

		if(sum%2!=0)

		{

			cout << -1 << ' ' << -1 << '\n';

			continue;

		}

		else 

		{

			sum/=2;

			if(a%2==0)cout << a/2 << ' ' << b/2 << '\n';

			else 

			if(a>sum)cout << a-sum << ' ' << b << endl;

			else cout << a << ' ' << b-sum << endl; 

			

		}

	}

	

	return 0;

}