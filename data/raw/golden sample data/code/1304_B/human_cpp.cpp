#include <bits/stdc++.h>

using namespace std;

#define ll long long 

#define fr(i,a,b,k) for(int i=a;i<b;i+=k)

#define frrev(i,a,b,k) for(int i=a;i>=b;i-=k)

#define NO cout<<"NO\n"

#define YES cout<<"YES\n"

#define V vector<ll int> 

#define VP vector<pair<ll int,ll int>> 

#define MP map<ll int,ll int> 

#define pb push_back

#define ff first 

#define ss second 

#define input(A) for(auto &i:A)cin>>i

#define output(A) for(auto &i:A)cout<<i<<" "

string s[100];

void solve(int t)

{

  set<string> d;

	int n, m, i;

	cin >> n >> m;

	for (i = 0; i < n; i++)

	{

		cin >> s[i];

		d.insert(s[i]);

	}

	vector<string> l, r;

	string mid;

	for (i = 0; i < n; i++)

	{

		string t = s[i];

		reverse(t.begin(), t.end());

		if (t == s[i])

			mid = t;

		else if (d.find(t) != d.end())

		{

			l.pb(s[i]);

			r.push_back(t);

			d.erase(s[i]);

			d.erase(t);

		}

	}

  string ans="";

	for (string x : l)

	ans+=x;

	ans+=mid;

	reverse(r.begin(), r.end());

	for (string x : r)

	ans+=x;

	cout<<ans.size()<<endl;

  cout<<ans;

}

int main()

{

    ios_base::sync_with_stdio(false);

    cin.tie(NULL);

    int t;

    //cin>>t;

    t=1;

    for(int i=1;i<=t;i++)

    {

       solve(i);

    }

    return 0;

}



