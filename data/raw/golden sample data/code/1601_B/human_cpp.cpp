#include<bits/stdc++.h>

using namespace std;

#define ll long long

#define speed ios::sync_with_stdio(0),cin.tie(0),cout.tie(0)

int main()

{

    ll n;

    cin>>n;

    ll a[n],b[n];

    for (ll i=0;i<n;i++)

        cin>>a[i];

    for (ll i=0;i<n;i++)

        cin>>b[i];

    vector<ll> ans;

    vector<ll> c;

    c.push_back(n-1);

    vector<ll> s;

    for (ll i=0;i<n-1;i++)

        s.push_back(i);

    pair<ll,ll> v[n];

    for (ll i=0;i<n;i++)

        v[i].first=v[i].second=0;

    v[n-1].first=1;

    ll z=-1;

    for (ll i=0;i<c.size();i++)

    {

        if (c[i]-a[c[i]]<0)

        {

            z=c[i];

            break;

        }



        if (s.empty())

            continue;

        ll x=s[s.size()-1];

        while(x>=c[i]-a[c[i]])

        {



            s.pop_back();

            if (x+b[x]>=n)

                continue;

            else if (v[x+b[x]].first==0)

            {

                v[x+b[x]].first=x;

                v[x+b[x]].second=c[i];

                c.push_back(x+b[x]);

            }

            if (s.empty())

                break;

            x=s[s.size()-1];

        }

    }

    if (z==-1)

        cout<<z;

    else

    {

        while(z!=n-1)

        {

            ans.push_back(v[z].first);

            z=v[z].second;

        }

        cout<<ans.size()+1<<"\n";

        for (ll i=ans.size()-1;i>=0;i--)

            cout<<ans[i]+1<<" ";

        cout<<0;

    }



}

