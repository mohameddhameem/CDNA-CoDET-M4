#include<bits/stdc++.h>

#include<ext/pb_ds/tree_policy.hpp> 

#include <ext/pb_ds/assoc_container.hpp>

#include<functional> 

 using namespace __gnu_pbds;

using namespace std;

template <typename T> using ordered_set = tree<T, null_type,less_equal<T>, rb_tree_tag,tree_order_statistics_node_update>;  

typedef long long ll;

#define l(z,a,n) for(ll z=a;z<n;z++)

#define ll          long long

#define vi          vector<int>

#define vll         vector<ll>

#define pll         pair<ll, ll>

#define pii         pair<int, int>

#define ld          long double

#define ff          first

#define ss          second

#define vs          vector<string>

#define vpll        vector<pll>

#define vb          vector<bool>

#define mp          make_pair

#define pb          push_back

#define all(x)      x.begin(),x.end()

#define elapsed_time 1.0 * clock() / CLOCKS_PER_SEC

#define endl        '\n'

#ifndef ONLINE_JUDGE

#define debug(x) cerr << #x <<'='; cerr<<x; ; cerr << endl;

#else

#define debug(x)

#endif



const ll INF       = 2e18;

const ll mod       = 1000000007;

const ll mod2      = 998244353;





template <typename T, typename Y> ostream& operator << (ostream& x, const pair<T,Y>& v) {x<<v.ff<<" "<<v.ss; return x;}

template <typename T, typename Y> ostream& operator << (ostream& x, const map<T, Y>& v) {for (auto it : v) x << it.ff << ": " << it.ss << endl; return x;}

template <typename T, typename Y> istream& operator >> (istream& x, pair<T,Y>& v) {x>>v.ff>>v.ss; return x;}

template <typename T> istream& operator >> (istream& x,vector<T>& v) {for(T& it:v) x>>it;return x;}

template <typename T> ostream& operator << (ostream& x,const vector<T>& v) {for(int i=0;i<(int)v.size();i++) x<<v[i]<<" ";return x;}

template <typename T> ostream& operator << (ostream& x,const set<T>& v) {for(auto it:v) x<<it<<" ";return x;}

template <typename T> ostream& operator << (ostream& x,const multiset<T>& v) {for(auto it:v) x<<it<<" ";return x;}



ll power(ll a, ll b, ll mod){

 ll ans=1;

while(b>0){

 if(b&1)

ans=(ans*a)%mod;

a=(a*a)%mod;

b>>=1;  

}

return ans%mod;

}

const int N = 1e5+5;

vector<int> parent(N);

vector<int> sz(N);

void make(int v){

    parent[v]=v;

    sz[v]=1;

}

int find(int v){

    if(v==parent[v])return v;

    return parent[v]=find(parent[v]);



}

void Union(int a,int b ){

    a = find(a);

    b = find(b);

    if(sz[a]<sz[b])swap(a,b);

    parent[b]=a;

    sz[a]+=sz[b];

}

void solve(){

    int n;

    cin>>n; 

   vector<pair<ll,ll>> v(n);

    l(i,0,n){

        cin>>v[i].ff;

        v[i].ss = i;

    }

    sort(all(v));

    vector<pair<ll,ll>> ans;

    l(i,1,n){

        if(v[i].ff % v[i-1].ff == 0)continue;

        ll temp ;

        temp = ((v[i].ff/v[i-1].ff)+1)*v[i-1].ff;

        ans.pb({v[i].ss,temp-v[i].ff});

        v[i].ff=temp;

    }

    cout<<ans.size()<<endl;

    for(auto &i:ans){

        cout<<i.ff+1<<" "<<i.ss<<endl;

    }



}

int main(){

ios_base::sync_with_stdio(false);

cin.tie(0);

cout.tie(0);



int t;

cin>>t;

while(t--)

solve();

return 0;

} 