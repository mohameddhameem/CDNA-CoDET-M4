#include <bits/stdc++.h>

#define int long long

using namespace std;

const int MAX=2e5;            //for limit of nodes

const int MOD=1e9+7;        //for those annoying ass questions

vector<int>adj[MAX];

void solve() {

    int n,k;

    cin>>n>>k;

    int p=1,ans=0;

    for (int i=0;i<31;i++) {

        if (k&(1<<i)) {

            ans=(ans+p)%MOD;

        }

        p*=n;

        p%=MOD;

    }

    cout<<ans<<"\n";

}

signed main() {

    ios_base::sync_with_stdio(false);

    cin.tie(NULL);

    int t;

    cin>>t;

    while (t--) {

       solve();

    }

return 0;

}

