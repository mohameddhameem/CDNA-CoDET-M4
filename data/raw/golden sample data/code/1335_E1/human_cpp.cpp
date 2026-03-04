// IN GOD WE TRUST ♥
#include <bits/stdc++.h>


#define ll long long
#define all(a) a.begin(), a.end()
#define endl '\n'
#define yes cout << "YES" << endl
#define no cout << "NO" << endl

using namespace std;

void Adham() {
    ios_base::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
}

const int N = 2e5 + 5;
const ll MOD = 1e9 + 7;
const ll inf = INFINITY;
int dx[] = {+0, +0, -1, +1, +1, +1, -1, -1};
int dy[] = {-1, +1, +0, +0, +1, -1, +1, -1};

using namespace std;

void solve() {
    ll n;
    cin >> n;
    vector<ll> arr(n);
    map<ll, ll> c;
    ll res = 0;
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];
        c[arr[i]]++;
        res = max(res, c[arr[i]]);
    }
    for(auto x: c){
        ll beg = 0, en = n-1, f=0, l=0;
        bool t = true;
        while(beg <= en){
            if(f == l && !t){
                for(auto y: c){
                    res = max(res, f+l+y.second);
                }
                t = true;
            }
            else if(f == l && t){
                if(arr[beg] == x.first){
                    f++;
                    t = false;
                }
                c[arr[beg]]--;
                beg++;
            }
            else{
                if(arr[en] == x.first){
                    l++;
                }
                c[arr[en]]--;
                en--;
            }
        }
        for (int i = 0; i < n; ++i) {
            c[arr[i]]++;
        }
    }
    cout << res << endl;
}

int main() {
    Adham();
    int t = 1;
    cin >> t;
    while (t--) {
        solve();
    }
}