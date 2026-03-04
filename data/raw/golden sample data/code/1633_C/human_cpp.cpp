#pragma GCC optimize("O3")

#include <bits/stdc++.h>

#define S second

#define F first

#define ll long long

#define ull unsigned long long

#define ld long double

#define npos ULLONG_MAX

#define INF LLONG_MAX

#define elif else if

#define vv(a) vector<a>

#define pp(a, b) pair<a, b>

#define pq(a) priority_queue<a>

#define qq(a) queue<a>

#define ss(a) set<a>

#define mm(a, b) map<a, b>

#define umm(a, b) unordered_map<a, b>

#define pb push_back

#define sync                      \

    ios_base::sync_with_stdio(0); \

    cin.tie();                    \

    cout.tie();

#define endl "\n"

#define allc(a) begin(a), end(a)

#define all(a) a, a + sizeof(a) / sizeof(a[0])

#define ins insert

using namespace std;

using namespace __cxx11;

typedef char chr;

typedef basic_string<chr> str;



void solve()

{

    ll h, d, hm, dm;

    ll k, w, a;

    cin >> h >> d >> hm >> dm >> k >> w >> a;

    for (ll i = 0; i <= k; i++)

    {

        ll j = k - i;

        if ((hm + d + j * w - 1) / (d + j * w) <= (h + i * a + dm - 1) / dm)

        {

            cout << "yes\n";

            return;

        }

    }

    cout << "no\n";

}

/*



*/

int main()

{

    sync

        // precomp();

        ll t = 1;

    cin >> t;

    for (ll i = 1; i <= t; i++)

        // cout << "Case " << i << ": \n",

        solve();

    cerr << "\nTime elapsed : " << clock() * 1000.0 / CLOCKS_PER_SEC << " ms\n";

}

