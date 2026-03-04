#include <bits/stdc++.h>

#pragma GCC optimize ("unroll-loops")

#pragma GCC optimize ("O3")

#define ll long long

#define st string

#define ull unsigned ll

#define pii pair <int, int>

#define pll pair <ll, ll>

#define pb push_back

#define ins insert

#define F first

#define S second

#define int ll

using namespace std;

int32_t main () {

    //freopen ("points.in", "r", stdin);

    //freopen ("points.out", "w", stdout);

    ios_base::sync_with_stdio();

    cin.tie(0);

    cout.tie(0);

    int t=1;

    cin >> t;

    while (t --) {

        int n;

        cin >> n;

        vector <int> v;

        set <int> sat;

        for (int i = 0; i < n; i ++) {

            int a;

            cin >> a;

            v.pb(a);

            sat.ins(a);

        }

        if (sat.size() == 1) {

            cout << "NO\n";

        }

        else {

            cout << "YES\n";

            sort(v.rbegin(), v.rend());

            int j = n-1;

            for (int i = 1; i < n; i ++) {

                if (v [i] == v [i - 1]) {

                    swap(v [i], v [j]);

                    j --;

                }

            }

            for (auto to: v) {

                cout << to << " ";

            }

            cout << "\n";

        }

    }

    return 0;

}

