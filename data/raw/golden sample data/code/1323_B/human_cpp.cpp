#include <bits/stdc++.h>

using namespace std;

#define ll long long



const int N = 4e5 + 10;

int a[N], b[N];

int n, m, k;

int slove(int x)

{

    int cnt1 = 0, cnt = 0;

    for (int i = 1; i <= n; i++) {

        if (a[i]) cnt1++;

        else cnt1 = 0;

        if (cnt1 >= x) cnt++;

    }

    int cnt2 = 0, y = k / x;

    int res = 0;

    for (int i = 1; i <= m; i++) {

        if (b[i]) cnt2++;

        else cnt2 = 0;

        if (cnt2 >= y) res += cnt;

    }

    return res;

}



int main()

{

    ios::sync_with_stdio(0);

    cout.tie(0), cin.tie(0);



    cin >> n >> m >> k;

    for (int i = 1; i <= n; i++) cin >> a[i];

    for (int i = 1; i <= m; i++) cin >> b[i];

    long long ans = 0;

    for (int i = 1; i <= sqrt(k); i++) {

        if (k % i == 0) {

            ans += slove(i);

            if (k / i != i) {

                ans += slove(k / i);

            }

        }

    }

    cout << ans << endl;



    return 0;

}

