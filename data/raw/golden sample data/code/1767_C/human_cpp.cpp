#include<bits/stdc++.h>

using namespace std;

const int maxn = 105, mod = 998244353;

int n;

int a[maxn][maxn];

int dp[maxn][maxn];

int p[maxn], lim[maxn];

int ans;

int main() {

    cin >> n;

    for (int i = 1; i <= n; i++) lim[i] = 1;

    for (int i = 1; i <= n; i++)

        for (int j = i; j <= n; j++) {

            cin >> a[i][j];

            if (a[i][i] == 2) {cout << 0 << endl; return 0;}

            if (a[i][j] == 1)

                for (int k = i + 1; k <= j; k++) p[k] = 1;

            if (a[i][j] == 2) lim[j] = max(lim[j], i + 1);

        }

    dp[1][1] = 1;

    for (int i = 2; i <= n; i++) {

        if (!p[i])

            for (int j = 1; j <= i - 1; j++) dp[i][i] = (dp[i][i] + dp[i - 1][j]) % mod;

        for (int j = lim[i]; j < i; j++) dp[i][j] = (dp[i][j] + dp[i - 1][j]) % mod;

    }

    for (int i = lim[n]; i <= n; i++) ans = (ans + dp[n][i]) % mod;

    cout << ans * 2 % mod << endl;

    return 0;

}