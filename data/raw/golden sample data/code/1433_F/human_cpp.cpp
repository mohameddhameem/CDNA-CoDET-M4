#include <bits/stdc++.h>

using namespace std;



const int N = 71, oo = (int)1e8;



int dp[N][N][N][N], arr[N][N], n, m, k;



int solve(int i, int j, int rem, int cnt) {

    if(j > m) {

        i++;

        j = 1;

        cnt = 0;

        if(i > n) return !rem? 0: -oo;

    }



    int &ret = dp[i][j][rem][cnt];

    if(~ret) return ret;



    int c1 = -oo, c2 = -oo;

    if(cnt + 1 <= m / 2) c1 = arr[i][j] + solve(i, j + 1, (rem + arr[i][j]) % k, cnt + 1);

    c2 = solve(i, j + 1, rem, cnt);



    return ret = max(c1, c2);



}



int main() {

    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);

    memset(dp, -1, sizeof dp);



    cin >> n >> m >> k;

    for(int i = 1; i <= n; ++i) {

        for(int j = 1; j <= m; ++j) {

            cin >> arr[i][j];

        }

    }



    cout << solve(1, 1, 0, 0) << "\n";



    return 0;

}

