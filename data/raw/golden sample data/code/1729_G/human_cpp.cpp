#include <bits/stdc++.h>
using namespace std;
using ll = long long;

constexpr int MOD = 1000000007;

struct modnum {
    int x;
    modnum(int64_t _x = 0) : x(int(_x % MOD)) { if (x < 0) x += MOD; }
    int value() const { return x; }
    modnum inv() const { return pow(MOD-2); }

    modnum& operator+=(const modnum& o) {
        x += o.x;
        if (x >= MOD) x -= MOD;
        return *this;
    }
    modnum& operator-=(const modnum& o) {
        x -= o.x;
        if (x < 0) x += MOD;
        return *this;
    }
    modnum& operator*=(const modnum& o) {
        x = int(int64_t(x) * int64_t(o.x) % MOD);
        return *this;
    }
    modnum& operator/=(const modnum& o) {
        return *this *= o.inv();
    }

    friend modnum operator+(const modnum& a, const modnum& b) {
        modnum ans = a;
        ans += b;
        return ans;
    }
    friend modnum operator-(const modnum& a, const modnum& b) {
        modnum ans = a;
        ans -= b;
        return ans;
    }
    friend modnum operator*(const modnum& a, const modnum& b) {
        modnum ans = a;
        ans *= b;
        return ans;
    }
    friend modnum operator/(const modnum& a, const modnum& b) {
        modnum ans = a;
        ans /= b;
        return ans;
    }

    modnum pow(int64_t b) const {
        modnum ans(1);
        modnum a = *this;
        while (b) {
            if (b % 2 == 1) ans = ans * a;
            a *= a; b /= 2;
        }
        return ans;
    }
};

const int inf = 1E9;
const int N = 505;
pair<int, modnum> dp[N][N];
modnum fact[N], invfact[N];

modnum C(int n, int k) {
    return fact[n] * invfact[k] * invfact[n - k];
}

vector<int> Z(const string& s) {
    int n = s.length();
    vector<int> z(n);
    for (int i = 1, l = 0, r = 0; i < n; i++) {
        if (i <= r) z[i] = min(r - i + 1, z[i - l]);
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) z[i]++;
        if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
    }
    return z;
}

void solve() {
    string s, t;
    cin >> s >> t;
    int n = s.length(), m = t.length();

    vector<int> occ;
    {
        auto z = Z(t + "$" + s);
        for (int i = 0; i < n; i++) {
            if (z[i + m + 1] >= m) {
                occ.push_back(i);
            }
        }
    }
    vector<int> next_left(occ.size()), next_right(occ.size());
    for (int i = 0; i < occ.size(); i++) {
        next_right[i] = upper_bound(occ.begin(), occ.end(), occ[i] + m - 1) - occ.begin();
        next_left[i] = lower_bound(occ.begin(), occ.end(), occ[i] - m + 1) - occ.begin() - 1;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dp[i][j] = {-1, 0};
        }
    }

    function<pair<int, modnum>(int, int)> go = [&](int i, int j) -> pair<int, modnum> {
        if (i > j) return {0, 1LL};
        if (dp[i][j].first != -1) return dp[i][j];
        if (i == j) return dp[i][j] = {1, 1LL};

        int min_steps = inf;
        for (int k = i; k <= j; k++) {
            auto l = go(i, next_left[k]);
            auto r = go(next_right[k], j);
            min_steps = min(min_steps, 1 + l.first + r.first);
        }
        modnum ways = 0;
        for (int k = i; k <= j; k++) {
            auto l = go(i, next_left[k]);
            auto r = go(next_right[k], j);
            if (1 + l.first + r.first == min_steps) {
                ways += (l.second * r.second * C(min_steps - 1, l.first));
            }
        }
        return dp[i][j] = {min_steps, ways};
    };

    auto ans = go(0, (int) occ.size() - 1);
    cout << ans.first << ' ' << (ans.second / fact[ans.first]).x << '\n';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    fact[0] = invfact[0] = 1;
    for (int i = 1; i < N; i++) {
        fact[i] = modnum(i) * fact[i - 1];
        invfact[i] = fact[i].inv();
    }

    int t = 1;
    cin >> t;
    for (int tc = 1; tc <= t; tc++) {
        // cout << "Case #" << tc << ": ";
        solve();
    }

    return 0;
}
