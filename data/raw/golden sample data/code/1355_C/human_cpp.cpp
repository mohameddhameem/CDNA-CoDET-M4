#include <bits/stdc++.h>

using namespace std;



#define int long long



int32_t main() {

  ios::sync_with_stdio(false);

  cin.tie(0);

  int a, b, c, d;

  cin >> a >> b >> c >> d;

  const int MAX = b + c;

  vector<int> cnt(MAX + 2);

  for (int x = a; x <= b; x++) {

    cnt[x + b]++;

    cnt[x + c + 1]--;

  }

  for (int i = 1; i <= MAX; i++) {

    cnt[i] += cnt[i - 1];

  }

  int res = 0;

  for (int i = 1; i <= MAX; i++) {

    int l = c;

    int r = min(d, i - 1);

    res += max(0ll, r - l + 1) * cnt[i];

  }

  cout << res << '\n';

  return 0;

}