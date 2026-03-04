// LUOGU_RID: 101203905
#include <bits/stdc++.h>



using namespace std;

using ll = long long;



constexpr int N = 2e5 + 10;



ll a[] = {0, 2, 3, 1};

void solve() {

	ll n;

	cin >> n;

	--n;

	ll k = n / 3, s = 1;

	while(k >= s)

		k -= s, s *= 4;

	ll x = s + k, y = 2 * s;

	for(int i = 0; (1ll << i) < s; i += 2)

		y += a[k >> i & 3] << i;

	if(n % 3 == 0)

		cout << x << '\n';

	else if(n % 3 == 1)

		cout << y << '\n';

	else 

		cout << (x ^ y) << '\n';

}



int main() {

	ios::sync_with_stdio(false);

	cin.tie(nullptr);

	

	int t = 1;

	cin >> t;

	while(t--) {

		solve();

	}

	

	return 0;

}