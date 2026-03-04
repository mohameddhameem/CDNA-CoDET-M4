#include <bits/stdc++.h>
using namespace std;

int main() {
	int t; cin >> t;
	for (int i = 0; i < t; ++i) {
		long n; cin >> n;
		bool look = true;
		bool ft = false;
		bool tt = false;
		long sum = 0;
		vector<bool> a(n);
		for (long j = 0; j < n; ++j) {
			int x; cin >> x;
			a[j] = (x != 1);
			sum += x;
			if (j > 0 && look) {
				if (a[j] && a[j-1]) {
					look = false;
					sum += 4;
				} else if (!a[j] && !a[j-1]) {
					tt = true;
				} else if ((!a[j] && a[j-1] || a[j] && !a[j-1])) {
					ft = true;
				}
			}
		}
		if (look) {
			if (!ft) sum -= 4;
		}
		cout << sum << "\n";
	}
}
