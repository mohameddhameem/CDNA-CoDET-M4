// LUOGU_RID: 93945740
#include <bits/stdc++.h>

#define debug std::cout << "debug " << __LINE__ << std::endl;

typedef long long LL;

namespace IO {

	char buf[(1 << 23) | 5], *p1 = buf, *p2 = buf, obuf[(1 << 23) | 5], *O = obuf;

	char getc() { return p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, 1 << 23, stdin), p1 == p2) ? EOF : *p1 ++; }

	inline void End() { fwrite(obuf, O - obuf, 1, stdout), O = obuf; }

	inline void _(char ch) { *O++ = ch; if (O - obuf > (1 << 23)) End(); }

	inline void __(std::string s) { int k = s.size(); for (int i = 0; i < k; ++ i) _(s[i]); }

	void print(LL x) { if (x < 0) _('-'), x = -x; if (x > 9) print(x / 10); _(x % 10 ^ 48); }

	inline int read() {

		int res = 0; char ch = getc(); bool bo = false;

		while (ch < '0' || ch > '9') bo = (ch == '-'), ch = getc();

		while (ch >= '0' && ch <= '9') res = (res << 1) + (res << 3) + (ch ^ 48), ch = getc();

		return bo ? -res : res;

	}

	template<typename T> void read(T &x) { x = read(); }

	template<typename T, typename ...Argv> void read(T &x, Argv &...argv) { read(x); read(argv...); }

}

using IO::read;

const int N = 510;

int a[N], vis[N];

void swap(int &a, int &b, int &c) {

	std::swap(b, c);

	std::swap(a, b);

}

std::vector<int> ans;

void swap(int x) {

	ans.push_back(x);

	swap(a[x], a[x + 1], a[x + 2]);

}



int main() {

	int T = read();

	for (; T -- > 0; ) {

		int n = read();

		for (int i = 1; i <= n; ++ i) a[i] = read();

		ans.clear();

		for (int i = 1; i + 2 <= n; ++ i) {

			int t = i;

			for (int j = i; j <= n; ++ j) {

				if (a[t] > a[j]) t = j;

			}

			while (t - 2 >= i) swap(t -= 2);

			if (t == i + 1) swap(i), swap(i);

		}

		for (int i = n; i >= 3; -- i) {

			int t = i;

			for (int j = i; j >= 1; -- j) {

				if (a[t] < a[j]) t = j;

			}

			while (t + 2 <= i) { swap(t); swap(t); t += 2; }

			if (t == i - 1) swap(i - 2);

		}

		int siz = ans.size();

		for (int i = 1; i < n; ++ i) if (a[i] > a[i + 1]) siz = -1;

		printf("%d\n", siz);

		if (siz >= 0) {

			for (auto e : ans) printf("%d ", e);

			puts("");

		}

	}

}

