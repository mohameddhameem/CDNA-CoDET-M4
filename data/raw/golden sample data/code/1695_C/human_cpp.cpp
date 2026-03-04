// Jinqian kan qi xiang shang xiezhe
// Qinxianglian na sanshi er sui na zhuang gao dang chao fuma lang
// ta qi junwang a man huangshang na huihun nan er zhao dongchuang

#include<bits/stdc++.h>
using namespace std;

// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
// using namespace __gnu_pbds;

// template <class T>
// using ordered_set = tree<T, null_type, less<T>, rb_tree_tag,tree_order_statistics_node_update>;

// #pragma GCC target ("avx2")
// #pragma GCC optimize("O3")
// #pragma GCC optimization ("unroll-loops")
// #pragma comment(linker, "/stack:200000000")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
// #pragma GCC target("fpmath=387") //extra precision

//start debugger
#define tpc template<class c
#define ckt(x) tpc> typename enable_if<sizeof ptb<c>(0) x 1, debug&>::type
#define rpt return *this
#define dop debug& operator <<
tpc> struct rge { c b, e; };
tpc> rge<c> range(c x, c y) { return rge<c>{x, y}; }
tpc> auto ptb(c* x) -> decltype(cerr << *x, 0);
tpc> char ptb(...);
struct debug {
#ifdef LOCAL
// ~debug() {cerr << "\u001b[0m\n";}
// debug() {cerr << "\u001b[33m";}
ckt(!=) operator <<(c x) { cerr << boolalpha << x; rpt; }
ckt(==) operator <<(c x) { rpt << range(x.begin(), x.end()); }
tpc, class b> dop(pair<c,b> x) { rpt << "(" << x.first << ", " << x.second << ")"; }
tpc> dop(rge<c> r) { *this << "{";
for(auto it = r.b; it != r.e; ++it) *this << ", "+2*(it==r.b) << *it; rpt << "}";}
#else
tpc> dop(const c&) {rpt;}
#endif
};
#define imie(...) "[" << #__VA_ARGS__ ": " << (__VA_ARGS__) << "] "
//end debugger

#define exe_time(...) fprintf(stderr, __VA_ARGS__), fflush(stderr)

typedef double db;
typedef long double ld;
typedef string str;
typedef long long ll;
typedef unsigned long long ull;

typedef pair<int,int> pi;
typedef pair<ll,ll> pl;
typedef pair<double,double> pd;

#define F first
#define S second
#define endl '\n'
#define ALL(a) a.begin(),a.end()
#define pb push_back
#define LEN(x)  (int) x.size()
#define FOR(i, a, b)    for(int i = a; i < (int) b; i++)
#define FORN(i, a, b)   for(int i = a; i <= (int) b; i++)
#define FORB(i, a, b)   for(int i = b; i >= (int) a; i--)
#define REP(i, n)       FOR(i, 0, n)

typedef vector<int> vi;
typedef vector<ll> vl;
typedef vector<str> vs;

typedef vector<pi> vii;
typedef vector<pl> vll;

const int MOD = 1e9 + 7;
// const ll INF = 1e18;
const ld PI = acos((ld) - 1);

const int d4i[4]={-1, 0, 1, 0}, d4j[4]={0, 1, 0, -1};
const int d8i[8]={-1, -1, 0, 1, 1, 1, 0, -1}, d8j[8]={0, 1, 1, 1, 0, -1, -1, -1};

// ll rand(ll l, ll r) {
//     static mt19937
//     rng(chrono::steady_clock::now().time_since_epoch().count());
//     uniform_int_distribution<ll> ludo(l, r);
//     return ludo(rng);
// }

/*
NOTES:



*/

// void initial() {}

const int MAXN = 1000;
const int INF = 1e6;

int a[MAXN + 5][MAXN + 5], n, m;

int dpMin[MAXN + 5][MAXN + 5], dpMax[MAXN + 5][MAXN + 5];
int genMin(int i, int j) {
	if (i == 0 && j == 0)	return a[0][0];
	if (dpMin[i][j] != -INF)	return dpMin[i][j];

	int &ans = dpMin[i][j];
	if (i == 0) {
		ans = dpMin[i][j] = a[i][j] + genMin(i, j - 1);
	} else if (j == 0) {
		ans = dpMin[i][j] = a[i][j] + genMin(i - 1, j);
	} else {
		ans = dpMin[i][j] = a[i][j] + min(genMin(i - 1, j), genMin(i, j - 1));
	} return ans;
}
int genMax(int i, int j) {
	if (i == 0 && j == 0)	return a[0][0];
	if (dpMax[i][j] != -INF)	return dpMax[i][j];

	int &ans = dpMax[i][j];
	if (i == 0) {
		ans = dpMax[i][j] = a[i][j] + genMax(i, j - 1);
	} else if (j == 0) {
		ans = dpMax[i][j] = a[i][j] + genMax(i - 1, j);
	} else {
		ans = dpMax[i][j] = a[i][j] + max(genMax(i - 1, j), genMax(i, j - 1));
	} return ans;
}
void gen() {
	genMin(n - 1, m - 1);
	genMax(n - 1, m - 1);
}
void solve() {
	cin >> n >> m;
	REP(i, n)	REP(j, m) {
		cin >> a[i][j];
		dpMin[i][j] = dpMax[i][j] = -INF;
	}

	if ((n + m) % 2 == 0) {
		cout << "NO" << endl;
		return;
	}
	gen();

	// cout << "> " << dpMin[n - 1][m - 1] << ' ' << dpMax[n - 1][m - 1] << endl;
	if (dpMin[n - 1][m - 1] <= 0 && 0 <= dpMax[n - 1][m - 1]) {
		cout << "YES" << endl;
	} else {
		cout << "NO" << endl;
	}
}

signed main() {
    ios_base::sync_with_stdio(false);cin.tie(nullptr);
    // cout.tie(nullptr);

    #ifdef LOCAL
        // system("cls");
        // freopen("input.txt","r",stdin);
        // freopen("output.txt","w",stdout);
        // clock_t z = clock();
    #endif

    // initial();

    int t = 1;
    cin >> t;
    for(int tc = 1; tc <= t; tc++) {
        // cout << "Case #" << tc << ": " ;
        solve();
    }

    #ifdef LOCAL
        // exe_time("Total Time: %.3f\n", (double)(clock() - z) / CLOCKS_PER_SEC);
    #endif

    return 0;
}

 	     	 	  	  		 		 	 	 		 	 	