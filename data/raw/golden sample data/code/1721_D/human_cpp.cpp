#include <iostream>

#include <vector>

#include <set>

#include <unordered_map>

#include <unordered_set>

#include <map>

#include <stack>

#include <queue>

#include <deque>

#include <algorithm>

#include <numeric>

#include <math.h>

#include <cstring>

#include <time.h>

#include <chrono>

#include <random>

#include <ctime>

#include <iomanip>

using namespace std;

#define gc getchar_unlocked

#define fo(i, n) for (i = 0; i < n; i++)

#define Fo(i, k, n) for (i = k; k < n ? i < n : i > n; k < n ? i += 1 : i -= 1)

#define ll long long

#define deb(x) cout << #x << "=" << x << endl

#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl

#define pb push_back

#define mp make_pair

#define ff second

#define ss first

#define all(x) x.begin(), x.end()

#define clr(x) memset(x, 0, sizeof(x))

#define sortall(x) sort(all(x))

#define tr(it, arr) for (auto it = arr.begin(); it != arr.end(); it++)

#define PI 3.1415926535897932384626

typedef pair<int, int> pii;

typedef pair<ll, ll> pl;

typedef vector<int> vi;

typedef vector<ll> vl;

typedef vector<pii> vpii;

typedef vector<pl> vpl;

typedef vector<vi> vvi;

typedef vector<vl> vvl;

const double pi = 3.141592653589793238;

const ll mod = 1e9 + 7;

bool comp2(pair<ll, ll> &arr, pair<ll, ll> &b)

{

    if (arr.first == b.first)

        return arr.second < b.second;

    return arr.first < b.first;

};



template <typename T>

void read(T i, T n, vector<T> &arr)

{

    for (T j = i; j < n; j++)

        cin >> arr[j];

}



string decimalToBinary(ll n)

{

    string s = "";

    for (int i = 30; i >= 0; i--)

    {

        if ((n >> i) & 1)

            s += '1';

        else

            s += '0';

    }

    return s;

}



void solve()

{

    ll n;

    cin >> n;

    vl a(n);

    vl b(n);

    for (auto &i : a)

        cin >> i;

    for (auto &i : b)

        cin >> i;

    

    ll ans = 0;

    auto bitOrNot = [&] (ll num) {

        vl temp1, temp2;

        for(auto &i : a) temp1.push_back(i&num);

        for(auto &i : b) temp2.push_back(~i&num);

        sort(all(temp1));

        sort(all(temp2));

        if(temp1==temp2) return true;

        else return false;

    };



    for(int i = 29; i>=0; i--) {

        if(bitOrNot(ans | (1 << i))) ans += (1<<i);

    }

    cout<<ans<<endl;



}



int main()

{

    ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);

    srand(chrono::high_resolution_clock::now().time_since_epoch().count());

    int t = 1;

    cin >> t;

    while (t--)

    {

        solve();

    }

    return 0;

}

