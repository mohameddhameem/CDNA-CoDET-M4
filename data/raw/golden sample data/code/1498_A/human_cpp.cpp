#include <bits/stdc++.h>

using namespace std;

const int MAXINT = 1e7;

int fr[MAXINT];

using ll = long long;

using ld = long double;

using vi = vector<int>;

using vl = vector<ll>;

using vs = vector<string>;

using deq =  deque<int> ;

using sti = stack <int>  ;

using stc = stack <char>  ;

using ld = long double;

#define all(x) begin(x), end(x)

// #define rall(vec) vec.rbegin(), vec.rend()

#define el "\n"

#define YES cout<<"YES\n";

#define NO cout<<"NO\n";

#define rep(i,a,b) for (int i = a; i < b; ++i)

#define loop(n) for(int i=1;i<=n;i++)

#define siz(arr) sizeof(arr)/sizeof(arr[0])

#define sz(x) int(x.size())

//#define pair<ll, ll> pl

#define Allah_yostork_t3deha ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);

ll  prf [MAXINT]  ;



void print(vector<int> &a){

    for(auto &i: a)

        cout << i << ' ';

    cout << '\n';

}

ld  intlog(ld base, ld x) {

    return (ld)(log(x) / log(base));

}

int mod = 100;

ll mul(ll A, ll B, ll mod){

    A %= mod;

    B %= mod;

    return (A * B) % mod;

}

int fpm(int base, int exp){

    int res = 1, x = base;

    while (exp){

        if(exp & 1)

            res = mul(res, x, mod);

        exp >>= 1;

        x = mul(x, x, mod);

    }

    return res;

}

long long fp(long long base, long long exp){

    if (exp == 0) return 1;

    long long ans = fp(base, exp / 2);

    ans = (ans * ans) ;

    if (exp % 2 != 0) ans = (ans * (base));

    return ans;

}

bool isprime(ll x){

    if(x==2) return 1;

    if(x==1 || x%2==0) return 0;

    for(int i=2;i<=sqrt(x);i++){

        if(x%i==0)

            return false;}

    return true;

}

vector<int> primeFactorization(int n){ // O(sqrt(n))

    vector<int>v;

    for(int i = 2; i * i <= n; i++){

        while (n % i == 0){

            n /= i;

            v.push_back(i);

        }

    }

    if(n > 1)

        v.push_back(n);

    return v;

}

ll gcd(ll a, ll b){

    if(b == 0) return a;

    return gcd(b, a % b);

}



ll lcm(ll a, ll b){

    return (a * b) / gcd(a, b);

}

vector<int> divisors(int n){ // O(sqrt(n))

    vector<int>v;

    for (int i = 1; i * i <= n; i++) {

        if (n % i == 0){

            v.push_back(i);

            if(n / i != i)

                v.push_back(n / i);

        }

    }

    return v;

}



const int  N = 1e6 + 5 ;

int d[N] ;

ll sumdiget (ll n){

    ll x= n %10 ;

    n/=10 ;

    while (n > 0) {

        x += n% 10 ;

        n /=10 ;

    }

    return x ;



}

ll greaters (ll n) {

    ll x= sumdiget(n) ;

    ll gc = gcd(n,x) ;

    return gc;



}





void Mt2olesh_Wrong_Argooook() {

ll n; cin >>n;

    while (greaters(n) <=1) n++;

    cout << n << el;

}



//cout<< (    (  cnt==0 )     ? 1 : cnt ) <<el;



int main(){

    Allah_yostork_t3deha

 //  freopen("GoBaby","r",stdin);



    int tc=1 ;   cin>>tc;

    while (tc--)

        Mt2olesh_Wrong_Argooook();

}

