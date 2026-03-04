//submission by ffaith

/*

** author: ffaith

** created: i dont remember

** [solved by] : also ffaith

*/

#include <bits/stdc++.h>

#define ld                  long double

#define ins                 insert

#define ull                 unsigned long long

#define pb                  push_back

#define endl                "\n"

#define ll                  long long

#define lp(a,b)             for(int i = a; i<b; i++)

#define deflp(a,b)          for(i = a; i<b; i++)

#define pl(a,b)             for(int i = a; i>=b; i--)

#define pj(a,b)             for(int j = a; j>=b; j--)

#define jp(a,b)             for(int j = a; j<b; j++)

#define kp(a,b)             for(int k = a; k<b; k++)

#define funcget(a,b);       for(int i = 0;i<b;i++){ cin>>a[i];}

#define sumfuncget(a,b,c);  for(int i = 0;i<b;i++){ cin>>a[i];c+=a[i];}

#define fsort(a)            sort(a.begin(), a.end())

#define fsort2(a)           sort(a.begin(), a.end(), greater <int>())

#define fcnt(a,b)           count(a.begin(), a.end(), b)

#define setget(s,n)         lp(0,n){ll tmp; cin >> tmp;s.ins(tmp);}

#define vectup();           ll n; cin >> n;vector <ll> fn(n);funcget(fn,n);

#define sumvectup();        ll n,sum = 0; cin >> n;vector <ll> fn(n);sumfuncget(fn,n,sum);

#define alphab()            "abcdefghijklmnopqrstuvwxyz"

#define f_it                vector<ll>::iterator 

#define pq(fn)              priority_queue <ll> fn;

#define pq1(fn)             priority_queue <ll, vector<ll>, greater<ll> > fn;

#define fprint(a);          for(ll i:a){cout<<i<<' ';}cout<<endl;

using namespace std;

/*

to solve{

    926A

    1646C

    1786C

    1560C

    489C

    1490A

    1373A

    1422B

    550A

    1611B

    75A

}

*/

const ll MODULO = 1e9 + 7;

const ll rll18 = (ll) 1e18;

const ll rll17 = (ll) 1e17;

const ll rll16 = (ll) 1e16;

const ll rll15 = (ll) 1e15;

const ll rll14 = (ll) 1e14;

const ll rll13 = (ll) 1e13;

const ll rll12 = (ll) 1e12;

const ll rll11 = (ll) 1e11;

const ll rll10 = (ll) 1e10;

const ll rll9 = (ll) 1e9;

const ll rll8 = (ll) 1e8;

const ll rll7 = (ll) 1e7;

const ll rll6 = (ll) 1e6;

const ll rll5 = (ll) 1e5;

const ll rll4 = (ll) 1e4;

ll comb(ll n){

    return n*(n-1)/2;

}

void yesno(bool flag){

    if(flag){

        cout<<"YES"<<endl;

    }

    else{

        cout<<"NO"<<endl;

    }

}

ll fsumv(vector<ll> a){

    ll cnt = 0;

    lp(0,a.size()){

        cnt+=a[i];

    }

    return cnt;

}

ll digsum(ll a){

    ll sum = 0;

    while(a!=0){

        sum+=a%10;a/=10;

    }

    return sum;

}

bool powerof2(ll n){

    if (n == 0)

        return 0;

    while (n != 1) {

        if (n % 2 != 0)

            return 0;

        n = n / 2;

    }

    return 1;

}

ll gcd(ll m, ll n) {

   ll r = 0, a, b;

   a = (m > n) ? m : n;

   b = (m < n) ? m : n;

   r = b;

   while (a % b != 0) {

      r = a % b;

      a = b;

      b = r;

   }

   return r;

}

ll lcm(ll m, ll n) {

   ll a;

   a = (m > n) ? m: n;

   while (true) {

      if (a % m == 0 && a % n == 0)

         return a;

         a++;

   }

}

bool isPrime(int n){

    if (n <= 1)

        return false;

    else if (n==2){

        return true;

    }

    for (int i = 2; i < n; i++)

        if (n % i == 0)

            return false;

 

    return true;

}

vector <ll> divisors(ll n){

    vector <ll> fn;

    lp(1,n){

        if(n%i==0){

            fn.pb(i);

        }

    }

    return fn;

}

vector <ll> SieveOfEratosthenes(ll n){

    bool prime[n + 1];

    memset(prime, true, sizeof(prime));

    vector <ll> primes;

 

    for (int p = 2; p * p <= n; p++) {

        if (prime[p] == true) {

            for (int i = p * p; i <= n; i += p)

                prime[i] = false;

        }

    }

 

    for (int p = 2; p <= n; p++)

        if (prime[p])

            primes.pb(p);

    return primes;

}

string ftolow(string a){

    lp(0,a.size()){

        a[i] = towlower(a[i]);

    }

    return a;

}

string ftoup(string a){

    lp(0,a.size()){

        a[i] = towupper(a[i]);

    }

    return a;

}

string casefold(string s){

    lp(0,s.size()){

        if (s[i] == toupper(s[i])){

            s[i] = tolower(s[i]);

        }

        else{

            s[i] = toupper(s[i]);

        }

    }

    string d = s;

    return d;

}

ll fib(ll n){

    vector <ll> dp(n+1);

    dp[0] = 0;dp[1] = 1;

    lp(2,n+1){

        dp[i] = dp[i-2] + dp[i-1];

    }

    return dp[n];

}

ll fibsum(ll n){

    vector <ll> dp(n+1);ll sum = 0;

    dp[0] = 0;dp[1] = 1;

    lp(2,n+1){

        dp[i] = dp[i-2] + dp[i-1];

    }

    lp(0,n){

        sum+=dp[i];

    }

    return sum;

}

bool isInteger(double N)

{

 

    ll X = N;

 

    double fuuuuc = N - X;

 

    if (fuuuuc > 0) {

        return false;

    }

    return true;

}

bool nozeroes(string s){

    if(count(s.begin(), s.end(), '0') == 0){

        return true;

    }

    return false;

}

bool no1s(string s){

    if(count(s.begin(), s.end(), '1') == 0){

        return true;

    }

    return false;

}

string removal(string s){

    lp(1,s.size()){

        if((s[i] == '0' && s[i-1] == '1') || (s[i] == '1' && s[i-1] == '0')){

            s.erase(s.begin()+i);s.erase(s.begin()+(i-1));

        }

    }

    return s;

}

vector <ll> merge(vector <ll> &a, vector <ll> &b){

    ll x = 0,y = 0,k = 0;vector <ll> c(a.size() + b.size());

    while(x<a.size() || y<b.size()){

        if((y == b.size() )|| (x < a.size() && (a[x] < b[y]))){

            c[k++] = a[x++];

        }

        else{

            c[k++] = b[y++];

        }

    }

    return c;

}

ll findget(vector <ll> v, ll isk){

    auto it = find(v.begin(), v.end(), isk);

    if(it!=v.end()){

        return it - v.begin();

    }

    else{

        return -1;

    }

}

ll power(ll n, ll p){

    ll z = n;

    lp(0,p-1){

        n = n * z;

    }

    return n;

}

ll distinctness(vector <ll> &a){

    set <ll> s;

    lp(0,a.size()){

        s.ins(a[i]);

    }

    return s.size();

}

bool isdistinct(vector <ll> &a){

    set <ll> s;

    lp(0,a.size()){

        s.ins(a[i]);

    }

    if(s.size() == a.size()){

        return true;

    }

    else{

        return false;

    }

}

vector <ll> permutation(ll n){

    vector <ll> a(n);

    lp(0,n){

        a[i] = i+1;

    }

    return a;

}

ll evenoddsum(vector <ll>&a,string z){

    ll cnt0 = 0,cnt1 = 0;

    lp(0,a.size()){

        if(a[i]%2){

            cnt1++;

        }

        else{

            cnt0++;

        }

    }

    if(ftolow(z) == "even"){

        return cnt0;

    }

    else{

        return cnt1;

    }

}

ll factorial(ll n){

    ll z = 1;

    lp(1,n+1){z = (z*i);}

    return z;

}

pair <ll, ll> maxsegment(vector <ll>&fn,ll n){

    ll mx = -200,mxend = 0,start = 0, end = 0, s = 0;

    lp(0,n){

        mxend+=fn[i];

        if(mxend > mx){

            mx = mxend;

            start = s; end = i;

        }

        if(mxend < 0){

            mxend = 0;s = i + 1;

        }

    }

    pair <ll, ll> z;

    z.first = start;z.second = end;

    return z;

}

ll secondmin(vector <ll>&a){

    fsort(a);return a[1];

}

ll secondmax (vector<ll>&a){

    fsort2(a);return a[1];

}

// fn[n-1] is faster than *max_element(fn.begin(), fn.end())

bool intdistinct(ll a){

    vector <ll> aa;

    while(a!=0){

        aa.pb(a%10);a/=10;

    }

    if(isdistinct(aa)){

        return true;

    }

    return false;

}

vector<ll> findPrefixSums(vector<ll>& a) {

    ll n = a.size();

    vector<ll> prefixSums(n + 1, 0);

    for (ll i = 0; i < n; i++) {

        prefixSums[i + 1] = prefixSums[i] + a[i];

    }

    return prefixSums;

}

// 6 8 2 8 h = 2

ll ugliness(vector <ll>&a){

    fsort(a);return abs(a[0] - a[a.size() - 1]);

}

bool palindrome(string s){

    string prev = s;

    reverse(s.begin(), s.end());

    if(prev == s){

        return true;

    }return false;

}

ll Max(vector <ll>&a){

    vector <ll> v = a;

    fsort2(v);return v[0];

}

ll Min(vector <ll>&a){

    vector <ll> v = a;

    fsort(v);return v[0];

}

ll doublefactorial(ll n){

    for(ll i = 1;i<n;i+=2){

        n*=i;n%=MODULO;

    }

    return n%MODULO;

}



/*

1 2

3 4

5 6

00 00

01 10

02 20

03 30

04 40

05 50

10 01

11 11

12 21

13 31

14 41

15 51

20 02

21 12

22 22

23 32



	 	  							 		 			   	  			  	

*/

vector <string> strsegments(string s){

    vector <string> f;

    map <string ,bool> used;

    ll n = s.size();

    lp(0,n){

        jp(1,n+1){

            string temp = s.substr(i,j);

            if(!used[temp]){

                f.pb(temp);used[temp] = true;

            }

        }

    }

    return f;

}

ll vectorxor(vector <ll>&a){

    ll z = 0;

    lp(0,a.size()){

        z = z^a[i];

    }

    return z;

}

bool square(ll n){

    ll z = sqrt(n);

    if(z*z == n){

        return true;

    }return false;

}

bool equal(vector <ll> a){

    set <ll> s;

    lp(0,a.size()){

        s.ins(a[i]);

    }

    if(s.size() == 1){

        return true;

    }return false;

}

ll dignum(ll a){

    ll cnt = 0;

    while(a!=0){

        a/=10;cnt++;

    }

    return cnt;

}

ll power10(ll a){

    return pow(10,a);

}

ll allneg(vector <ll> a){

    bool flag = true;

    lp(0,a.size()){

        if(a[i] > 0){

            flag = false;break;

        }

    }

    if(flag){

        return true;

    }

    return false;

}

ll takes(ll a, ll m){

    ll cnt = 0;

    while(a>0){

        a-=m;cnt++;

    }

    return cnt;

}

string strlcm(string x,string y){

    string z = x,d = y;

    while(z.size()!=d.size()){

        if(z.size() < d.size()){

            z+=x;

        }

        else{

            d+=y;

        }

    }

    if(z == d){

        return z;

    }

    return "-1";

}

vector <ll> sieve2(ll z){

    vector <ll> a(z+1,0);

    for(ll i=2; i<=z; i++)

    {

        if(a[i]==0)

        {

           for(ll j=2; i*j<=z; j++)

           {

               a[i*j]=1;

           }

        }

    }

    return a;

}

bool isin(vector <ll> a,ll n){

        bool z = false;ll l = 0,r = a.size()-1, mid;

        while(r>=l){

            mid = (l+r)/2;if(a[mid]==n){z = true;break;}

            else{

                if(a[mid]<n){

                    l = mid+1;

                }

                else{

                    r = mid-1;

                }

            }

        }

        if(z){

            return true;

        }

        return false;

}

/*

5 2 5

*/



string binary(ll n)

{

    string s = "";

    for (ll i = n-1; i >= 0; i--) {

        ll k = n >> i;

        if (k & 1)

            s+="1";

        else

            s+="0";

    }

    ll z = 0;

    lp(0,s.size()){

        if(s[i] == '1'){

            z = i;break;

        }

    }

    string f = "";

    lp(z,s.size()){

        f+=s[i];

    }

    return f;

}

bool substr(string s,string c){

    ll i = -1;

    for(char&z : s){

        i = s.find(z, i+1);

        if(i == string::npos){

            return false;

        }

    }return true;

}

bool nondeg(ll a, ll b, ll c){

    if(a >= b+c || b >= a+c || c >= b+a){

        return false;

    }return true;

}

double cnr(ll n, ll r){

    double z = factorial(n);

    double y = factorial(n - r) * factorial(r);

    return z / y;

}

vector <ll> dpp(rll4,-1);

ll intsize(ll n){

    n = abs(n);

    if(n < 10){

        return 1;

    }

    if(n < 100){

        return 2;

    }

    if(n < 1000){

        return 3;

    }

    if(n < 10000){

        return 4;

    }

    if(n < 100000){

        return 5;

    }

    if(n < 1000000){

        return 6;

    }

    if(n < 10000000){

        return 7;

    }

    if(n < 100000000){

        return 8;

    }

    if(n < 1000000000){

        return 9;

    }

    if(n < 10000000000){

        return 10;

    }

    if(n < 100000000000){

        return 11;

    }

    if(n < 1000000000000){

        return 12;

    }

    if(n < 10000000000000){

        return 13;

    }

    if(n < 100000000000000){

        return 14;

    }

    if(n < 1000000000000000){

        return 15;

    }

    if(n < 10000000000000000){

        return 16;

    }

    if(n < 100000000000000000){

        return 17;

    }

    return 18;

}

vector <ll> gr[5001];map <ll, bool> marked;vector <ll> topsort;

void dfs(ll v){

    marked[v] = true;

    for(ll to:gr[v]){

        if(!marked[to]){

            dfs(to);

        }

    }

    topsort.pb(v);

}

void thepersonwhoaskedstillhasntbeenfoundyethowdepressingperhapsthatsthereasonigotdepression(){

    ll n;string s;cin >> n >> s;bool flag = false;string a,b;

    lp(0,n){

        if(s[i] == '2'){

            if(flag){

                a+='2';b+='0';

            }

            else{

                a+='1';b+='1';

            }

        }

        else if(s[i] == '1'){

            if(flag){

                a+='1';b+='0';

            }

            else{

                a+='0';b+='1';flag = true;

            }

        }

        else{

            a+='0';b+='0';

        }

    }

    cout<<a<<endl<<b<<endl;

}



int main()

{

    

    ios_base::sync_with_stdio(0);cin.tie(NULL);cout.tie(NULL);ll idontthinkicancuremydepressioninanywaywellmaybefindingthepersonwhoasked;

    cin >> idontthinkicancuremydepressioninanywaywellmaybefindingthepersonwhoasked;

    //idontthinkicancuremydepressioninanywaywellmaybefindingthepersonwhoasked = 1;

    while(idontthinkicancuremydepressioninanywaywellmaybefindingthepersonwhoasked--){

        thepersonwhoaskedstillhasntbeenfoundyethowdepressingperhapsthatsthereasonigotdepression();

    }

    return 0;

}