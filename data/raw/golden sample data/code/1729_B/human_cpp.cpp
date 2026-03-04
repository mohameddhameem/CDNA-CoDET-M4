/****************### With The name of ALLAH, most Gracious, most Compassionate ###****************/



/**============================================================================**\

|   Author      : Md Jubayer Hossen Jame

|   Study       : CSE, BRUR

|   Today       : Thursday, 02 February, 2023

|   Source      : Codeforces Online Judge

|   Problem link:

|   Algorithm   :

\**============================================================================**/



#include "bits/stdc++.h"

using namespace std;



#define FI                    freopen("input.txt", "r", stdin)

#define FO                    freopen("output.txt", "w", stdout)

#define FasterIO              ios_base::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);



//#define int long long

typedef long long             ll;

//typedef __int128             lll;

typedef unsigned long long    ull;

typedef unsigned int          ui;



#define       mem(ara,val)    memset(ara, val, sizeof(ara));

#define                pii    pair<int,int>

#define                pll    pair<ll, ll>

#define                 mk    make_pair

#define                 ff    first

#define                 ss    second

#define                 pb    push_back

#define                ppb    pop_back

#define             all(x)    (x).begin(), (x).end()

#define            rall(x)    (x).rbegin(), (x).rend()

//#define       for0(i, n)    for (int i = 0; i < (int)(n); ++i)

#define         for0(i, n)    for(__typeof(n)i = 0; i < n; ++i)

//#define       for1(i, n)    for (int i = 1; i <= (int)(n); ++i)

#define         for1(i, n)    for(__typeof(n)i = 1; i <= n; ++i)

#define       forstl(i, n)    for(__typeof(n.begin()) i = n.begin(); i != n.end(); ++i)

#define             dbg(x)    cout << "line " << __LINE__ << " : " << #x << " :: " << x << "\n"

#define         dbg2(x, y)    cout << "line " << __LINE__ << " : " << #x << " :: " << x << " --> " << #y << " :: " << y << "\n"

#define               endl    '\n'



const double PI = acos(-1.0);



//Negative num?

#define gcd(a,b)              __gcd(a,b)

#define lcm(a,b)              (a*(b/gcd(a,b)))



template <typename T> string numTostr(T num)           { return to_string(num); }

long long str_num(string s)                            { long long num; istringstream iss(s); iss >> num; return num; }

const ll  INF =     1e18+2;

/**************************************************************************************************/



//min heap

//priority_queue<ll, vector<ll>, greater<ll>> qq;



/***************************** debug begin *****************************/

//https://codeforces.com/blog/entry/102631



string to_string(string s){return '"'+s+'"';}

string to_string(const char* s){return to_string((string)s);}

string to_string(const bool& b){return(b?"true":"false");}



template<class T>string to_string(T x){ostringstream sout;sout<<x;return sout.str();}

template<class A,class B>string to_string(pair<A,B> p){return "("+to_string(p.first)+", "+to_string(p.second)+")";}

template<class A>string to_string(const vector<A> v){

	int f=1;string res="{";for(const auto x:v){if(!f)res+= ", ";f=0;res+=to_string(x);}res+="}";

	return res;

}

void debug_out(){cout << endl; /*puts("");*/}

template<class T,class... U>void debug_out(const T& h,const U&... t){cout<<" "<<to_string(h);debug_out(t...);}

#ifdef tokitsukaze 

#define debug(...) cout<<"["<<#__VA_ARGS__<<"]:",debug_out(__VA_ARGS__);

#else

#define debug(...) 233;

#endif

/*****************************  debug end  *****************************/



void test_case()

{

	string s; int len; cin >> len >> s;



	for(int i = len - 1; i >= 0; --i)if(s[i] == '0')

	{

		int num = s[i-2] - '0';

		    num = num * 10 + s[i-1] - '0';



		s[i-2] = s[i-1] = s[i] = '*';

		s[i-2] = (num + 'a' - 1);

	}



	for0(i, s.size())if(s[i] > '0' and s[i] <= '9')

	{

		s[i] = (s[i] - '0') + 'a' - 1;

	}

	for0(i, len)if(s[i] != '*')cout << s[i];

	cout << endl;

}



//★゜・。。・゜゜・。。・゜☆゜・。。・゜゜・。。・゜★゜・。。・゜゜・。。・゜☆゜・。。・゜゜・。。・゜★゜・。。・゜゜・。。・゜☆



int main()

{



#ifndef ONLINE_JUDGE

    //FI;

    //FO;

#endif // ONLINE_JUDGE

    FasterIO

    cout.precision(15); cout << fixed; //or

    //cout << setprecision(15) << fixed;



    int T = 1;

    cin >> T;

    //cin.ignore(100, '\n'); //for '\n'

    //while( T --> 0 )

    for(int tc = 1; tc <= T; ++tc)

    {

        //cout << "Case #" << tc << ":" << endl;

        test_case();

    }

    return 0;

}