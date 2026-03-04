#include <bits/stdc++.h>
using namespace std;
#define ll long long
 
ll x, batas;

bool check() {
    int b;
    for (ll i = 1; i < batas; i++) {
        b = cbrt(x - (i*i*i));
        if (b == ceil(cbrt(x - (i*i*i)))) return true;
    }
    return false;
}
 
int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
 
    int t;
    cin >> t;

    while (t--) {
        cin >> x;

        batas = ceil(cbrt(x));

        if (check()) cout << "YES\n";
        else cout << "NO\n";
    }
 
    return 0;
}
	 	    	  		  	 														