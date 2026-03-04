#include <bits/stdc++.h>

using namespace std;

#define f first

#define s second

#define ll long long





int main(){

 

    ios_base :: sync_with_stdio(false);

    cin.tie(NULL);

    #ifndef ONLINE_JUDGE

     freopen("input.txt", "r", stdin); 

    freopen("output.txt", "w", stdout);

    #endif

string s; cin >> s;

vector <ll> A;

for(int i=0;i<s.length();i++) A.push_back(s[i]-'A'); 

bool ans = true;

int n = A.size();

for(int i=0;i<n-2;i++){

    if( (A[i+1]+A[i])%26!=A[i+2] ) {ans = false; break;}

}

if(ans) cout << "YES" << endl;

else cout << "NO" << endl;

}







// 0 1 1 2 3 5 8 13 21 35 56 



// A B B 