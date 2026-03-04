#include<bits/stdc++.h>

using namespace std;

typedef long long lli;

typedef vector<int>vi;

typedef vector<long long>vll;

typedef vector<char>vc;

typedef vector<string>vs;

#define endl "\n"

#define pb push_back

#define all(x) x.begin(),x.end()	

#define mp(a,b) make_pair(a,b)

typedef vector<bool>vb;

int gcd(int a, int b)

{

    if (b == 0)

        return a;

    return gcd(b, a % b);

}



lli power(lli a,lli b){

	lli ans=1;

	while(b!=0){

		if(b&1)ans*=a;

		a*=a;

		b=b>>1;

	}

	//this works because of every number can be written as the sum of powers of 2

	return ans;

}





// int check_prime(int n){

// 	if(n==1)return 0;

// 	else if(n<=3)return 1;

// 	else if(n%2==0 || n%3==0)return 0;

// 	for(int i=5;i*i<=(n);i+=6){

// 		if(n%i==0 || n%(i+2)==0)return 0;

// 	}

// 	return 1;

// }





// void seive(int n){

// 	vector<bool>is_prime(n+1,true);

// 	for(int i=2;i*i<=n;i++){

// 		if(is_prime(i)){

// 			for(int j=i*i;j<=n;j+=i){

// 				is_prime[j]=false;

// 			}

// 		}

// 	}

// 	for(int i=2;i<=n;++i){

// 		if(is_prime[i])cout<<i<<" ";

// 	}

// }



// int min(int a,int b){

// 	if(a<b)return a;

// 	return b;

// }

// int max(int a,int b){

// 	if(a>b)return a;

// 	return b;

// }

// int min(int a,int b){

// 	if(a<b)return a;

// 	return b;

// }

lli min(lli a,lli b)

{

	if(a<b)return a;

	return b;

}

#define const 1000000007



lli max(lli a,lli b){

	if(a>b)return a;

	return b;

}

bool is_pal(string &s){

	int i = 0,j = s.length() - 1;

	while(i < j){

		if(s[i] != s[j])return false;

		++i;--j;

	}

	return true;

}

void solve(){

	int n;

	cin>>n;

	string s;

	cin>>s;

	if(s[0] == '0' || s[s.length()-1]=='0'){

		cout<<"NO"<<endl;return;

	}

	int count = 0;

	for(auto &x:s){

		if(x == '1')count++;

	}

	if(count&1){

		cout<<"NO"<<endl;return;

	}

	int c = 0;

	bool is_open = true;

	string a = "";

	string b = "";

	for(int i = 0;i<n;++i){

		if(s[i] == '0'){

			if(is_open){a+='(';b+=')';}

			else{ a+=')';b+='(';}

			is_open=!is_open;

		}

		else{

			if(c < count/2){

				a+='(';b+='(';

			}

			else{

				a+=')';b+=')';

			}

			++c;

		}

	}

	cout<<"YES"<<endl;

	cout<<a<<endl<<b<<endl;

}





int main(){

	ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);

	int t;

	cin>>t;

	while(t--){

		solve();

	}

	return 0;

}



// int main(){

// 	ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);

// 	solve();

// 	return 0;

// }

