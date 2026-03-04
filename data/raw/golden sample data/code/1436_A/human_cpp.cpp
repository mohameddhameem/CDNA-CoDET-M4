#include<bits/stdc++.h>
#define ll long long
using namespace std;

ll arr[1000005];
int main(){
	ll t;
	cin >> t;
	
	while(t--){
		ll n,m,ans=0;
		cin >> n >> m;
		
		for(int i = 1; i <= n; i++){
			cin >> arr[i];
		}
		
		sort(arr+1,arr+n+1);
		
		for(int i = 1; i <= n; i++){
			ans += arr[i];
			}
			
		if(ans==m){
			cout << "YES" << endl;
		}else{
			cout << "NO" << endl;
		}
	}
}

	 		   									  	 	  	 			