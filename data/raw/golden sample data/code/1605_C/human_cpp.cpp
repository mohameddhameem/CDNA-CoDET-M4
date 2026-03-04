#include<bits/stdc++.h>

using namespace std;

#define IOS ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL)

#define ll long long

#define rep(i,n) for (ll i = 0; i < n; i++)

#define frev(i, x, y) for (ll i = x - 1; i >= y; i--)

#define feach(x, y) for (auto &x : y)

#define fsq(i, x, y) for (ll i = x; i*i<=y; i++)

#define vl vector<ll>

#define vp vector<pair<ll,ll>>

#define all(v) v.begin(), v.end()

#define backk(v) v.rbegin(), v.rend()

#define pb push_back

#define pf push_front

#define fr first

#define sc second

#define maxv(v) *max_element(v.begin(), v.end())

#define minv(v) *min_element(v.begin(), v.end())

#define lv 10000000000000000

void solve(){

   int n;

   cin>>n;

   int cnt_of_a=0;

  int prefix_a[n+1];

     int prefix_b[n+1];

       int prefix_c[n+1];

  char arr[n+2];

  for(int i=1;i<=n;i++){

  cin>>arr[i];

  }

   prefix_a[0]=0;

   prefix_b[0]=0;

   prefix_c[0]=0;

   for(int i=1;i<=n;i++){

   if(arr[i]=='a'){

    cnt_of_a++;

      prefix_a[i]=prefix_a[i-1]+1;

     prefix_b[i]=prefix_b[i-1];

     prefix_c[i]=prefix_c[i-1];

   }

   if(arr[i]=='b'){

    prefix_a[i]=prefix_a[i-1];

     prefix_b[i]=prefix_b[i-1]+1;

     prefix_c[i]=prefix_c[i-1];

   }

   if(arr[i]=='c'){

    

      prefix_a[i]=prefix_a[i-1];

     prefix_b[i]=prefix_b[i-1];

     prefix_c[i]=prefix_c[i-1]+1;

   }

   }





if(cnt_of_a<=1){

  cout<<"-1"<<"\n";

  return;

}

vector<int>vec;

for(int i=1;i<=n;i++){

 if(arr[i]=='a'){

  vec.pb(i);

 }

}

for(int i=1;i<=n-1;i++){

  if(arr[i]=='a'&&arr[i+1]=='a'){

    cout<<"2"<<"\n";

    return;

  }

}

for(int i=0;i<vec.size()-1;i++){

 if(vec[i+1]-vec[i]==2){

  cout<<"3"<<"\n";

  return;

 }

}

vector<int>ans;

int i=0;

for( i=0;i<vec.size()-2;i++){

   if((prefix_b[vec[i+1]]-prefix_b[vec[i]]<2)&&(prefix_c[vec[i+1]]-prefix_c[vec[i]]<2)){

   ans.pb(vec[i+1]-vec[i]+1);

   }

    if((prefix_b[vec[i+2]]-prefix_b[vec[i]]<3)&&(prefix_c[vec[i+2]]-prefix_c[vec[i]]<3)){

   ans.pb(vec[i+2]-vec[i]+1);

   }





}

 if((prefix_b[vec[i+1]]-prefix_b[vec[i]]<2)&&(prefix_c[vec[i+1]]-prefix_c[vec[i]]<2)){

   ans.pb(vec[i+1]-vec[i]+1);

   }

   if(ans.size()!=0)

   cout<<*min_element(ans.begin(),ans.end())<<"\n";

   else{

    cout<<"-1"<<endl;

   }







    }



int main(){

 ios_base::sync_with_stdio(false);

   cin.tie(NULL);

int t;

cin>>t;

while(t--){

solve();

}

return 0;

}