#include<bits\stdc++.h>

using namespace std;

#define FAST std::ios_base::sync_with_stdio(0);cin.tie(NULL);cout.tie(NULL);

#define all(c) (c).begin(),(c).end()

const long long N = 2e5+5, M =1e9+7, OO = 0x3f3f3f3, mod=1e7+7;

#define Count(s,a) (std::count((s).begin(), (s).end(),(a)))

int dx[8]={-1,-1,-1,0,1,1,1,0};

int dy[8]={-1,0,1,1,1,0,-1,-1};

#define ll long long

#define stl(i,s) for(auto i : s)



ll n,c,d,l,r,mid,sum,days;  ll arr[N];

bool valid(ll x){

    sum=0,days=d;

    int i=0;

    while(days>0){

        days--;

        if(i<=(n-1)){

            sum+=arr[i];

        }

        if(i==x) {

            i=-1;

        }

        i++;

    }

    return sum>=c;

}





void TestCase(){

    cin >> n >> c >> d;

    sum=0;

    for (int i = 0; i < n; ++i) {

        cin >> arr[i];

    }

    sort(arr,arr+n,greater<>());

    for (int i = 0; i < n; ++i) {

        if(i<d)sum+=arr[i];

        else break;

    }

    if(sum>=c){cout << "Infinity\n";

        return;}

    else if(arr[0]*d<c){cout << "Impossible\n";

        return;}

    l=0;    r=1e12;

    while(l+1<r){

        mid = (l+r)/2;

        if(valid(mid)){

            l=mid;

        }else{

            r=mid;

        }

    }

    cout << l<< endl;

}

int main(){

    FAST

    int t;  cin >> t;

    while(t--){

        TestCase();

    }

}