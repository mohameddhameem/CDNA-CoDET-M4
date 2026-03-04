#include<bits/stdc++.h>

using namespace std;

#define ll long long

const int mod = 1e9+7;

#define mp   make_pair

#define all(i)     i.begin() , i.end()

#define ft     first

#define sn     second

#define pb push_back

#define eps 1e-9

#define SP cout<<fixed<<setprecision(15);

#define mx 100005

//bool vis[100005];

//bool visited[100005];

vector<int>graph[mx];

//map<int,vector<bool>>grid;

int MX=1e6+1;



int main(){

    int t;cin>>t;

    while(t--){

        int n,u,v;cin>>n>>u>>v;

        int ar[n],cnt=0;

        for(int i=0;i<n;i++){cin>>ar[i]; if(i!=0&&(ar[i]==ar[i-1]))cnt++;}

        int ans=0;

        if(cnt==n-1){

            ans+=v;ans+=min(u,v);

            cout<<ans<<endl;continue;

        }

        int i=0;

    for( i=0;i<n-1;i++){



        if(abs(ar[i+1]-ar[i])>1){

           //cout<<ar[i]<<" ";

            ans=0;break;

        }

    }

 if(i==n-1)ans=min(u,v);



    cout<<ans<<endl;

    }

}