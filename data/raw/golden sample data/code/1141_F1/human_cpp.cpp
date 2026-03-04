#include<iostream>

#include<vector>

#include<cmath>

#include<map>

#include<unordered_map>

#include<unordered_set>

#include<set>

#include<string>

#include<stack>

#include<algorithm>

#include<bitset>

#include<queue>

#include<iomanip>

#include<cstring>

#include <numeric>

#include <functional>

using namespace std;

typedef long long ll;

const ll mod=998244353;

typedef pair<int,int> pi;



vector<pi> f(int val,vector<int>&v,int n){

    unordered_map<int,int>p;

    int sum=0;

    p[0]=-1;

    vector<int>dp(n);

    vector<pi>ans;

    vector<int>a(n,-1);

    for(int i=0;i<n;i++){

        sum+=v[i];

        if(p.count(sum-val)){



            if(i==0||(p[sum-val]==-1&&dp[i-1]==0)) ans.push_back(pi(0,i));

            else if(p[sum-val]!=-1&&dp[p[sum-val]]+1>dp[i-1]) ans.push_back(pi(p[sum-val]+1,i));

            if(p[sum-val]==-1) {

                if(i==0) dp[i]=1;

                else dp[i]=max(dp[i-1],1);

            }else{

                dp[i]=max(dp[i-1],dp[p[sum-val]]+1);

            }

        }else dp[i]=(i==0?0:dp[i-1]);

        p[sum]=i;

    }

    return ans;

}

void solve(){

    int n;

    cin>>n;

    vector<int>v(n);

    for(int i=0;i<n;i++) cin>>v[i];

    map<int,int>p;

    for(int i=0;i<n;i++){

        int sum=0;

        for(int j=i;j<n;j++){

            sum+=v[j];

            p[sum]++;

        }

    }

    int res=0;

    vector<pi>ans;

    for(auto i:p){

        vector<pi>k=f(i.first,v,n);

        if(k.size()>res){

            res=k.size();

            ans=k;

        }

    }

    cout<<res<<endl;

    for(auto i:ans){

        cout<<i.first+1<<" "<<i.second+1<<endl;

    }

}



int main(){

    ios::sync_with_stdio(false);

    cin.tie(nullptr);

    cout.tie(nullptr);

    solve();

    system("pause");

}