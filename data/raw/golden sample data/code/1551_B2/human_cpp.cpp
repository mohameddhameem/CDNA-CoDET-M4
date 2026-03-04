// Created by KHALED MOSHARRAF PARVEZ

#include <bits/stdc++.h>



#define whatis1(x) cout<<#x<<"="<<x<<'\n'

#define whatis2(x,y) cout<<#x<<"="<<x<<","<<#y<<"="<<y<<'\n'

#define whatis3(x,y,z) cout<<#x<<"="<<x<<","<<#y<<"="<<y<<","<<#z<<"="<<z<<'\n'

#define whatisArray(arr,n) cout<<#arr<<", Size: "<<n<<'\n';for(int m=0;m<n;m++)  cout<<arr[m]<<" ";cout<<'\n';

#define whatArray2D(arr,m,n) cout<<#arr<<", Size: "<<m<<","<<n<<'\n';for(int mm=0;mm<m;mm++){for(int nn=0;nn<n;nn++)cout<<arr[mm][nn]<<" ";cout<<'\n';} 

#define caseprint(caseno,ans) cout<<"Case "<<caseno<<": "<<ans<<'\n'

#define endl '\n'

#define all(v) v.begin(),v.end()

#define mod 1000000007

#define MX 200005



using namespace std;

using ll = long long;



void solve(int caseno){

    int n,k;

    cin>>n>>k;

    vector<int> v(n);

    int cnt[n+1]={0};

    for(int i=0;i<n;i++) {

        cin>>v[i];

        cnt[v[i]]++;

    }

   // whatis1("OK");

    vector<pair<int,int> > pii;

    for(int i=1; i<=n; i++) {

        cnt[i]=min(k,cnt[i]);

        if(cnt[i]>0){

            pii.emplace_back(cnt[i],i);

        }

    }

    //sort(pii.begin(),pii.end());

    int srt[n+1];

    int nw = 0;

    int sum= 0;

    for(int i=pii.size()-1; i>=0; i--) {

        int x = pii[i].first;

        int y = pii[i].second;

        sum+=x;

        srt[y]=nw;

        nw+=x;

        nw%=k;

    }   

    sum/=k;

  //  whatis1(sum);

    int used[n+1]={0};

    int color[n] ={0};

    for(int i=0; i<n; i++) {

        if(used[v[i]]>=k||color[srt[v[i]]]>=sum)

            cout<<0<<' ';

        else

            {color[srt[v[i]]]++;cout<<srt[v[i]]+1<<' ';}

        used[v[i]]++;

        srt[v[i]]=(srt[v[i]]+1)%k;

    }



    cout<<'\n';

    return;

    

}

int main()

{

    ios::sync_with_stdio(0);

    int cases,caseno=0;

    cin>>cases;

    while(cases--){

        solve(++caseno);

    }

    return 0;

}