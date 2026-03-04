#include<bits/stdc++.h>

using namespace std;

#define ll             long long int

#define chalo(i,a,b)    for(int i=a;i<b;i++)

#define all(x)         (x).begin(),(x).end()

#define srt(x)         sort(x.begin(),x.end())

#define pii            pair<int,int>

#define vec(v)         vector<int> v

#define ms(st)         multiset<int> st

#define set(s)         set<int> s

#define inmap(mp)      map<int,int> mp

#define chmap(mp)      map<char,int> mp



int main(){

    ios_base::sync_with_stdio(false);

    cin.tie(0);cout.tie(0);

    int xd,lmao;

    //the solution is written and directed by tony stark

    ll tt;

    cin>>tt;

    while(tt--){

        ll n;

        cin>>n;

        vec(v);

        chalo(i,0,n){

            int x;cin>>x;

            v.push_back(x);

        }

        int mn=*min_element(all(v));

        int cnt=0;

        chalo(i,0,n){

            if(v[i]>mn){

                cnt++;

            }

        }

        cout<<cnt<<'\n';

    }

    return 0;

}