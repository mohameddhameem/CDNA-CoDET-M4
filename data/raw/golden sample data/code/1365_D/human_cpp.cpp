#include <bits/stdc++.h>

#define ll long long

#define vi vector<long long>

#define vb vector<bool>

#define si set<long long>

#define ss set<string>

#define vs vector<string>

#define msi map<string, long long>

#define mci map<char, long long>

#define mss map<string, string>

#define msb map<string, bool>

#define REP(i,a,b) for (ll i = a; i < b; i++)

using namespace std;

#define LSEQ(n, seq) for (int i = 0; i < n; i++) cin >> seq[i]

#define PSEQ(n, seq, sep, end) for(int i = 0; i < n; i++) cout << seq[i] << sep; cout << end;

#define LVEQ(n, veq) for(int i = 0; i < n; i++) {ll num; cin >> num; veq.push_back(num);}

#define PVEQ(n, veq, sep, end) for(int i = 0; i < n; i++) cout << veq[i] << sep; cout << end;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());





void solve(){

    string ans = "Yes";

    ll n, m; cin >> n >> m;



    char tab[n][m];

    bool viz[n][m];

    memset(viz,false,sizeof viz);

    queue<pair<ll,ll>> fila;

    queue<pair<ll,ll>> fila2;

    ll goods = 0;

    REP(i,0,n){

        REP(j,0,m){

            cin >> tab[i][j];

            if(tab[i][j] == 'B') fila.push({i,j});

            if(tab[i][j] == 'G') goods++;

        }

    }



    while(!fila.empty()){

        auto prox = fila.front();

        fila.pop();

        if(prox.first == n || prox.first < 0 || prox.second == m || prox.second < 0) continue;

        if(tab[prox.first][prox.second] == '#') continue;

        

        if(tab[prox.first][prox.second] == '.'){

            tab[prox.first][prox.second] = '#';

            continue;

        }

        if(viz[prox.first][prox.second]) continue;

        viz[prox.first][prox.second] = true;



        if(prox.first == n - 1 && prox.second == m - 1){

            ans = "No";

            continue;

        }



        fila.push({prox.first + 1, prox.second});

        fila.push({prox.first - 1, prox.second});

        fila.push({prox.first, prox.second + 1});

        fila.push({prox.first, prox.second - 1});

    }



    fila.push({n - 1,m - 1});



    memset(viz,false,sizeof viz);

    ll finded = 0;

    while(!fila.empty()){

        auto prox = fila.front();

        fila.pop();



        if(prox.first == n || prox.first < 0 || prox.second == m || prox.second < 0) continue;

        if(tab[prox.first][prox.second] == '#') continue;

        

        if(viz[prox.first][prox.second]) continue;

        viz[prox.first][prox.second] = true;



        if(tab[prox.first][prox.second] == 'G') finded++;

        fila.push({prox.first + 1, prox.second});

        fila.push({prox.first - 1, prox.second});

        fila.push({prox.first, prox.second + 1});

        fila.push({prox.first, prox.second - 1});

    }



    if(ans == "No"){

        cout << ans << "\n";

    }else{

        cout << (finded == goods ? "Yes" : "No") << "\n";

    }



}





int main(){

    ios::sync_with_stdio(0);

    cin.tie(0);

    cout.tie(0);

    int t = 1;

    cin >> t;



    while(t--) solve();



    return 0;

}

