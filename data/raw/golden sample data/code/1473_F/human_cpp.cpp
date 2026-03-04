#include <bits/stdc++.h>
using namespace std;
// #define int long long
#define oo 1e9

int t, n, a[3005], b[3005], mod = 1e9 + 7;
mt19937_64 rng;

int mul(int x, int y){
    if(y == 0) return 1;
    int ans = mul(x, y / 2);
    if(y % 2 == 0) return (ans * ans) % mod;
    else return (((ans * ans) % mod) * x) % mod;
} 

struct edge{
	int u, v, cap, flow = 0;
	edge(int A, int B, int C){
		u = A, v = B, cap = C;
	}
};

struct Dinic{
	vector<edge> edges;
	vector<vector<int> > adj;
	vector<int> ptr, level;
	queue<int> q;
	int n, m = 0, s, t;
	Dinic(int A, int B, int C){
		n = A, s = B, t = C; 
		adj.resize(n);
		ptr.resize(n);
		level.resize(n);
	}
	void addedge(int u, int v, int cap){
		edges.emplace_back(u, v, cap);
		edges.emplace_back(v, u, 0);
		adj[u].push_back(m);
		adj[v].push_back(m + 1);
		m += 2;
	}
	bool reset(){
		while(!q.empty()){ 
			int v = q.front();  
			q.pop();
			for(auto i: adj[v]){ 
				if(level[edges[i].v] != -1 || edges[i].cap - edges[i].flow < 1) continue;
				level[edges[i].v] = level[v] + 1;
				q.push(edges[i].v);
			}
		} 
		return level[t] != -1;
	}
	int dfs(int u, int pushed){ 
		if(pushed == 0) return 0;
		if(u == t) return pushed;
		for(int i = ptr[u]; i < adj[u].size(); i++, ptr[u]++){
			int id = adj[u][i];
			int v = edges[id].v;
			if(level[u] + 1 != level[v] || edges[id].cap - edges[id].flow < 1) continue;
			int tr = dfs(v, min(pushed, edges[id].cap - edges[id].flow));
			if(tr == 0) continue;
			edges[id].flow += tr;
			edges[id ^ 1].flow -= tr;
			return tr;
		}
		return 0;
	}
	int flow(){
		int f = 0;
		while(1){
			fill(level.begin(), level.end(), -1);
			level[s] = 0;
			q.push(s); 
			if(!reset()) break;
			fill(ptr.begin(), ptr.end(), 0);
			while(int pushed = dfs(s, oo)) f += pushed;    
		}
		return f;
	}
};

void solve(){  
	Dinic osu(n + 2, 0, n + 1);
	int sum = 0;
	for(int i = 1; i <= n; i++){
		if(b[i] >= 0){
			sum += b[i];
			osu.addedge(0, i, b[i]);
		} 
		else osu.addedge(i, n + 1, -b[i]);
	}  
	for(int i = 1; i <= n; i++){
		bool mp[105];
		for(int j = 1; j <= 100; j++) mp[j] = 0; 
		for(int j = i - 1; j >= 1; j--){
			if(a[i] % a[j] == 0 && mp[a[j]] == 0){
				mp[a[j]] = 1, osu.addedge(i, j, oo); 
			}
		}
	}
	cout << sum - osu.flow();
}

signed main(){
    ios_base::sync_with_stdio(NULL); cin.tie(nullptr); cout.tie(nullptr);
	// rng.seed((int)main ^ time(0));
    #ifdef Kawaii
    	auto starttime = chrono::high_resolution_clock::now();
    #endif

    cin >> n;
    for(int i = 1; i <= n; i++) cin >> a[i];
    for(int i = 1; i <= n; i++) cin >> b[i];
    solve();

    #ifdef Kawaii
    	auto endtime = chrono::high_resolution_clock::now();
    	auto duration = chrono::duration_cast<chrono::milliseconds>(endtime - starttime).count(); 
    	cout << "\n=====" << "\nUsed: " << duration << " ms\n";
    #endif
}
    	    	  	 		  		 		  	  	 	