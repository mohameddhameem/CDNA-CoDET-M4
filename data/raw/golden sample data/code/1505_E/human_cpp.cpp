#include <bits/stdc++.h>

using namespace std;

 

int main() {

  int n, m;

  cin>>n>>m;

  vector<vector<int> > v(2+n, vector<int> (2+m, 0));

  char c;

  for(int i = 1; i <= n; i++) {

    for(int j=1; j<=m; j++) {

      cin>>c;

      if(c=='*') v[i][j]=1;

    }

  }

  int ans = 0;

  int i=1, j=1;

  while(i!=n || j!=m) {

    if(v[i][j]){

      ans++;

    } 

    if(v[i][j+1]) {

      j++;

      continue;

    }

    if(v[i+1][j]) {

      i++;

      continue;

    }

    if(i == n) {

      j++;

      continue;

    }

    if(j == m) {

      i++;

      continue;

    }

    j++;

  }

  if(v[n][m]) ans++;

  cout<<ans<<"\n";

    return 0;

}

