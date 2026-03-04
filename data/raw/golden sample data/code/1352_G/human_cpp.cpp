#include <bits/stdc++.h>

using namespace std;



bool can(vector<int>& v){

  bool t = 1;

  for (int i = 1 ; v.size() > i ; i++){

    int a = abs(v[i] - v[i - 1]);

    if (5 <= a || a <= 1){

      return 0;

    }

  }

  return 1;

}



void solve(){

  int n; cin >> n;

  vector<int> v(n);

  if (n == 2 || n == 3){

    cout << "-1\n";

    return;

  }



  if (n == 4){

    cout << "3 1 4 2\n";

    return;

  }



  int x = 1, a = n, y = 0, z = 0;

  if (n & 1){

    for ( ; x <= n ; x += 2, y++){

      v[y] = x;

    }



    z = y;

    x = n;





    x--;



    for ( ; x > 0 ; x -= 2, y++){

      v[y] = x;

    }



    y = z;



    swap(v[y], v[y + 1]);

  }

  else{

    x = 2;

    for (int i = 0 ; n / 2 > i ; x += 2, i++){

      v[i] = x;

    }



    x = n - 1;

    for (int i = n / 2 ; n > i ; x -= 2, i++){

      v[i] = x;

    }

    swap(v[n / 2], v[n / 2 + 1]);



  }



  for (auto& x: v){

    cout << x << " ";

  }



  cout << '\n';

}



int main(){

  ios_base::sync_with_stdio(0);

  cin.tie(0); cout.tie(0);



  int _= 1; cin >> _;

  while (_--){

    solve();

  }

}

