#include<bits/stdc++.h>

using namespace std;

#define ll long long 

void fun()

{

 



ll n,m;

cin>>n>>m;



string s,a = "";

cin>>s;

a = s;

 while(m--)

 {

  for(int i=0;i<n;i++)

  {

    if(i ==0)

    {

      if(s[i] == '0' and s[i+1] == '1')

      {

        a[i] = '1';

      }

      continue;



    }



    if(i == n-1)

    {

      if(s[i-1] == '1' and s[i] == '0')

      {

        a[i] = '1';

      }

      continue;

    }

    if(s[i] == '0' and  (s[i+1] == '1' || s[i-1] == '1'))

    {

      if(s[i+1] != s[i-1])

      {

        a[i] = '1';

      }

    }

  }

  if(s==a)

    break;

  s = a;

 }

cout<<s<<endl;



}









int main()

 {

    ll t;

    cin >> t;



    while(t--)

    {

        fun();

    }

  // fun();

}

