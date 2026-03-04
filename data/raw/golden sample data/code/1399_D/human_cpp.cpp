#include<bits/stdc++.h>

using namespace std;

#define int long long

#define endl '\n'



//digit_dp snipet,used when range and a function is and range is very large 

//create memo for storing transition states : idx,is_last,(..,..,..things qus want) and some optimization on range of qus req

//int memo[][][][];

// int digit_dp(string&s,int idx,int sum,bool last)

// {

//     if(idx== s.size())

//     {

//         return sum;

//     }



//     if(memo[idx][sum][last]!=-1)

//     {

//         return memo[idx][sum][last];

//     }`

		

//         int till = (last?s[idx]-'0':9);

//         int ans = 0;

//         for(int digits=0;digits<=till;digits++)

//         {

//             ans += digit_dp(s,idx+1,sum+digits,(last and (digits==till)));

//         }





//         return memo[idx][sum][last]=ans;



// }

//string_view shows substr in O(1) , use when read intensive only



// void work( int i ,string&s,vector<int>&a,char last,int sub_n)

// {

//     if(i>s.size())

//     return;



//     if(last=='1')

//     {

//         int j=i;

//         while( j<s.size()and s[j]!='0' and a[j]==0)

//         {

//             j++;

//         }

//         // start +=s[j];

//         a[j]=sub_n;

//         work(j+1,s,a,'0',sub_n);

//     }

//     else

//     {

//         int j=i;

//         while(j<s.size()and s[j]!='1' and a[j]==0)

//         {

//             j++;

//         }

//         // start +=s[j];

//         a[j]=sub_n;

//         work(j+1,s,a,'1',sub_n);

//     }



//     return;



// }

void solve()

{

		int n;

		cin>>n;

		string s;

        cin>>s;



//         cout<<"working"<<endl;

//         //// string s_1,s_0;

//         vector<int>a(n+1,0);



//         int i=0;

//         int j=0;

//         while(i<s.size() )

//         {

//             // i++;

//             if(s[i] == '1')

//             break;



//             i++;

//         }

//         // s_1 +=s[i];

//         if(i<s.size())

//         { a[i] =1;

//         work(i,s,a,'1',1);

//         }



//         while(j<s.size() )

//         {

//             if(s[j] == '0')

//             break;



//             j++;

//         }

//         // s_0 +=s[j];

//         if(j<s.size())

//         { a[j]=2;

//         work(j,s,a,'0',2);

//         i=3;



//         }

//             if(i!=3)

//             i=2;



//         for(int x=0;x<n;x++)

//         {

//             if(a[x]==0)

//             {

//                 a[x] = i;

//                 i++;

//             }

//         }

//         cout<<i<<endl;

//         for(int x=0;x<n;x++)

//         {

//             cout<<a[x]<<" ";

//         }

//         // cout<<endl;

// 		return ;

    

	vector<int> zero, one;



	vector<int> ans(n, -1);

	int col = 1;

//if 1 then 0 if 0 then 1 

	for (int i = 0; i < n; ++i){

		if (s[i] == '0'){

			if (one.empty()){

				zero.push_back(i);

				ans[i] = col++;

			}

			else{

				zero.push_back(i);

				ans[i] = ans[one.back()];

				one.pop_back();

			}

		}

		else{

			if (zero.empty()){

				one.push_back(i);

				ans[i] = col++;

			}

			else{

				one.push_back(i);

				ans[i] = ans[zero.back()];

				zero.pop_back();

			}

		}

	}

 

	cout << col - 1 << endl;

	for(auto e: ans){

		cout << e << " ";

	}

   cout << endl;



}

signed main()

{

	    ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);

	    int t=1;

	    cin>>t;

		while(t--)

		{

			solve();

		}

		return 0;

}



