#include "bits/stdc++.h"

using namespace std;



#define int long long

#define You_Never_Know ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL)



#pragma GCC target ("avx2")

#pragma GCC optimization ("O3")

#pragma GCC optimization ("unroll-loops")



#ifdef Dhru08

#include "debug.hpp"

#else

#define debug(...) 8

#endif



int32_t main()

{

    You_Never_Know;

    int t;

    cin >> t;

    

    while(t--){

        

        int n,k;

        string s;

        cin >> n >> k >> s;

        

        if(n % k != 0){

            cout << "-1\n";

            continue;

        }

        

        map<char,int> mp;

        for(auto &c : s) mp[c]++;

        

        bool flag = true;

        for(auto &[c,f] : mp) if(f % k != 0) flag = false;

        

        if(flag) cout << s << '\n';

        else{

            string str(n,'z');

            for(int i = n - 1;i >= 0;i--){

                mp[s[i]]--;

                int len = (n - i - 1);

                for(char c = s[i] + 1;c <= 'z';c++){

                    mp[c]++;

                    int req = 0;

                    for(char ch = 'a';ch <= 'z';ch++) req += (k - (mp[ch] % k)) % k;

                    

                    if(req <= len){

                        int left = len - req;

                        if(left % k == 0){

                            str = s.substr(0,i) + c;

                            string extra;

                            for(char ch = 'a';ch <= 'z';ch++){

                                req = (k - (mp[ch] % k)) % k;

                                while(req--) extra.push_back(ch);

                            }

                            while(left--) extra.push_back('a');

                            sort(extra.begin(), extra.end());

                            str += extra;

                            flag = true;

                            break;

                        } 

                    }

                    mp[c]--;

                }

                if(flag) break;

            }

            cout << str << '\n';

        }

    }

    return 0;

}