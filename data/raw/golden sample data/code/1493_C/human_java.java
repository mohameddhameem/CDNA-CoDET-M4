import java.util.*;
import java.io.*;

public class kth_beautiful_strings {
	
	public static void main(String args[]) throws Exception {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        StringBuilder sb = new StringBuilder();
        
        int t = Integer.parseInt(in.readLine());
        
        while (t-- > 0) {
        	
            String s[] = in.readLine().split(" ");
            
            int n = Integer.parseInt(s[0]), k = Integer.parseInt(s[1]);
            
            char c[] = in.readLine().toCharArray();
            
            if (n % k != 0) {
            	sb.append("-1\n"); 
            	continue;
            }
            
            int i, j, d[][] = new int[n + 1][26], pos = -1;
            
            for (i = 1; i <= n; i++) {
            	
                int ch = c[i - 1] - 'a';
                
                for (j = 0; j < 26; j++) {
                	d[i][j] = d[i - 1][j];
                }
                
                int req = 0;
                
                for (j = 0; j < 26; j++) {
                	req += (k - d[i][j]) % k;
                }
                
                boolean ps = false;
                
                for (j = ch + 1; j < 26; j++) {
                    int re = req - ((k - d[i][j]) % k);
                    re += k - d[i][j];
                    int ex = (n - i + 1) - re;
                    if (ex >= 0 && ex % k == 0) {
                    	ps = true; 
                    	break;
                    }
                }
                
                if (ps) {
                	pos = i;
                }
                
                d[i][ch] = (d[i][ch] + 1) % k;
            }
            
            boolean sim = true;
            
            for (i = 0; i < 26; i++) {
            	
            	if (d[n][i] !=0 ) {
            		sim = false;
            		break;
            	}
            	
            }
            
            if (sim) {
            	sb.append((new String(c))+"\n"); 
            	continue;
            }
            
            if (pos == -1) {
            	sb.append("-1\n"); 
            	continue;
            }
            
 
            int add[] = new int[26], req = 0; j = c[pos-1]-'a';
            
            d[pos][j] = (d[pos][j] - 1 + k) % k;
            
            for (i = 0; i < 26; i++) {
            	
                add[i] = (k - d[pos][i]) % k;
                req += add[i];
                
            }
 
            int an = 0;
            
            for (i = j + 1; i < 26; i++) {
            	
                int re = req - add[i];
                re += k - d[pos][i];
                int ex = (n - pos + 1) - re;
                
                if (ex >= 0 && ex % k == 0) {
                	
                    an = i;
                    
                    if (add[i] == 0) {
                    	add[i] = k;
                    }
                    
                    add[0] += ex;
                    break;
                    
                }
            }
            
            for (i = 0; i < pos - 1; i++) {
            	sb.append(c[i]);
            }
            
            sb.append((char)(an +'a'));
            
            add[an]--;
            
            for (i = 0; i < 26; i++) {
	            for (j = 0; j < add[i]; j++) {
	            	sb.append((char)('a'+i));
	            }
            }
            
            sb.append("\n");
            
        }
        
        System.out.println(sb);
    }

}