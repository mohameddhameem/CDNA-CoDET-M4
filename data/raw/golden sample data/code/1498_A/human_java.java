
import java.io.*;
import java.util.*;
import java.lang.*;


public class Codeforces {
	static class Reader {
        final private int BUFFER_SIZE = 1 << 16;
        private DataInputStream din;
        private byte[] buffer;
        private int bufferPointer, bytesRead;
 
        public Reader()
        {
            din = new DataInputStream(System.in);
            buffer = new byte[BUFFER_SIZE];
            bufferPointer = bytesRead = 0;
        }
 
        public Reader(String file_name) throws IOException
        {
            din = new DataInputStream(
                new FileInputStream(file_name));
            buffer = new byte[BUFFER_SIZE];
            bufferPointer = bytesRead = 0;
        }
 
        public String readLine() throws IOException
        {
            byte[] buf = new byte[200001]; // line length
            int cnt = 0, c;
            while ((c = read()) != -1) {
                if (c == '\n') {
                    if (cnt != 0) {
                        break;
                    }
                    else {
                        continue;
                    }
                }
                buf[cnt++] = (byte)c;
            }
            return new String(buf, 0, cnt);
        }
 
        public int nextInt() throws IOException
        {
            int ret = 0;
            byte c = read();
            while (c <= ' ') {
                c = read();
            }
            boolean neg = (c == '-');
            if (neg)
                c = read();
            do {
                ret = ret * 10 + c - '0';
            } while ((c = read()) >= '0' && c <= '9');
 
            if (neg)
                return -ret;
            return ret;
        }
        public long nextLong() throws IOException
        {
            long ret = 0;
            byte c = read();
            while (c <= ' ')
                c = read();
            boolean neg = (c == '-');
            if (neg)
                c = read();
            do {
                ret = ret * 10 + c - '0';
            } while ((c = read()) >= '0' && c <= '9');
            if (neg)
                return -ret;
            return ret;
        }
 
        public double nextDouble() throws IOException
        {
            double ret = 0, div = 1;
            byte c = read();
            while (c <= ' ')
                c = read();
            boolean neg = (c == '-');
            if (neg)
                c = read();
 
            do {
                ret = ret * 10 + c - '0';
            } while ((c = read()) >= '0' && c <= '9');
 
            if (c == '.') {
                while ((c = read()) >= '0' && c <= '9') {
                    ret += (c - '0') / (div *= 10);
                }
            }
 
            if (neg)
                return -ret;
            return ret;
        }
 
        private void fillBuffer() throws IOException
        {
            bytesRead = din.read(buffer, bufferPointer = 0,
                                 BUFFER_SIZE);
            if (bytesRead == -1)
                buffer[0] = -1;
        }
 
        private byte read() throws IOException
        {
            if (bufferPointer == bytesRead)
                fillBuffer();
            return buffer[bufferPointer++];
        }
 
        public void close() throws IOException
        {
            if (din == null)
                return;
            din.close();
        }}
    static void cout(Object line) {System.out.println(line);}
	static void iop() {
		try {
			System.setIn(new FileInputStream("input.txt"));
			System.setOut(new PrintStream(new FileOutputStream("output.txt")));
		} catch (Exception e) {
			System.err.println("Error");
		}}
	static int gcd(int a, int b) {
		if(b == 0) return a;
		else return gcd(b,a%b);}
    static long gcd(long a, long b) {
        if(b == 0) return a;
        else return gcd(b,a%b);}
	static boolean isPerfectSquare(long n){
    	if (Math.ceil((double)Math.sqrt(n)) ==Math.floor((double)Math.sqrt(n))) return true; 
    	else return false;}
    static int fact(int n){
  		  return 
  		  	(n == 1 || n == 0) ? 1 : n * fact(n - 1);}
    static boolean isPowerOfTwo (int x) {return x!=0 && ((x&(x-1)) == 0);}
    static void printArray(char arr[]) {
    	for(int i = 0; i < arr.length; i++) {
    		System.out.print(arr[i] + " ");
    	}}
    static int highestPowerof2lessthanx(int x){
       
        // check for the set bits
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
         
        // Then we remove all but the top bit by xor'ing the
        // string of 1's with that string of 1's shifted one to
        // the left, and we end up with just the one top bit
        // followed by 0's.
        return x ^ (x >> 1); }
    static void reverse(int[] array) {
  
        // Length of the array
        int n = array.length;
  
        // Swaping the first half elements with last half
        // elements
        for (int i = 0; i < n / 2; i++) {
  
            // Storing the first half elements temporarily
            int temp = array[i];
  
            // Assigning the first half to the last half
            array[i] = array[n - i - 1];
  
            // Assigning the last half to the first half
            array[n - i - 1] = temp;
        }}

	public static void main(String[] args) throws IOException{
	 	iop(); 
		Reader s = new Reader();
		// Scanner s = new Scanner(System.in);
		int t = s.nextInt();
		while(t-- > 0) 
		{   
            long n = s.nextLong();
            long temp = n,sum=0;
            while(temp != 0) {
                sum += temp%10;
                temp/=10;
            }
            long gc = gcd(n,sum);
            if(gc > 1) cout(n);
            else {
                n++;
                temp = n;sum=0;
                while(temp != 0) {
                    sum += temp%10;
                    temp/=10;
                }
                gc = gcd(n,sum);
                if(gc > 1) cout(n);
                else {
                    n++;
                    temp = n;sum=0;
                    while(temp != 0) {
                        sum += temp%10;
                        temp/=10;
                    }
                    gc = gcd(n,sum);
                    if(gc > 1) cout(n);
                }
            }
		}
  	}
}
