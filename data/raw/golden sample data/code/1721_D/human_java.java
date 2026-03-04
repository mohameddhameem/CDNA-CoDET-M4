





import java.io.*;

import java.math.*;

import java.util.*;







// @author : Dinosparton 



public class test {

	 

	   static class Pair{ 

		   long x;

		   long y;

		   

		   Pair(long x,long y){ 

			   this.x = x;

			   this.y = y;

			  

		   }

	   }

	   

	  static class Nodal_Point {

		    

		     String PointName;



		    Nodal_Point(){



		    }



		    Nodal_Point(String PointName){

		        this.PointName = PointName;

		    }



		    public String getPointName() {

		        return PointName;

		    }



		    public void setPointName(String pointName) {

		        PointName = pointName;

		    }

		}

	   

	   static class Road {



		     int road_id;



		     String src;



		     String des;



		    Road() {

		    }



		    public Road(int id, String src, String des) {

		        this.road_id = id;

		        this.src = src;

		        this.des = des;

		    }



		    public int getRoad_id() {

		        return road_id;

		    }



		    public void setRoad_id(int road_id) {

		        this.road_id = road_id;

		    }



		    public String getSrc() {

		        return src;

		    }



		    public void setSrc(String src) {

		        this.src = src;

		    }



		    public String getDes() {

		        return des;

		    }



		    public void setDes(String des) {

		        this.des = des;

		    }

		}

	   

	   static class Student {



		    private String StudentName;



		    private String ContactNo;



		    private String NodalPoint;



		    Student(){



		    }



		    public Student(String studentName, String contactNo, String nodalPoint) {

		        StudentName = studentName;

		        ContactNo = contactNo;

		        NodalPoint = nodalPoint;

		    }



		    public String getStudentName() {

		        return StudentName;

		    }



		    public void setStudentName(String studentName) {

		        StudentName = studentName;

		    }



		    public String getContactNo() {

		        return ContactNo;

		    }



		    public void setContactNo(String contactNo) {

		        ContactNo = contactNo;

		    }



		    public String getNodalPoint() {

		        return NodalPoint;

		    }



		    public void setNodalPoint(String nodalPoint) {

		        NodalPoint = nodalPoint;

		    }

		}

	   static class Compare { 

		   

		     void compare(Pair arr[], int n) 

		    { 

		        // Comparator to sort the pair according to second element 

		        Arrays.sort(arr, new Comparator<Pair>() { 

		            @Override public int compare(Pair p1, Pair p2) 

		            { 

		            	if(p1.x!=p2.x) {

		                return (int)(p1.x - p2.x); 

		            	}

		            	else { 

		            		return (int)(p1.y - p2.y);

		            	}

		            } 

		        }); 

		  

//		        for (int i = 0; i < n; i++) { 

//		            System.out.print(arr[i].x + " " + arr[i].y + " "); 

//		        } 

//		        System.out.println(); 

		    } 

		} 

	 

	   static class Scanner {

	        BufferedReader br;

	        StringTokenizer st;

	 

	        public Scanner()

	        {

	            br = new BufferedReader(

	                new InputStreamReader(System.in));

	        }

	 

	        String next()

	        {

	            while (st == null || !st.hasMoreElements()) {

	                try {

	                    st = new StringTokenizer(br.readLine());

	                }

	                catch (IOException e) {

	                    e.printStackTrace();

	                }

	            }

	            return st.nextToken();

	        }

	 

	        int nextInt() { return Integer.parseInt(next()); }

	 

	        long nextLong() { return Long.parseLong(next()); }

	 

	        double nextDouble()

	        {

	            return Double.parseDouble(next());

	        }

	 

	        String nextLine()

	        {

	            String str = "";

	            try {

	                str = br.readLine();

	            }

	            catch (IOException e) {

	                e.printStackTrace();

	            }

	            return str;

	        }

	    }

	  

	  

	   static boolean function(long val,int a[],int b[]) {

		   

		   ArrayList<Long> list1 = new ArrayList<>();

		   ArrayList<Long> list2 = new ArrayList<>();

		   

		   for(int i=0;i<a.length;i++) {

			   list1.add(a[i]&val);

		   }

		   for(int i=0;i<b.length;i++) {

			   list2.add((~b[i])&val);

		   }

		   

		   Collections.sort(list1);

		   Collections.sort(list2);

		   

		   return list1.equals(list2);

	   }

	   public static void main(String args[]) throws Exception { 

		

		   Scanner sc = new Scanner();

		   StringBuilder res = new StringBuilder();

		   

		   int tc = sc.nextInt();

		  

		   while(tc-->0) {

			   

			   int n = sc.nextInt();

			   

			   int a[] = new int[n];

			   int b[] = new int[n];

			   

			   for(int i=0;i<n;i++) {

				   a[i] = sc.nextInt();

			   }

			   

			   for(int i=0;i<n;i++) {

				   b[i] = sc.nextInt();

			   }

			   

			   long ans = 0;

			   

			   for(int i = 29;i>=0;i--) {

				   

				   long temp = (ans | (1 << i));

				   

				   if(function(temp,a,b)) {

					   ans = temp;

				   }

			   }

			   

			   res.append(ans+"\n");

		   }

		   System.out.println(res);

	   }

}  









