/*
 * A Contest to Meet (ACM) is a reality TV contest that sets three contestants at three random
 * city intersections. In order to win, the three contestants need all to meet at any intersection
 * of the city as fast as possible.
 * It should be clear that the contestants may arrive at the intersections at different times, in
 * which case, the first to arrive can wait until the others arrive.
 * From an estimated walking speed for each one of the three contestants, ACM wants to determine the
 * minimum time that a live TV broadcast should last to cover their journey regardless of the contestants’
 * initial positions and the intersection they finally meet. You are hired to help ACM answer this question.
 * You may assume the following:
 *     Each contestant walks at a given estimated speed.
 *     The city is a collection of intersections in which some pairs are connected by one-way
 * streets that the contestants can use to traverse the city.
 *
 * This class implements the competition using Dijkstra's algorithm
 */
import java.util.*;
import java.io.*;

public class CompetitionDijkstra {
	
	// The key will be the vertex, the key of the value hash map will be the vertices it connects to
	// and the value of the value hash map will be the distance between each of the vertices.
	HashMap<Integer, HashMap<Integer, Double>> vertices = new HashMap<Integer, HashMap<Integer, Double>>();
	
	int numberOfVertices, numberOfPaths;
	int slowestSpeed;

    /**
     * @param filename: A filename containing the details of the city road network
     * @param sA, sB, sC: speeds for 3 contestants
     * @throws Exception 
    */
    CompetitionDijkstra (String filename, int sA, int sB, int sC) throws Exception
    {
    	if (sA < sB && sA < sC)
    		slowestSpeed = sA;
    	else if (sB < sA && sB < sC)
    		slowestSpeed = sB;
    	else
    		slowestSpeed = sC;
    	
    	createTree(filename);
    }

    /**
    * @return int: minimum minutes that will pass before the three contestants can meet
     */
    public int timeRequiredforCompetition() 
    {
    	double longestPath = -1.0;
    	// Generate the longest shortest path
    	for (int allPaths = 0; allPaths < numberOfVertices; allPaths++)
    	{
    		double longestLocalPath = dijkstra(allPaths);
    		if (longestLocalPath > longestPath)
    			longestPath = longestLocalPath;
    	}
    	
    	// divide that by the slowest competitor to get worst case scenario
    	// time required for the competition
		int timeForCompetition = (int) Math.ceil(((longestPath * 1000) / slowestSpeed));
    	
    	return timeForCompetition;
    }

    /**
     * @return the time it takes to get from the starting intersection to the finish
     */
    public double dijkstra(int startingVertex)
    {
    	// The key of hashmap is the distance of this path, the array list within has the path 
    	//HashMap<Double, ArrayList<Integer>> dist = new HashMap<Double, ArrayList<Integer>>();
    	    	
    	double dist[] = new double[numberOfVertices];
    	Boolean sptSet[] = new Boolean[numberOfVertices];
    	
    	int i;
    	for (i = 0; i < numberOfVertices; i++)
    	{
    		dist[i] = Integer.MAX_VALUE;
    		sptSet[i] = false;
    	}
    	
    	dist[startingVertex] = 0;
    	
    	for (i = 0; i < numberOfVertices; i++)
    	{
    		int u = minDist(dist, sptSet);
    		
    		sptSet[u] = true;
    		
    		for (int j = 0; j < numberOfVertices; j++)
    		{
    			if (vertices.get(u).containsKey(j) && !sptSet[j] && vertices.get(u).get(j) != 0 && dist[u] != Integer.MAX_VALUE && dist[u] + vertices.get(u).get(j) < dist[j]) 
    				dist[j] = dist[u] + vertices.get(u).get(j); 
    		}
    	}
    	
    	double distanceToReturn = longestShortestPath(dist);
    	
    	return distanceToReturn;
    }
    
    public int minDist(double dist[], Boolean sptSet[])
    {
         
        // Initialize min value 
        int min_index = -1; 
        double min = Double.MAX_VALUE;
  
        for (int v = 0; v < numberOfVertices; v++) 
        {
            if (sptSet[v] == false && dist[v] <= min) 
            { 
                min = dist[v]; 
                min_index = v; 
            }  
        }
        
    	return min_index;
    }
    
    public double longestShortestPath(double dist[])
    {
    	double longestDist = -1;
    	for (int i = 0; i < dist.length; i++)
    	{
    		if (dist[i] > longestDist)
    			longestDist = dist[i];
    	}
    	
    	return longestDist;
	}
	
	/**
     * @param filename
     * @throws Exception
     */
    public void createTree(String filename) throws Exception
    {
    	File file = new File(filename);
    	BufferedReader br = new BufferedReader(new FileReader(file));
    	String st;
    	st = br.readLine();
    	Scanner sc;
    	
    	numberOfVertices = Integer.parseInt(st);
    	for (int i = 0; i < numberOfVertices; i++) 
    	{
    		vertices.put(i, new HashMap<Integer, Double>());
    	}
    	
    	st = br.readLine();
    	numberOfPaths = Integer.parseInt(st);    	
    	while ((st = br.readLine()) != null)
    	{
    		sc = new Scanner(st);
    		
    		int vertex1 = Integer.parseInt(sc.next());
    		int vertex2 = Integer.parseInt(sc.next());
    		double distance = Double.parseDouble(sc.next());
    		
    		vertices.get(vertex1).put(vertex2, distance);
    		
        	sc.close();
		}
		br.close();
    }

    public static void main(String[] args) throws Exception
    {
    	CompetitionDijkstra d = new CompetitionDijkstra("1000EWD.txt", 55, 75, 75);
    	System.out.println(d.timeRequiredforCompetition() + " minutes");
    }
    
}