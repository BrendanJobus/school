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
 * This class implements the competition using Floyd-Warshall algorithm
 */

import java.util.*;
import java.io.*;

public class CompetitionFloydWarshall {

    // The key will be the vertex, the key of the value hash map will be the vertices it connects to
	// and the value of the value hash map will be the distance between each of the vertices.
	HashMap<Integer, HashMap<Integer, Double>> vertices = new HashMap<Integer, HashMap<Integer, Double>>();
	
	int numberOfVertices, numberOfPaths;
	int slowestSpeed;

    /**
     * @param filename: A filename containing the details of the city road network
     * @param sA,       sB, sC: speeds for 3 contestants
     * @throws Exception
     */
    CompetitionFloydWarshall(String filename, int sA, int sB, int sC) throws Exception {
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

		for (int allPaths = 0; allPaths < numberOfVertices; allPaths++)
		{
			double longestLocalPath = floydWarshall(allPaths);
			if (longestLocalPath > longestPath)
				longestPath = longestLocalPath;
		}

		int timeForCompetition = (int) Math.ceil(((longestPath * 1000) / slowestSpeed));

        return timeForCompetition;
	}
	

	public double floydWarshall(int startingVertex)
	{
		double dist[][] = new double[numberOfVertices][numberOfVertices];
		int i, j, k;

		for (i = 0; i < numberOfVertices; i++)
		{
			for (j = 0; j < numberOfVertices; j++)
			{
				dist[i][j] = Double.MAX_VALUE;
			}
		}

		for (i = 0; i < numberOfVertices; i++)
			for (j = 0; j < numberOfVertices; j++)
				if(vertices.get(i).containsKey(j))
					dist[i][j] = vertices.get(i).get(j);

		for (k = 0; k < numberOfVertices; k++)
		{
			for (i = 0; i < numberOfVertices; i++)
			{
				for (j = 0; j < numberOfVertices; j++)
				{
					if(dist[i][k] + dist[k][j] < dist[i][j])
						dist[i][j] = dist[i][k] + dist[k][j];
				}
			}
		}

		double worstCasePath = longestPath(dist);
		
		return worstCasePath;
	}

	public double longestPath(double dist[][])
	{
		double worstCasePath = -1.0;
		for (int i = 0; i < numberOfVertices; i++)
		{
			for (int j = 0; j < numberOfVertices; j++)
			{
				if (dist[i][j] > worstCasePath)
					worstCasePath = dist[i][j];
			}
		}

		return worstCasePath;
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
    	CompetitionFloydWarshall d = new CompetitionFloydWarshall("tinyEWD.txt", 55, 75, 75);
    	System.out.println(d.timeRequiredforCompetition() + " minutes");
    }

}