import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/*					10 		100			1000			1000 duplicate			nearlyOrdered			reverse			sorted
 * insertion:		7443	64168		6770565			3098309					6404163					3387756			77354
 * selection:		5313	115586		3842771			12132678				5815258					2576593			3785964
 * quick:			19837	406218		4884580			7534113					6495079					10598498		12063241
 * merge Iterative:	26612	64792		306685			458079					443882					735399			630459
 * merge Recursive:	19610	218262		187194			312626					333267					345894			375113
 */

//-------------------------------------------------------------------------
/**
 *  Test class for SortComparison.java
 *
 *  @author
 *  @version HT 2020
 */
@RunWith(JUnit4.class)
public class SortComparisonTest
{
    //~ Constructor ........................................................
    @Test
    public void testConstructor()
    {
        new SortComparison();
    }

    //~ Public Methods ........................................................

    // ----------------------------------------------------------
    /**
     * Check that the methods work for empty arrays
     */
    @Test
    public void testEmpty()
    {
    	double a[] = {};
    	double answer[];
    	
    	answer = SortComparison.insertionSort(a);
    	assertEquals(answer, a);
    	
    	answer = SortComparison.selectionSort(a);
    	assertEquals(answer, a);
    	
    	answer = SortComparison.quickSort(a);
    	assertEquals(answer, a);
    	
    	answer = SortComparison.mergeSortIterative(a);
    	assertEquals(answer, a);
    	
    	answer = SortComparison.mergeSortRecursive(a);
    	assertEquals(answer, a);
    	
    }
    
    // TODO: add more tests here. Each line of code and ech decision in Collinear.java should
    // be executed at least once from at least one test.
    
    @Test
    public void testTrivial()
    {
    	double a[] = {1};
    	double answer[];
    	
    	answer = SortComparison.insertionSort(a);
    	assertEquals(answer, a);
    	
    	answer = SortComparison.selectionSort(a);
    	assertEquals(answer, a);
    	
    	answer = SortComparison.quickSort(a);
    	assertEquals(answer, a);
    	
    	answer = SortComparison.mergeSortIterative(a);
    	assertEquals(answer, a);
    	
    	answer = SortComparison.mergeSortRecursive(a);
    	assertEquals(answer, a);
    }
    
    @Test
    public void testInsertion()
    {
    	double a[] = {7, 3};
    	double expected[] = {3, 7};
    	double answer[] = SortComparison.insertionSort(a);
    	assertEquals(Arrays.toString(expected), Arrays.toString(answer));
    }
    
    @Test
    public void testSelection()
    {
    	double a[] = {9, 7, 13, 46, 2};
    	double expected[] = {2, 7, 9, 13, 46};
    	double answer[] = SortComparison.selectionSort(a);
    	assertEquals(Arrays.toString(expected), Arrays.toString(answer));
    }

    @Test
    public void testQuick()
    {
    	double a[] = {72, 13, 45, 76, 9, 12, 600};
    	double expected[] = {9, 12, 13, 45, 72, 76, 600};
    	double answer[] = SortComparison.selectionSort(a);
    	assertEquals(Arrays.toString(expected), Arrays.toString(answer));
    	
    	
    	double b[] = {600, 13, 45, 76, 9, 12, 72};
    	answer = SortComparison.selectionSort(b);
    	assertEquals(Arrays.toString(expected), Arrays.toString(answer));
    	
    	double c[] = {9, 600, 13, 45, 76, 12, 72};
    	answer = SortComparison.selectionSort(c);
    	assertEquals(Arrays.toString(expected), Arrays.toString(answer));
    }
    
    @Test
    public void testMergeIterative()
    {
    	double a[] = {72, 13, 45, 76, 9, 12, 600};
    	double expected[] = {9, 12, 13, 45, 72, 76, 600};
    	double answer[] = SortComparison.mergeSortIterative(a);
    	assertEquals(Arrays.toString(expected), Arrays.toString(answer));
    }
    
    @Test
    public void testMergeRecursive()
    {
    	double a[] = {72, 13, 45, 76, 9, 12, 600};
    	double expected[] = {9, 12, 13, 45, 72, 76, 600};
    	double answer[] = SortComparison.mergeSortRecursive(a);
    	assertEquals(Arrays.toString(expected), Arrays.toString(answer));
    }
    // ----------------------------------------------------------
    /**
     *  Main Method.
     *  Use this main method to create the experiments needed to answer the experimental performance questions of this assignment.
     *
     */
    public static void main(String[] args) throws FileNotFoundException
    {
    	Scanner fileScanner = new Scanner(new File("/home/brendan/eclipse-workspace/numbersSorted1000.txt")); 
        int counter = 0;
        while(fileScanner.hasNextDouble())
        {
        	counter++;
        	fileScanner.nextDouble();
        }
              
        fileScanner = new Scanner(new File("/home/brendan/eclipse-workspace/numbersSorted1000.txt"));
        
        double[] array = new double[counter];
        for(int i = 0; i < array.length; ++i)
        	array[i] = fileScanner.nextDouble();
        fileScanner.close();
        double array2[] = array.clone();
        double array3[] = array.clone();
        
        long after;
        long time = System.nanoTime();
        SortComparison.insertionSort(array);
        SortComparison.insertionSort(array2);
        SortComparison.insertionSort(array3);
        after = System.nanoTime();
        
        long duration = after - time;
        System.out.println("insertion: " + (duration / 3));
        
        time = System.nanoTime();
        SortComparison.selectionSort(array);
        SortComparison.selectionSort(array2);
        SortComparison.selectionSort(array3);
        after = System.nanoTime();
        
        duration = after - time;
        System.out.println("selection: " + (duration / 3));
        
        time = System.nanoTime();
        SortComparison.quickSort(array);
        SortComparison.quickSort(array2);
        SortComparison.quickSort(array3);
        after = System.nanoTime();
        
        duration = after - time;
        System.out.println("quick: " + (duration / 3));
        
        time = System.nanoTime();
        
        SortComparison.mergeSortIterative(array);
        SortComparison.mergeSortIterative(array2);
        SortComparison.mergeSortIterative(array3);
        after = System.nanoTime();
        
        duration = after - time;
        System.out.println("merge iterative: " + (duration / 3));
        
        time = System.nanoTime();
        
        SortComparison.mergeSortRecursive(array);
        SortComparison.mergeSortRecursive(array2);
        SortComparison.mergeSortRecursive(array3);
        after = System.nanoTime();
        
        duration = after - time;
        System.out.println("merge recursive: " + (duration / 3));
    }

}