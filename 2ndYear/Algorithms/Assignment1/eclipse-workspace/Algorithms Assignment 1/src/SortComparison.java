import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

// -------------------------------------------------------------------------

/**
 *  This class contains static methods that implementing sorting of an array of numbers
 *  using different sort algorithms.
 *
 *  @author
 *  @version HT 2020
 */

 class SortComparison {

    /**
     * Sorts an array of doubles using InsertionSort.
     * This method is static, thus it can be called as SortComparison.sort(a)
     * @param a: An unsorted array of doubles.
     * @return array sorted in ascending order.
     *
     */
    static double [] insertionSort (double a[])
    {    	
    	if(isEmpty(a))
    		return null;
    	
    	if(isTrivial(a))
    		return a;
    	
    	int size = a.length;
    	for(int i = 0; i < size; i++)
    	{
    		double key = a[i];
    		int j = i - 1;
    		while(j >= 0 && key < a[j])
    		{
    			a[j + 1] = a[j];
    			j--;
    		}
    		a[j + 1] = key;
    	}
    	
    	return a;
    }//end insertionsort
	
	    /**
     * Sorts an array of doubles using Selection Sort.
     * This method is static, thus it can be called as SortComparison.sort(a)
     * @param a: An unsorted array of doubles.
     * @return array sorted in ascending order
     *
     */
    static double [] selectionSort (double a[])
    {
    	if(isEmpty(a))
    		return null;
    	
    	if(isTrivial(a))
    		return a;
    	
    	int size = a.length;
    	int minPos;
    	for(int i = 0; i < size - 1; i++)
    	{
    		minPos = i;
    		for(int j = i + 1; j < size; j++)
    		{
    			if(a[j] < a[minPos])
    				minPos = j;
    		}
    		
    		double temp = a[minPos];
    		a[minPos] = a[i];
    		a[i] = temp;
    	}
    	
    	return a;
    }//end selectionsort

    /**
     * Sorts an array of doubles using Quick Sort.
     * This method is static, thus it can be called as SortComparison.sort(a)
     * @param a: An unsorted array of doubles.
     * @return array sorted in ascending order
     *
     */
    static double [] quickSort (double a[])
    {
		 //todo: implement the sort
    	if(isEmpty(a))
    		return null;
    	
    	if(isTrivial(a))
    		return a;
    	
    	quickRecursive(a, 0, a.length - 1);
    	return a;
    }//end quicksort

    /**
     * Sorts an array of doubles using Merge Sort.
     * This method is static, thus it can be called as SortComparison.sort(a)
     * @param a: An unsorted array of doubles.
     * @return array sorted in ascending order
     *
     */
    /**
     * Sorts an array of doubles using iterative implementation of Merge Sort.
     * This method is static, thus it can be called as SortComparison.sort(a)
     *
     * @param a: An unsorted array of doubles.
     * @return after the method returns, the array must be in ascending sorted order.
     */

    static double[] mergeSortIterative (double a[]) 
    {
    	int N = a.length;
    	double[] aux = new double[N];
    	for(int sz = 1; sz < N; sz = sz + sz)
    	{
    		for(int lo = 0; lo < N - sz; lo += sz + sz)
    		{
    			merge(a, aux, lo, lo + sz - 1, Math.min(lo + sz + sz - 1, N - 1));
    		}
    	}
    	
    	return a;
    }//end mergesortIterative
    
    
    
    /**
     * Sorts an array of doubles using recursive implementation of Merge Sort.
     * This method is static, thus it can be called as SortComparison.sort(a)
     *
     * @param a: An unsorted array of doubles.
     * @return after the method returns, the array must be in ascending sorted order.
     */
    static double[] mergeSortRecursive (double a[]) 
    {
    	double aux[] = new double[a.length];
    	mergeSort(a, aux, 0, a.length - 1);
    	return a;
   }//end mergeSortRecursive
    
    
    
    
    	
    static boolean isEmpty(double a[])
    {
    	if(a == null)
    		return true;
    	return false;
    }
    
    static boolean isTrivial(double a[])
    {
    	if(a.length == 1)
    		return true;
    	return false;
    }
    
    static int partition(double a[], int lo, int hi)
    {
    	int i = lo;
    	int j = hi + 1;
    	double pivot = a[lo];
    	while(true)
    	{
    		while( (Double.compare(a[++i], pivot) < 0) )					//(a[++i].compareTo(pivot) < 0) )
    		{
    			if(i == hi)
    				break;
    		}
    		while( (Double.compare(pivot, a[--j]) <= 0) )				//(pivot.compareTo(a[--j]) < 0) )
    		{
    			if(j == lo)
    				break;
    		}
    		if(i >= j) 
    			break;
    		double temp = a[i];
    		a[i] = a[j];
    		a[j] = temp;
    	}
    	
    	a[lo] = a[j];
    	a[j] = pivot;
    	return j;
    }

    static void quickRecursive(double a[], int lo, int hi)
    {
    	if(hi <= lo)
    	{
    		return;
    	}
    	
    	int pivotPos = partition(a, lo, hi);
    	quickRecursive(a, lo, pivotPos - 1);
    	quickRecursive(a, pivotPos + 1, hi);
    }
   
    static void merge(double[] a, double[] aux, int lo, int mid, int hi)
    {
    	for(int k = lo; k <= hi; k++)
    		aux[k] = a[k];
    	
    	int i = lo, j = mid + 1;
    	for(int k = lo; k <= hi; k++)
    	{
    		if(i > mid)
    			a[k] = aux[j++];
    		else if(j > hi)
    			a[k] = aux[i++];
    		else if(aux[j] < aux[i])                       //less(aux[j], aux[i]))
    			a[k] = aux[j++];
    		else
    			a[k] = aux[i++];
    	}
    }

    static void mergeSort(double a[], double aux[], int lo, int hi)
    {
    	if(hi <= lo)
    		return;
    	int mid = lo + (hi - lo) / 2;
    	mergeSort(a, aux, lo, mid);
    	mergeSort(a, aux, mid + 1, hi);
    	merge(a, aux, lo, mid, hi);
    }

    public static void main(String[] args)
    {	
    }

 }//end class