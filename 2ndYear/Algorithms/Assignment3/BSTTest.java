import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

//-------------------------------------------------------------------------
/**
 *  Test class for Doubly Linked List
 *
 *  @version 3.1 09/11/15 11:32:15
 *
 *  @author  Brendan Jobus
 */

@RunWith(JUnit4.class)
public class BSTTest
{
  
  //TODO write more tests here.

  @Test
  public void testHeight()
  {
	  BST<Integer,Integer> bst = new BST<Integer,Integer>();
	  assertEquals(bst.height(), -1);
	  
	  bst.put(7, 7);
      bst.put(8, 8);
      bst.put(3, 3);   
      bst.put(1, 1);   
      bst.put(2, 2);   
      bst.put(6, 6);   
      bst.put(4, 4);   
      bst.put(5, 5);
      
      assertEquals("Checking the height of tree, should be 4", bst.height(), 4);
  }
  
  @Test
  public void testMedian()
  {
	  BST<Integer,Integer> bst = new BST<Integer,Integer>();
	  assertNull(bst.median());
	  
	  bst.put(7, 7);
      bst.put(8, 8);
      bst.put(3, 3);   
      bst.put(1, 1);   
      bst.put(2, 2);   
      bst.put(6, 6);   
      bst.put(4, 4);   
      bst.put(5, 5);
      bst.put(2, 12);
      
      int answer = bst.median();
      assertEquals(4, answer);
      
      BST<Integer, Integer> bst2 = new BST<Integer, Integer>();
      bst2.put(7, 7);
      bst2.put(8, 8);
      bst2.put(3, 3);
      
      answer = bst2.median();
      assertEquals(7, answer);
      
      BST<Integer, Integer> bst3 = new BST<Integer, Integer>();
      bst3.put(7, 7);
      bst3.put(8, 8);
      bst3.put(3, 3);     
      bst3.put(6, 6);   
      bst3.put(4, 4);   
      bst3.put(5, 5);
      
      answer = bst3.median();
      assertEquals(5, answer);
  }
  
  @Test
  public void testPrintKeysInOrder()
  {
	  BST<String,Integer> bst = new BST<String,Integer>();
	  
	  	bst.put("S", 1);
	  	bst.put("E", 2);
	  	bst.put("A", 3);
	  	bst.put("C", 4);
	  	bst.put("R", 5);
	  	bst.put("H", 6);
	  	bst.put("M", 7);
	  	bst.put("X", 8);
	  
	  assertEquals(bst.printKeysInOrder(), "(((()A(()C()))E((()H(()M()))R()))S(()X()))");
  }
  
  /** <p>Test {@link BST#prettyPrintKeys()}.</p> */
      
 @Test
 public void testPrettyPrint() {
     BST<Integer, Integer> bst = new BST<Integer, Integer>();
     assertEquals("Checking pretty printing of empty tree",
             "-null\n", bst.prettyPrintKeys());
      
                          //  -7
                          //   |-3
                          //   | |-1
                          //   | | |-null
     bst.put(7, 7);       //   | |  -2
     bst.put(8, 8);       //   | |   |-null
     bst.put(3, 3);       //   | |    -null
     bst.put(1, 1);       //   |  -6
     bst.put(2, 2);       //   |   |-4
     bst.put(6, 6);       //   |   | |-null
     bst.put(4, 4);       //   |   |  -5
     bst.put(5, 5);       //   |   |   |-null
                          //   |   |    -null
                          //   |    -null
                          //    -8
                          //     |-null
                          //      -null
     
     String result = 
      "-7\n" +
      " |-3\n" + 
      " | |-1\n" +
      " | | |-null\n" + 
      " | |  -2\n" +
      " | |   |-null\n" +
      " | |    -null\n" +
      " |  -6\n" +
      " |   |-4\n" +
      " |   | |-null\n" +
      " |   |  -5\n" +
      " |   |   |-null\n" +
      " |   |    -null\n" +
      " |    -null\n" +
      "  -8\n" +
      "   |-null\n" +
      "    -null\n";
     assertEquals("Checking pretty printing of non-empty tree", result, bst.prettyPrintKeys());
     }

  
     /** <p>Test {@link BST#delete(Comparable)}.</p> */
     @Test
     public void testDelete() {
         BST<Integer, Integer> bst = new BST<Integer, Integer>();
         bst.delete(1);
         assertEquals("Deleting from empty tree", "()", bst.printKeysInOrder());
         
         bst.put(1, 1);
         bst.delete(1);
         assertEquals("Deleting from tree of size 1", "()", bst.printKeysInOrder());

         
         bst.put(7, 7);   //        _7_
         bst.put(8, 8);   //      /     \
         bst.put(3, 3);   //    _3_      8
         bst.put(1, 1);   //  /     \
         bst.put(2, 2);   // 1       6
         bst.put(6, 6);   //  \     /
         bst.put(4, 4);   //   2   4
         bst.put(5, 5);   //        \
                          //         5
         
         assertEquals("Checking order of constructed tree",
                 "(((()1(()2()))3((()4(()5()))6()))7(()8()))", bst.printKeysInOrder());
         
         bst.delete(9);
         assertEquals("Deleting non-existent key",
                 "(((()1(()2()))3((()4(()5()))6()))7(()8()))", bst.printKeysInOrder());
 
         bst.delete(8);
         assertEquals("Deleting leaf", "(((()1(()2()))3((()4(()5()))6()))7())", bst.printKeysInOrder());
 
         bst.delete(6);
         assertEquals("Deleting node with single child",
                 "(((()1(()2()))3(()4(()5())))7())", bst.printKeysInOrder());
 
         bst.delete(3);
         assertEquals("Deleting node with two children",
                 "(((()1())2(()4(()5())))7())", bst.printKeysInOrder());
         
         bst.put(2, null);
         assertEquals("Deleting node with put", "((()1(()4(()5())))7())", bst.printKeysInOrder());
     }
     
}