//
// CSU33014 Annual Exam, May 2021
// Question 1
//

// Please examine version each of the following routines with names
// starting 'routine_'. Where the routine can be vectorized, please
// replace the corresponding 'vectorized' version using SSE vector
// intrinsics. Where it cannot be vectorized please explain why.

// To illustrate what you need to do, routine_0 contains a
// non-vectorized piece of code, and vectorized_0 shows a
// corresponding vectorized version of the same code.

// Note that to simplify testing, I have put a copy of the original
// non-vectorized code in the vectorized version of the code for
// routines 1 to 6. This allows you to easily see what the output of
// the program looks like when the original and vectorized version of
// the code produce equivalent output.

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".


#include <immintrin.h>
#include <stdio.h>

#include "csu33014-annual-q1-code.h"

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void routine_0(float * restrict a, float * restrict b, float * restrict c) {
  for (int i = 0; i < 1024; i++ ) {
    a[i] = b[i] * c[i];
  }
}

// here is a vectorized solution for the example above
void vectorized_0(float * restrict a, float * restrict b, float * restrict c) {
  __m128 a4, b4, c4;
  
  for (int i = 0; i < 1024; i = i+4 ) {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    a4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&a[i], a4);
  }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float routine_1(float * restrict a, float * restrict b, int size) {
  float sum_a = 0.0;
  float sum_b = 0.0;
  
  for ( int i = 0; i < size; i++ ) {
    sum_a = sum_a + a[i];
    sum_b = sum_b + b[i];
  }
  return sum_a * sum_b;
}

// in the following, size can have any positive value
float vectorized_1(float * restrict a, float * restrict b, int size) {
  float sum_a = 0.0;
  float sum_b = 0.0;

  __m128 a4, b4, sum_a4, sum_b4;
  sum_a4 = _mm_setzero_ps();
  sum_b4 = _mm_setzero_ps();

  // we're getting the number of cycles that we are going to have to do at the end, this is to ensure that our data is aligned properly
  int leftOverCycles = size % 4;
  int i;

  // loop through size-leftOverCycles times, the last few elements that we dont do here we do after
  for ( i = 0; i < size-leftOverCycles; i = i+4 ) {
    a4 = _mm_load_ps(&a[i]);
    b4 = _mm_load_ps(&b[i]);

    // add together all of the a's and b's, we will add the four columns that constitute sum_a and sum_b outside of the for loop
    sum_a4 = _mm_add_ps(sum_a4,a4);
    sum_b4 = _mm_add_ps(sum_b4,b4);
  }

  float temp1[4];
  _mm_store_ps(&temp1[0], sum_a4);
  float temp2[4];
  _mm_store_ps(&temp2[0], sum_b4);

  // add the values that were in sum_a4 and sum_b4, into sum_a and sum_b
  sum_a = temp1[0] + temp1[1] + temp1[2] + temp1[3];
  sum_b = temp2[0] + temp2[1] + temp2[2] + temp2[3];

  // using the same i as before, we can continue with the last few elements that we haven't done, this will be leftOverCycles number of elements
  for (; i < size; i++) {
    sum_a = sum_a + a[i];
    sum_b = sum_b + b[i];
  }

  return sum_a * sum_b;
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void routine_2(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = 1.5379 - (1.0/b[i]);
  }
}


void vectorized_2(float * restrict a, float * restrict b, int size) {
  __m128 a4, b4, subbedFrom, ones, divided;

  // setting up the constants into vectors
  subbedFrom = _mm_set1_ps(1.5379);
  ones = _mm_set1_ps(1.0);

  // dealing with unaligned data the same way as before
  int leftOverCycles = size % 4;
  int i;

  for (i = 0; i < size-leftOverCycles; i = i+4) {
    b4 = _mm_load_ps(&b[i]);
    // we just do the operations sequentially since there is no interaction between the elements of the array
    divided = _mm_div_ps(ones, b4);
    a4 = _mm_sub_ps(subbedFrom, divided);
    _mm_store_ps(&a[i], a4);
  }

  for (; i < size; i++) {
    a[i] = 1.5379 - (1.0/b[i]);
  }
}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void routine_3(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] < b[i] ) {
      a[i] = b[i];
    }
  }
}


void vectorized_3(float * restrict a, float * restrict b, int size) {
  __m128 a4, b4, comp;

  int leftOverCycles = size % 4;
  int i;

  for (i = 0; i < size-leftOverCycles; i = i+4) {
    a4 = _mm_load_ps(&a[i]);
    b4 = _mm_load_ps(&b[i]);

    // comparing a4 and b4
    comp = _mm_cmplt_ps(a4, b4);
    
    // by anding b4 with the comp and andnoting a4 with comp, we can get two values, that when or'd together
    // give us a set that represents a[i] = b[i] iff a[i] < b[i]
    b4 = _mm_and_ps(b4, comp);
    a4 = _mm_andnot_ps(comp, a4);
    _mm_store_ps(&a[i], _mm_or_ps(a4, b4));
  }

  for (; i < size; i++) {
    if (a[i] < b[i]) {
      a[i] = b[i];
    }
  }
}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void routine_4(float * restrict a, float * restrict b, float * restrict c) {
  for ( int i = 0; i < 2048; i = i+2  ) {
    a[i] = b[i]*c[i+1] + b[i+1]*c[i];
    a[i+1] = b[i]*c[i] - b[i+1]*c[i+1];
  }
}


void vectorized_4(float * restrict a, float * restrict b, float * restrict  c) {
  // the b4_first represents the first occurence of a b in the equations, b4_second is the second b, that can be found in the second  product
  // i.e. b4_first = [b[i], b[i], b[i+2], b[i+2]] and b4_second = [b[i+1], b[i+1], b[i+2], b[i+2]], where element 1 corresponds to variables from
  // equation a[i], element 2 corresponds to variables from equation a[i+1] and so on, the same is true for c4_first and second, and for m4_first and
  // second, m4 is the actual product
  __m128 a4, b4, c4, b4_first, b4_second, c4_first, c4_second, m4_first, m4_second;
  
  __m128 signs = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);

  for (int i = 0; i < 2048; i = i+4) {
    b4 = _mm_load_ps(&b[i]);
    c4 = _mm_load_ps(&c[i]);

    // Shuffleing here will place our arrays, for example, like this, b4_first = [b[i+2], b[i+2], b[i], b[i]], since sse works with values in reverse, 
    // this is equivalent of a vector [b[i], b[i], b[i+2], b[i+2]]
    b4_first = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(2, 2, 0, 0));
    c4_first = _mm_shuffle_ps(c4, c4, _MM_SHUFFLE(2, 3, 0, 1));
    b4_second = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(3, 3, 1, 1));
    c4_second = _mm_shuffle_ps(c4, c4, _MM_SHUFFLE(3, 2, 1, 0));

    // multiplying the corresponding b4 and c4's together will give us the two products that make up each equation, however, the signs are not correct on
    // the second one, and and since we cannot add and subtract each element independently, we multiply by our signs variable to correct this
    m4_first = _mm_mul_ps(b4_first, c4_first);
    m4_second = _mm_mul_ps(b4_second, c4_second);
    m4_second = _mm_mul_ps(m4_second, signs);
    // after correcting the signs we can now add them together and move on
    a4 = _mm_add_ps(m4_first, m4_second);
    _mm_store_ps(&a[i], a4);
  }

  /*
  /   essentially, our vectors look like this after shuffling(the operations are not done after shuffling, they are there for clarity):
  /   vector          b4_first  c4_first        b4_second   c4_second
  /   data in vector  b[i]    * c[i+1]     +    b[i+1]    * c[i]
  /   data in vector  b[i]    * c[i]       +    b[i+1]    * c[i+1]
  /   data in vector  b[i+2]  * c[i+3]     +    b[i+3]    * c[i+2]
  /   data in vector  b[i+2]  * c[i+2]     +    b[i+3]    * c[i+3]
  */
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
int routine_5(unsigned char * restrict a, unsigned char * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] != b[i] )
      return 0;
  }
  return 1;
}

int vectorized_5(unsigned char * restrict a, unsigned char * restrict b, int size) {
  // we're going to treat the characters as unsigned ints
  __m128i_u a4, b4;

  int leftOverCycles = size % 16;
  int i;

  for (i = 0; i < size-leftOverCycles; i = i+16) {
    // load the characters in as unsigned 128 bit ints
    a4 = _mm_lddqu_si128((const __m128i_u*) &a[i]);
    b4 = _mm_lddqu_si128((const __m128i_u*) &b[i]);
    // we use movemask to turn the output of epi8 into the form that was shown in the lecture slides 4, slide 11
    // we just have to check to see if ANY of the numbers are not equal, so we can just do a != with all ones, which is
    // the output should they all be equal
    if (_mm_movemask_epi8(_mm_cmpeq_epi8(a4, b4)) != 0xFFFF) {
      return 0;
    }
  }

  for (; i < size; i++) {
    if (a[i] != b[i]) {
      return 0;
    }
  }
  return 1;
}

/********************* routine 6 ***********************/

void routine_6(float * restrict a, float * restrict b, float * restrict c) {
  a[0] = 0.0;
  for ( int i = 1; i < 1023; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}

void vectorized_6(float * restrict a, float * restrict b, float * restrict c) {
  a[0] = 0.0;

  // basically, the algorithm is, add the data from the element that we are currently on, 
  // aswell as the data in the elements immediatley infront and behind it
  // we are also resetting on c everytime, so we only use the first three elements of c, so we load them in first

  __m128 a4, b4, c4, product4;
  c4 = _mm_load_ps(&c[0]);
  float product[4] = {0,0,0,0};

  for (int i = 1; i < 1023; i++) {
    // since we are interested in the 3 elements immediately after i-1, we load in from there, we have to use loadu this time as our data isnt
    // correctly aligned
    b4 = _mm_loadu_ps(&b[i - 1]);
    product4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&product[0], product4);
    // we then add the products together into a[i]
    a[i] = product[0] + product[1] + product[2];
  }
  a[1023] = 0.0;
}