// CSU33014 Annual Exam, May 2021
// Question 1
//
// For instructions see csu33014-annual-q1-code.c
//
// PLEASE DO NOT CHANGE ANY CODE IN THIS FILE

#ifndef csu33014_annual_q1_code_H
#define csu33014_annual_q1_code_H

void routine_0(float * restrict a, float * restrict b,
		    float * restrict c);
void vectorized_0(float * restrict a, float * restrict b,
		       float * restrict c);
float routine_1(float * restrict a, float * restrict b,
		     int size);
float vectorized_1(float * restrict a, float * restrict b,
			int size);
void routine_2(float * restrict a, float * restrict b, int size);
void vectorized_2(float * restrict a, float * restrict b, int size);
void routine_3(float * restrict a, float * restrict b, int size);
void vectorized_3(float * restrict a, float * restrict b, int size);
void routine_4(float * restrict a, float * restrict b,
		    float * restrict c);
void vectorized_4(float * restrict a, float * restrict b,
		       float * restrict  c);
int routine_5(unsigned char * restrict a,
		    unsigned char * restrict b, int size);
int vectorized_5(unsigned char * restrict a,
		       unsigned char * restrict b, int size);
void routine_6(float * restrict a, float * restrict b,
		    float * restrict c);
void vectorized_6(float * restrict a, float * restrict b,
		       float * restrict c);

#endif
