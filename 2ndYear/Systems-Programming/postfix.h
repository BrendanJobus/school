#ifndef __POSTFIX_H__
#define __POSTFIX_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "stack.h"

struct stack {
  char ** items;
  int max_size;
  int top;
};

void stack_push(struct stack * this, char * value);

char * stack_pop(struct stack * this);

struct stack * stack_new(int max_size);

void apply_operator(struct double_stack * stack, char * operator);

double postfix_calculation(struct stack * this, int nterms);

double evaluate_postfix_expression(char ** expr, int nterms);

#endif
