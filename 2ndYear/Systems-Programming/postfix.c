#include "postfix.h"

/*
struct stack
{
	char ** items;
	int max_size;
	int top;
};
*/

void stack_push(struct stack * this, char * value)
{
	this->items[this->top] = value;
	this->top++;
}

char * stack_pop(struct stack * this)
{
	char * poppedCharacter;
	poppedCharacter = this->items[--this->top];

	return poppedCharacter;
}

struct stack * stack_new(int max_size)
{
	struct stack * result;

	result = malloc(sizeof(struct stack));
	result->max_size = max_size;
	result->top = 0;

	result->items = malloc(sizeof(char **) * max_size);

	return result;
}

void apply_operator(struct double_stack * stack, char * operator)
{
/*	for(int i = 0; i < stack->top; i++)
	{
		printf("%f\n", stack->items[i]);
	}
	*/
	double firstNumber = double_stack_pop(stack);
	double secondNumber = double_stack_pop(stack);
	double answer;

	if(*operator == '+')
	{
		answer = firstNumber + secondNumber;
		double_stack_push(stack, answer);
	}
	else if(*operator == '-')
	{
		answer = secondNumber - firstNumber;
		double_stack_push(stack, answer);
	}
	else if(*operator == 'X')
	{
		//printf("%f\n", firstNumber);
		//printf("%f\n", secondNumber);
		answer = firstNumber * secondNumber;
		//printf("%f\n", answer);
		double_stack_push(stack, answer);
	}
	else if(*operator == '/')
	{
		answer = secondNumber / firstNumber;
		double_stack_push(stack, answer);
	}
	else if(*operator == '^')
	{
		answer = pow(secondNumber, firstNumber);
		double_stack_push(stack, answer);
	}
	else
	{
		//printf("%f\n", atof(operator) );
		double_stack_push( stack, secondNumber );
		double_stack_push( stack, firstNumber );
		double_stack_push( stack, atof(operator) );
	}
	return;
}

double postfix_calculation(struct stack * this, int nterms)
{
	struct double_stack * stack;
	stack = double_stack_new(nterms);
	double answer;
	int i;

	for(i = 0; i < this->top; i++)
	{
		if( strlen( this->items[i] ) == 1)
		{
			apply_operator(stack, this->items[i] );
		}
		else
		{
			char * value = this->items[i];
			//printf("%f\n", atof( value ));
			double_stack_push(stack, atof( value ) );
		}
	}

	answer = double_stack_pop(stack);
	return answer;
}

// evaluate expression stored as an array of string tokens
double evaluate_postfix_expression(char ** args, int nargs) {
  // Write your code here
  int i;

  struct stack * this = stack_new(nargs);

  for(i = 0; i < nargs; i++)
  {
    stack_push( this, args[i] );
  }

  double answer = postfix_calculation(this, this->top);
  return answer;
}
