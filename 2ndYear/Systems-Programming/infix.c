#include "infix.h"

char ** changeInfixToPostfix(char ** inputs, int numberOfInputs)
{
	struct stack * this = stack_new(numberOfInputs);
	char ** outputString;
	outputString = malloc(sizeof(char **) * numberOfInputs);
	char * addToOutput;

	int i = 0;
	int j;

	for(j = 0; j < numberOfInputs; j++)
	{
		if(strcmp(inputs[j], "+") != 0 && strcmp(inputs[j], "-") != 0 && strcmp(inputs[j], "X") != 0 && strcmp(inputs[j], "/") != 0 && strcmp(inputs[j], "^") != 0 && strcmp(inputs[j], "(") != 0 && strcmp(inputs[j], ")") != 0)
		{
			outputString[i] = inputs[j];
			i++;
		}
		else if(strcmp(inputs[j], "(") == 0)
		{
			stack_push(this, inputs[j]);
		}
		else if(strcmp(inputs[j], "+") == 0 || strcmp(inputs[j], "-") == 0 || strcmp(inputs[j], "X") == 0 || strcmp(inputs[j], "/") == 0 || strcmp(inputs[j], "^") == 0)
		{
			if(strcmp(inputs[j], "+") == 0 || strcmp(inputs[j], "-") == 0)
			{
				if(this->top != 0)
				{
					while( this->top != 0 && strcmp( this->items[this->top - 1], "(") != 0 && ( strcmp( (this->items[this->top - 1]) , "X") == 0 ||
									strcmp( (this->items[this->top - 1]) , "/") == 0 || strcmp( (this->items[this->top - 1]) , "^") == 0
									|| strcmp( (this->items[this->top - 1]) , "+") == 0 || strcmp( (this->items[this->top - 1]), "-") == 0) )
					{
						addToOutput = stack_pop(this);
						outputString[i] = addToOutput;
						i++;
					}
				}
			}
			else if(strcmp(inputs[j], "X") == 0 || strcmp(inputs[j], "/") == 0)
			{
				if(this->top != 0)
				{
					while( this->top != 0 && strcmp( this->items[this->top - 1], "(") != 0 && strcmp(this->items[this->top - 1], "^") == 0 &&
							strcmp( this->items[this->top - 1], "X") == 0 && strcmp( this->items[this->top - 1], "/") == 0 )
					{
						addToOutput = stack_pop(this);
						outputString[i] = addToOutput;
						i++;
					}
				}
			}
			else
			{
			}
			stack_push(this, inputs[j]);
		}
		else if(strcmp(inputs[j], ")") == 0)
		{
			for(int k = this->top; k > 0; k++)
			{
				addToOutput = stack_pop(this);
				if( strcmp(addToOutput, "(") == 0 )
				{
					break;
				}
				else
				{
					outputString[i] = addToOutput;
					i++;
				}
			}
		}
		else
		{
		}
	}

	if(this->top > 0)
	{
		for(j = 0; j <= this->top; j++)
		{
			outputString[i] = stack_pop(this);
			i++;
		}
	}

	return outputString;
}

int newInputSize(char ** inputs, int numberOfOriginalInputs)
{
	int i;
	int numberOfInputs = 0;

	for(i = 0; i < numberOfOriginalInputs; i++)
	{
		if( strcmp(inputs[i], "(") != 0 && strcmp(inputs[i], ")") != 0)
		{
			numberOfInputs++;
		}
	}

	return numberOfInputs;
}

// evaluate expression stored as an array of string tokens
double evaluate_infix_expression(char ** args, int nargs) {
  // Write your code here
  char ** postfixInputs = changeInfixToPostfix(args, nargs);
	int i;

	int numberOfPostfixInputs;
	numberOfPostfixInputs = newInputSize(args, nargs);

	struct stack * this = stack_new(nargs);

	for(i = 0; i < numberOfPostfixInputs; i++)
	{
		stack_push( this, postfixInputs[i] );
	}

	double answer = postfix_calculation(this, numberOfPostfixInputs);
	return answer;
		//answer;
}
