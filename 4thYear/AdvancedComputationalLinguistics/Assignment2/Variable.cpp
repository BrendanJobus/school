#include "Variable.h"

Variable::Variable(void) {}

void Variable::set_name(std::string n) {
  name = n;

}

void Variable::set_range_size(int n) {
  range_size = n;
}