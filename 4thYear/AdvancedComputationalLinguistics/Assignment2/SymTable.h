#include <vector>
#include <map>
#include <string>
//#include <stdlib.h>


#if !defined SYMTABLE_H
#define SYMTABLE_H
/*! \brief class for a symbol table mapping the strings which are the values of variables 
 * to integers representing those strings */
class SymTable {
 public:
  SymTable(void);

  int symbol_total; //!< the total number of symbols in the table
  int get_code(std::string cat);
  int check_code(std::string cat);

  std::string decode_to_symbol(int code); 
  std::map<std::string,int> the_symbols;

 private:
  std::map<int,std::string> sym_decoder;

};

#endif