#include "SymTable.h"
#include "CoinTrial.h"
#include "Variable.h"
#include "prob_tables_coin.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>

Variable chce; /*!< \brief encapsulates the 'disk' being used to choose a type of coin.
		 contains symbol table for its possible outcomes (eg. A vs B) */ 
Variable ht; /*!< \brief encapsulates a heads-vs-tails coin toss.
	       contain symbol table for its possible outcomes (eg. H vs T) */

std::vector<CoinTrial> data; /*!< \brief represents all the data 
                             as a vector of CoinTrial objects */

void process_corpus(std::string file); /*!< \brief turn contents of filename into data 
                                   * each line of the file is represented by a CoinTrial object */

//! splits a line into into tokens using white-space as separator
void tokenize(std::string line, std::vector<std::string>& words);

int main(int argc, char **argv) {
 
  std::string filename;
  filename = std::string(argv[1]);

  // after this call the vector data corresponds to the contents of filename
  // each line of the file is represented by a CoinTrial object
  process_corpus(filename);

  std::cout << "read all data\n";
  std::cout << "total amount of extracted data is: " << data.size() << std::endl;

  // just show the data
  for(unsigned int d=0; d < data.size(); d++) {
    data[d].show();
  }

  // hard wire some probs
  chce_probs[0] = 0.2; chce_probs[1] = 0.8;
  ht_probs[0][0] = 0.4;   ht_probs[0][1] = 0.6;
  ht_probs[1][0] = 0.3;   ht_probs[1][1] = 0.7;


  std::vector<std::vector<double> > gamma;
  /* purpose of gamma[d][z] is that it should be cond prob of (chce=z) given 
    the visible coin toss outcomes of the d-th data item */

  // this just makes into a table of the right size
  gamma.resize(data.size());
  for(int dn=0; dn < data.size(); dn++) {
    gamma[dn].resize(2);
  }

  // BEGIN INSERT: at present all gamma's entries are 0
  // insert code here to set the content of gamma based on data
  // and the probs in chce_probs and ht_probs so that gamma[d][z] does
  // give cond prob (chce=z) given the visible coin toss outcomes of
  // the d-th data item
  // feel free to add additional helper functions to this file also
  // note that can definitely complete this *without* modifying any other files

  // This could've been three lines, but I feel this is more expressive
  // Coin and gammaCalc only here to be obvious what work I did, should be at the top of the file
  enum Coin {
    a = 0,
    b = 1,
  };

  auto gammaCalc = [] (int coin, int dh, int dt) {
    return (
      ( chce_probs[coin] * pow(ht_probs[coin][0], dh) * pow(ht_probs[coin][1],dt) ) / ( (chce_probs[0] * pow(ht_probs[0][0], dh) * pow(ht_probs[0][1],dt) ) + (chce_probs[1] * pow(ht_probs[1][0], dh) * pow(ht_probs[1][1], dt) ) )
    );
  };

  for(int d = 0; d < data.size(); d++) {
    int dh = data[d].ht_cnts[a], dt = data[d].ht_cnts[b];
    gamma[d][a] = gammaCalc(a, dh, dt);
    gamma[d][b] = gammaCalc(b, dh, dt);
  }

  // END INSERT
  // show gamma
  for(int dn=0; dn < data.size(); dn++) {
    std::cout << dn+1 << ": ";
    for(int z=0; z < 2; z++) {
      std::cout << chce.table.decode_to_symbol(z) << "(" << gamma[dn][z] << ")   ";
    }
    std::cout << std::endl;
  }
}

void process_corpus(std::string afile) {

  std::ifstream f;
  f.open(afile.c_str());
  if(!f) {
    std::cout << "prob opening " << afile << std::endl;
    exit(1);
  }
  else {
    std::cout << "processing " << afile << std::endl;
  }

  std::vector<std::string> raw_line;
  CoinTrial line_rep;
  std::string line = "";
 
  while(getline(f,line)) {
    std::vector<std::string> pre_words;
    tokenize(line,raw_line);
    line_rep.outcomes.clear();
    // make line_rep from raw_line
    // then push to data
    std::string word;
    for(unsigned int i=0; i < raw_line.size(); i++) {
      word = raw_line[i];
      if(i == 0) {
	line_rep.coin_choice = (chce.table.get_code(word));
      }
      else {
	line_rep.outcomes.push_back(ht.table.get_code(word));
      }
    }
    data.push_back(line_rep);
  }

  for(unsigned int d=0; d < data.size(); d++) {
     data[d].set_ht_cnts();
  }

  f.close();
}

void tokenize(std::string line, std::vector<std::string>& words) {
  /* empty the words vector */
  words.clear();

  if(line == "") {
    return;
  }

  /* update the words vector from line */
  std::string::iterator word_itr, space_itr;
  std::string token = "";
  word_itr = line.begin();             /* word_itr is beginning of line */
  space_itr = find(word_itr,line.end(),' '); /* find space */

  while(space_itr != line.end()) {
    token = std::string(word_itr,space_itr);
    words.push_back(token);

    word_itr = space_itr+1;
    space_itr = find(word_itr,line.end(),' '); /* find space */
  }

  token = std::string(word_itr,space_itr);
  words.push_back(token);

  return;
}