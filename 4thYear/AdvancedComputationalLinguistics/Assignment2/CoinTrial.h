#include <vector>
#include <string>
#include <map>

#if !defined COINTRIAL_H
#define COINTRIAL_H

/*! \brief represents a 'trial' of the coin tossing scenario 
 *
 *  a coin is tossed to choose between
 *  two other coins and then the chosen coin is tossed some number of times
 *
 * eg. the show method will make something like the following be displayed
 * CHOICE: A
 * TOSSES: HHHHHHHHTT H:8 T:2  
 */
class CoinTrial {
public:
  // a choice of coin to throw
  // and N outcomes of the throws of that coin
  CoinTrial();
  std::vector<int> outcomes; //!< vector of codes of the heads or tails coin toss outcomes
  int coin_choice; //!< code of the chosen coin
  void show(void);
  std::string outcomes_string(void);
  void set_ht_cnts(void); //!< makes the heads and tails counts based on 'outcomes'
  int ht_cnts[2]; //!< stores counts heads and tails
};
#endif