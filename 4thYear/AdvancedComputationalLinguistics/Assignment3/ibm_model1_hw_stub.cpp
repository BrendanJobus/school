#include<vector>
#include<string>
#include<iostream>
#include<iomanip>
#include<cmath>

using namespace std;

// want to represents vocab items by integers because then various tables 
// need by the IBM model and EM training can just be represented as 2-dim 
// tables indexed by integers

// the following #defines, defs of VS, VO, S, O, and create_vocab_and_data()
// are set up to deal with the specific case of the two pair corpus
// (la maison/the house)
// (la fleur/the flower)

// S VOCAB
#define LA 0
#define MAISON 1
#define FLEUR 2
// O VOCAB
#define THE 0
#define HOUSE 1
#define FLOWER 2

#define VS_SIZE 3
#define VO_SIZE 3
#define D_SIZE 2

vector<string> VS(VS_SIZE); // S vocab: VS[x] gives Src word coded by x 
vector<string> VO(VO_SIZE); // O vocab: VO[x] gives Obs word coded by x

vector<vector<int> > S(D_SIZE); // all S sequences; in this case 2
vector<vector<int> > O(D_SIZE); // all O sequences; in this case 2

// sets S[0] and S[1] to be the int vecs representing the S sequences
// sets O[0] and O[1] to be the int vecs representing the O sequences
void create_vocab_and_data(); 

// functions which use VS and VO to 'decode' the int vecs representing the 
// Src and Obs sequences
void show_pair(int d);
void show_O(int d); 
void show_S(int d);

// My function definitions
void initialize();
void show_probs(string, float[VO_SIZE][VS_SIZE]);
void gamma_d(vector<int>, vector<int>);
float normalise();
void reset_e();

// My structures
vector<int> wordsO = {THE, HOUSE, FLOWER};
vector<int> wordsS = {LA, MAISON, FLEUR};

// Array that holds the sum of gamma_d's
float e[VO_SIZE][VS_SIZE] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
// The translation probabilites
float tr[VO_SIZE][VS_SIZE] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

int main() {
  create_vocab_and_data();

  // guts of it to go here
  // you may well though want to set up further global data structures
  // and functions which access them 
  initialize();
  show_probs("initial trans probs are", tr);
  for(int i = 0; i < 50; i++) {
    for(int seq = 0; seq < S.size(); seq++) {
      // s is the current source sentence and o is the current origin sentence
      vector<int> s = S.at(seq);
      vector<int> o = O.at(seq);

      gamma_d(o, s);
    }
    show_probs("unnormalised counts in iteration " + to_string(i) + " are", e);
    normalise();
    show_probs("after iteration " + to_string(i) + " trans probs are", tr);
    // need to reset e cause we're adding to it every iteration assuming we're starting new
    reset_e();
  }
}

// My functions
void initialize() {
  // The initial probs are the probability that any o translates to any s and all are uniform,
  // which means that the probability is 1 / #(oj) for each si
  for(auto si : wordsS) {
    auto numOfOs = 0;
    for(auto oj : wordsO) {
      numOfOs += 1;
    }
    for(auto oj : wordsO) {
      tr[oj][si] = 1.0 / numOfOs;
    }
  }  
}

// Will print out a message and a words oj, a word si and the probability of it being the correct translation
// as well as formating the code so that everything is inline
void show_probs(string output_string, float probs[VO_SIZE][VS_SIZE]) {
  cout << output_string << endl;
  for(auto s : wordsS) {
    for(auto o : wordsO) {
      cout << VO[o] + string((7 - VO[o].size()), ' ') + VS[s] + string((7 - VS[s].size()), ' ') << probs[o][s] << endl;
    }
  }
}

// Calculates all gammas for some d and updates the matrix e with these values
void gamma_d(vector<int> o, vector<int> s) {
  for(auto oj : o) {
    // Getting E[tr(oj, si')]
    auto sum = 0.0;
    for(auto si : s) {
      sum += tr[oj][si];
    }
    // then for all si
    // getting Yd(oj, si) = tr(oj, si) / E[tr(oj, si')]
    for(auto si : s) {
      e[oj][si] += tr[oj][si] / sum;
    }
  }
}

// normalises the counts in e and places these values into tr
float normalise() {
  for(auto s : wordsS) {
    auto denom = 0.0;
    for(auto o : wordsO) {
      denom += e[o][s];
    }
    for(auto o : wordsO) {
      tr[o][s] = e[o][s] / denom;
    }
  }
}

// Resets e as for each iteration, we are adding to e, under the assumption that e currently only holds data
// related to the current iteration
void reset_e() {
  for(auto o : wordsO) {
    for(auto s : wordsS) {
      e[o][s] = 0;
    }
  }
}

// Default functions
void create_vocab_and_data() {

  VS[LA] = "la";
  VS[MAISON] = "maison";
  VS[FLEUR] = "fleur";

  VO[THE] = "the";
  VO[HOUSE] = "house";
  VO[FLOWER] = "flower";

  cout << "source vocab\n";
  for(int vi=0; vi < VS.size(); vi++) {
    cout << VS[vi] << " ";
  }
  cout << endl;
  cout << "observed vocab\n";
  for(int vj=0; vj < VO.size(); vj++) {
    cout << VO[vj] << " ";
  }
  cout << endl;

  // make S[0] be {LA,MAISON}
  //      O[0] be {THE,HOUSE}
  S[0] = {LA,MAISON};
  O[0] = {THE,HOUSE};

  // make S[1] be {LA,FLEUR}
  //      O[1] be {THE,FLOWER}
  S[1] = {LA,FLEUR};
  O[1] = {THE,FLOWER};

  for(int d = 0; d < S.size(); d++) {
    show_pair(d);
  }
}

void show_O(int d) {
  for(int i=0; i < O[d].size(); i++) {
    cout << VO[O[d][i]] << " ";
  }
}

void show_S(int d) {
  for(int i=0; i < S[d].size(); i++) {
    cout << VS[S[d][i]] << " ";
  }
}

void show_pair(int d) {
  cout << "S" << d << ": ";
  show_S(d);
  cout << endl;
  cout << "O" << d << ": ";
  show_O(d);
  cout << endl;
}
