#pragma once
#include <math.h>
#include <string.h>

#include "core/util/floating.h"

/*

An implementation of the Lemke-Howson algorithm [1], which calculates the Nash
equilibrium for bi-matrix game.

Notes & TODOs:
1. This algorithm can only give one equilibrium at a time. Enumerating all
equilibria needs other methods, such as the one in [2].
2. The input should be two matrices of each player's payoff. For zero-sum games,
the input should be inversed for the second player.
3. There is a hardcoded max_iteration as 3000 in the algorithm, avoiding extreme
cases that the complexity is exponential. A better solution should be using
methods in [3] to improve the choice of pivots where the algorithm starts.

Cite:
1. Lemke, C.E., Howson, J.T. Jr.: Equilibrium points of bimatrix games. J Soc
Ind Appl Math 12, 413–423. (1964)
2. D. Avis, G. Rosenberg, R. Savani, and B. von Stengel: Enumeration of Nash
Equilibria for Two-Player Games. Economic Theory 42, 9-37. (2010)
3. Bruno Codenotti, Stefano De Rossi, Marino Pagan: An Experimental Analysis of
Lemke-Howson Algorithm. arXiv:0811.3247v1. (2008)

*/

// Public functions

double** create_bimatrix(int dim1, int dim2);
double* solve_equilibrium(double** bimatrix, int dim1, int dim2,
                          int startpivot = 1);
double* solve_equilibrium_positive(double** bimatrix, int dim1, int dim2,
                                   int startpivot = 1);

/*
MIT License

Copyright (c) 2021 BeeKay Koozie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Private functions & implementations

typedef struct equilibrium_ {
  int label;
  double prob;
  struct equilibrium_* next;
} equilibrium;

typedef struct eqlist_ {
  equilibrium* eq;
  struct eqlist_* next;
} eqlist;

// Adds a strategy to an existent equilibrium, or creates a new one
equilibrium* add_strategy(equilibrium*, int, double);

// Lexicographically compares two equilibria
int lex_comp(equilibrium*, equilibrium*);

// Checks if the equilibrium is the artificial equilibrium
int is_artificial(equilibrium*);

// Return the support size of the equilibrium
int eq_size(equilibrium*);

// Frees the memory occupied by an equilibrium
void free_equilibrium(equilibrium*);

// Searchs for an equilibrium in eqlist. If it not finds it,
// it adds it, and puts 0 in found
eqlist* search_add_equilibrium(eqlist*, equilibrium*, int* found);

// Frees the memory occupied by a list of equilibria
void free_eqlist(eqlist*);

// Creates the tableaus starting from the bimatrix
double*** create_systems(double** bimatrix, int dim1, int dim2);

// Adds an offset to all payoffs to have them positive
void positivize_bimatrix(double** bimatrix, int dim1, int dim2, double min);

// Creates a copy of the system
double** system_copy(double**, int);

// Tells if strategy 'strategy' is in the current tableau's base.
int get_pivot_gen(double*** tableaus, int dim1, int dim2, int strategy);

// Returns the tableau in wich the strategy is contained
int get_tableau(int dim1, int dim2, int strategy);

// Returns the column that corresponds to the given strategy
int get_column(int dim1, int dim2, int strategy);

// Memory managment functions
void free_tableaus(double*** tableaus, int dim1, int dim2);
void free_bimatrix(double** bimatrix, int dim1, int dim2);

double* lemke_howson_gen(double*** tableaus, double** bimatrix, int dim1,
                         int dim2, int pivot);

double** create_bimatrix(int dim1, int dim2) {
  double** bimatrix = (double**)malloc(sizeof(double*) * 2 * dim1);
  for (int i = 0; i < (2 * dim1); i++)
    bimatrix[i] = (double*)malloc(sizeof(double) * dim2);
  return bimatrix;
}

double* solve_equilibrium(double** bimatrix, int dim1, int dim2,
                          int startpivot) {
  double minimo = 0;
  for (int i = 0; i < dim1 + dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      if (bimatrix[i][j] < minimo) {
        minimo = bimatrix[i][j];
      }
    }
  }
  positivize_bimatrix(bimatrix, dim1, dim2, minimo);
  return solve_equilibrium_positive(bimatrix, dim1, dim2, startpivot);
}

double* solve_equilibrium_positive(double** bimatrix, int dim1, int dim2,
                                   int startpivot) {
  auto tableaus = create_systems(bimatrix, dim1, dim2);
  auto found_equilibria =
      lemke_howson_gen(tableaus, bimatrix, dim1, dim2, startpivot);
  free_tableaus(tableaus, dim1, dim2);
  return found_equilibria;
}

double* lemke_howson_gen(double*** tableaus, double** bimatrix, int dim1,
                         int dim2, int startpivot) {
  int newpivot;
  double coeff, min, val;
  int i, j, index = 0;
  int updated;

  /*
    startpivot is the index of the variable we want to pivot on. get_pivot
    determines, looking at the tableau, if we want the real strategy to enter
    the basis, or the corresponding complementary variable. This happens, for
    example, when we don't start pivoting from the artificial equilibrium, but
    from an actual one, having in basis some real strategies: if we pivot on
    them, we want the complementary variable to enter the basis.
  */
  int pivot = get_pivot_gen(tableaus, dim1, dim2, startpivot);

  for (int iterations = 0; iterations < 3000; iterations++) {
    min = 0.0;
    updated = 0;

    // ntab is the tableau we are in (0 or 1)
    int ntab = get_tableau(dim1, dim2, pivot);

    // nlines is the number of rows of the tableau we are in (dim1 or dim2)
    int nlines = ntab == 0 ? dim1 : dim2;

    // column is the tableau column wich corresponds to the variable we are
    // pivoting on
    int column = get_column(dim1, dim2, pivot);

    /*
      Minimum ratio test: we choose the index of the row in our tableau, for
      which the coefficient of the variable entering the basis is less than zero
      (if it's > 0 we cannot choose this row) and so that we minimize the ratio
      between the value of the variable in basis ( tableaus[ntab][i][1] ) and
      the coefficient of the variable entering the basis
      (tableaus[ntab][i][column])
    */
    for (i = 0; i < nlines; i++) {
      if (tableaus[ntab][i][column] >
          -Eps)  // We check that the coefficient is > 0
        continue;

      val = -(long double)tableaus[ntab][i][1] /
            tableaus[ntab][i][column];  // Ratio

      if (!updated ||
          val < (min - Eps)) {  // We update the index of the row following the
                                // minimum ratio. Updated checks if it's the
                                // first feasible row we are trying
        min = val;
        updated = 1;
        index = i;
      }
    }
    /*
      If we didn't update the index of our row, this means there isn't a row for
      which the coefficient of the variable entering the basis if less than
      zero. This cannot happen, so if we are in this condition, we got something
      wrong.
    */
    // assert(updated != 0);
    if (!updated) break;

    // Finally we choose what variable will go out of the basis
    newpivot = (int)tableaus[ntab][index][0];

    /*
      Now we know what variable will go out of the basis, so we only need to do
      two things:
      - Solve the equation we chose with the minimum ratio test, updating the
      variable in basis
      - Solve all other equations of the tableau, updating all the coefficients.

      So the first step is to update the row chosen with the minimum ratio test:
      we update tableaus[ntab][index][0], which tells what variable is in basis
      and we calculate the coefficient we will divide all other coefficient
      with.
    */

    tableaus[ntab][index][get_column(dim1, dim2, newpivot)] = -1;
    tableaus[ntab][index][0] = pivot;
    coeff = -tableaus[ntab][index][column];

    /*
       Then we update the whole row, and we put the coefficient of variable
       entering basis to zero.
    */

    for (i = 1; i < (dim1 + dim2 + 2); i++) tableaus[ntab][index][i] /= coeff;
    tableaus[ntab][index][column] = 0;

    /*
      The second step is to solve all other equations in the tableau:
      - We check if the coefficient of the variable entering in basis in this
      row is nonzero
      - If so, we update the coefficients, and set to zero the coefficient of
      the variable entering basis
    */

    for (i = 0; i < nlines; i++) {
      if (tableaus[ntab][i][column] < -Eps || tableaus[ntab][i][column] > Eps) {
        for (j = 1; j < (dim1 + dim2 + 2); j++) {
          tableaus[ntab][i][j] +=
              (long double)tableaus[ntab][i][column] * tableaus[ntab][index][j];
        }
        tableaus[ntab][i][column] = 0;
      }
    }

    /*
      Following the complementary pivoting rule, the new variable to pivot on is
      the complementary of the old variable
    */

    pivot = -newpivot;

    /*
      We stop the execution of the algorithm when either we pivot on the first
      variable we pivoted on (that strategy is leaving the basis), or on the
      complement of that variable (that complement is leaving the basis). In
      both cases, we are not in a k-almost complete equilibrium, but in an
      actual Nash equilibrium.
    */

    if (newpivot == startpivot || newpivot == -startpivot) break;
  }

  double tot1 = 0.0;
  double tot2 = 0.0;

  /*
    The only thing to do is to normalize the vector of strategy, thus obtaining
    sum of 1 for the probabilities.
  */
  for (i = 0; i < dim1; i++)
    if (tableaus[0][i][0] > 0) tot1 += tableaus[0][i][1];
  for (i = 0; i < dim2; i++)
    if (tableaus[1][i][0] > 0) tot2 += tableaus[1][i][1];

  // We create the actual equilibrium data structure with the normalized
  // strategies
  double* eq = (double*)calloc(dim1 + dim2, sizeof(double));
  for (i = 0; i < dim1; i++) {
    if (tableaus[0][i][0] > 0) {
      eq[(int)round(tableaus[0][i][0]) - 1] = tableaus[0][i][1] / tot1;
    }
  }
  for (i = 0; i < dim2; i++) {
    if (tableaus[1][i][0] > 0) {
      eq[(int)round(tableaus[1][i][0]) - 1] = tableaus[1][i][1] / tot2;
    }
  }

  return eq;
}

/*
  Adds a new strategy (with the related probability) in the equilibrium, and
  returns the updated linked list. The equilibrium itself is the head of this
  linked list.
*/

equilibrium* add_strategy(equilibrium* old, int label, double prob) {
  equilibrium* i;

  equilibrium* neweq = (equilibrium*)malloc(sizeof(equilibrium));
  neweq->label = label;
  neweq->prob = prob;
  neweq->next = 0;

  // In this case, we need to create a new equilibrium, so we simply return item
  // we just created.

  if (old == 0) return neweq;

  // The strategies are sorted, so this checks if we need to put the new element
  // in the head of the list.

  if (old->label > label) {
    neweq->next = old;
    return neweq;
  }

  for (i = old;; i = i->next) {
    // Checks if we reached the end of the list
    if (i->next == 0) {
      i->next = neweq;
      return old;
    }

    // In this case, we have to insert the item in this position
    if (i->next->label > label) {
      neweq->next = i->next;
      i->next = neweq;
      return old;
    }
  }
}

/*
  Lexicographical comparison function to compare two equilibria.
*/

int lex_comp(equilibrium* x, equilibrium* y) {
  if (x == 0 && y == 0) return 0;
  if (x != 0 && y == 0) return 1;
  if (x == 0 && y != 0) return -1;

  if (x->label < y->label)
    return -1;
  else if (x->label > y->label)
    return 1;
  else
    return lex_comp(x->next, y->next);
}

/*
  Checks whether the equilibrium we found is artificial or not.
*/

int is_artificial(equilibrium* x) { return (x == 0); }

/*
  Return the equilibrium support size
*/

int eq_size(equilibrium* x) {
  int size = 0;

  while (x != 0) {
    size++;
    x = x->next;
  }

  return size;
}

void free_equilibrium(equilibrium* eq) {
  if (!eq) return;

  free_equilibrium(eq->next);
  free(eq);
}

/*
  Adds a new equilibrium in the list, and takes care of keeping it sorted in
  lexicographical order. This is done simply inserting the element in the
  correct position, checking if we need to put it in the head or in the end of
  the list. In addiction, this function checks if the equilibrium is already in
  the list: in this case, it returns the same list, and sets found = 1. This is
  needed in order to stop the recursive implementation of all_lemke.
*/

eqlist* search_add_equilibrium(eqlist* list, equilibrium* eq, int* found) {
  eqlist* i;

  eqlist* newlist = (eqlist*)malloc(sizeof(eqlist));
  newlist->next = 0;
  newlist->eq = eq;

  if (list == 0) {
    *found = 0;
    return newlist;
  }

  if (lex_comp(list->eq, eq) > 0) {
    *found = 0;
    newlist->next = list;
    return newlist;
  } else if (lex_comp(list->eq, eq) == 0) {
    *found = 1;
    free(newlist);
    return list;
  }

  for (i = list;; i = i->next) {
    if (i->next == 0) {
      *found = 0;
      i->next = newlist;
      return list;
    }

    if (lex_comp(i->next->eq, eq) == 0) {
      *found = 1;
      free(newlist);
      return list;
    }

    if (lex_comp(i->next->eq, eq) > 0) {
      *found = 0;
      newlist->next = i->next;
      i->next = newlist;
      return list;
    }
  }
}

void free_eqlist(eqlist* lista) {
  if (!lista) return;

  free_eqlist(lista->next);
  free_equilibrium(lista->eq);
  free(lista);
}

/*
  Creates (and allocates necessary memory) the two tableaus needed by the
  algorithm, starting from the bimatrix.
*/

double*** create_systems(double** bimatrix, int dim1, int dim2) {
  int i, j;

  double*** tableaus = (double***)malloc(2 * sizeof(double**));

  // Memory allocation for the two tableaus

  tableaus[0] = (double**)malloc(dim1 * sizeof(double*));
  for (i = 0; i < dim1; i++) {
    tableaus[0][i] = (double*)calloc((2 + dim1 + dim2), sizeof(double));
  }
  tableaus[1] = (double**)malloc(dim2 * sizeof(double*));
  for (i = 0; i < dim2; i++) {
    tableaus[1][i] = (double*)calloc((2 + dim1 + dim2), sizeof(double));
  }

  /*
    Initialization of the two tableaus. The first column represents the index of
    the variable, with the convention that a negative number represents the
    slack variable associated with the corresponding positive index variable.
    The second column is the actual first column of the tableau, and represent,
    during the execution of the algorithm, the value of the variable in basis
    for that row.
  */

  for (i = 0; i < dim1; i++) {
    tableaus[0][i][0] = -i - 1.0;
    tableaus[0][i][1] = 1.0;
  }
  for (i = 0; i < dim2; i++) {
    tableaus[1][i][0] = -i - dim1 - 1.0;
    tableaus[1][i][1] = 1.0;
  }

  /*
    We now only need to copy the bimatrix in the correct cells in the tableau.
  */
  for (i = 0; i < dim1; i++) {
    for (j = (2 + dim1); j < (dim1 + dim2 + 2); j++) {
      tableaus[0][i][j] = -bimatrix[i][j - 2 - dim1];
    }
  }
  for (i = 0; i < dim2; i++) {
    for (j = (2 + dim2); j < (dim1 + dim2 + 2); j++) {
      tableaus[1][i][j] = -bimatrix[dim1 + (j - 2 - dim2)][i];
    }
  }

  return tableaus;
}

/*
  Due to some facts we assume in our implementation of the algorithm, we need
  all of the payoffs to be positive, so we simply add an offset to all payoffs,
  thus having them all > 0.
*/

void positivize_bimatrix(double** bimatrix, int dim1, int dim2, double minimo) {
  int i, j;

  for (i = 0; i < (2 * dim1); i++) {
    for (j = 0; j < dim2; j++) {
      bimatrix[i][j] -= (minimo - 1.0);
    }
  }
}

/*
  Tells if strategy 'strategy' is in base (looking at the current tableau). If
  not, it returns the same strategy, if it's in base, it returns the
  corresponding slack variable. This is needed by all_lemke, because after each
  execution of the LH algorithm we pivot on every variable from 1 to dim1+dim2,
  without knowing if that variable is in fact in base or not.
*/

int get_pivot_gen(double*** tableaus, int dim1, int dim2, int strategy) {
  int i;

  for (i = 0; i < dim1; i++) {
    if (tableaus[0][i][0] == strategy) return -strategy;
  }

  for (i = 0; i < dim2; i++) {
    if (tableaus[1][i][0] == strategy) return -strategy;
  }

  return strategy;
}

// Returns the tableau that contains the given strategy

int get_tableau(int dim1, int dim2, int strategy) {
  if (strategy > dim1 || (strategy < 0 && strategy >= -dim1)) return 0;
  if (strategy < -dim1 || (strategy > 0 && strategy <= dim1)) return 1;

  return -1;
}

// Returns the column that corresponds to the given strategy

int get_column(int dim1, int dim2, int strategy) {
  if (strategy > 0 && strategy <= dim1) {
    return (1 + dim2 + strategy);
  }
  if (strategy > 0 && strategy > dim1) {
    return (1 + dim1 + strategy - dim1);
  }
  if (strategy < 0 && strategy >= -dim1) {
    return (1 - strategy);
  }

  return (1 - strategy - dim1);
}

void free_tableaus(double*** tableaus, int dim1, int dim2) {
  int i;

  for (i = 0; i < dim1; i++) {
    free(tableaus[0][i]);
  }
  free(tableaus[0]);

  for (i = 0; i < dim2; i++) {
    free(tableaus[1][i]);
  }
  free(tableaus[1]);

  free(tableaus);
}

void free_bimatrix(double** bimatrix, int dim1, int dim2) {
  int i;

  for (i = 0; i < (2 * dim1); i++) {
    free(bimatrix[i]);
  }
  free(bimatrix);
}
