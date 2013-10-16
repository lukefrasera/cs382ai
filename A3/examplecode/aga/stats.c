#include <stdio.h>
#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"
#include <math.h>

void statistics(IPTR pop)
{ /* calculate population stats */
  int size, i, j, s;
  IPTR pj;

  smax = pop[0].scaled_fitness;
  smin = smax;
  scaled_sumfitness = smax;

  SUMfitness = pop[0].fitness;
  max = SUMfitness;
  min = SUMfitness;
  maxi = mini = 0;
  for(j = 1; j < popsize;j++){
    pj = &(pop[j]);
    SUMfitness += pj->fitness; 
    scaled_sumfitness += pj->scaled_fitness;
    if (max < pj->fitness) {
      max = pj->fitness;   maxi = j;
    }
    if (min > pj->fitness){
      min = pj->fitness;   mini = j;
    }
  }
  smax = scale_constA * max + scale_constB;
  smin = scale_constA * min + scale_constB;
  avg = SUMfitness / (double) popsize;
  if(bigmax < max) {
    bigmax = max; biggen = gen; bigind = maxi;
  }
}
