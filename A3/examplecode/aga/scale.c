#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"


void scalepop(pop)
     IPTR pop;
{ 

  /* linearly scale the population */

  IPTR pj;
  int i;
  
  find_coeffs(pop);

  scaled_sumfitness = 0.0;
  for(i = 0; i < popsize; i++){
    pj = &pop[i];
    pj->scaled_fitness = scale_constA * pj->fitness + scale_constB;
    scaled_sumfitness += pj->scaled_fitness;
  }
}

void find_coeffs(IPTR pop)
{
  /* find coeffs scale_constA and scale_constB for linear scaling according to 
     f_scaled = scale_constA * f_raw + scale_constB */  

  double d;

  if(min > (Cmult * avg - max)/(Cmult - 1.0)) { /* if nonnegative smin */
    d = max - avg;
    scale_constA = (Cmult - 1.0) * avg / d;
    scale_constB = avg * (max - Cmult * avg)/d;
  } else {  /* if smin becomes negative on scaling */
    d = avg - min;
    scale_constA = avg/d;
    scale_constB = -min * avg/d;
  }
  if(d < 0.00001 && d > -0.00001) { /* if converged */
    scale_constA = 1.0;
    scale_constB = 0.0;
  }
}


