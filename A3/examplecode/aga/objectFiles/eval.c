#include <stdio.h>
#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"
#include <math.h>

/*==================================================================*/


double eval_org(IPTR pj) 
                        /* eval_org obviously takes a pointer to the org 
                        Called from gen.c and init.c */
{
  int n, pid, status;
  int i;
  double val[30];
  double sum;

  sum = 0.0;
  for(i = 0; i < nvars; i++){
    val[i] = decode(pj, i+10, 10);
    sum += val[i] * val[i];
  }
  return sum;
}


double decode(IPTR pj, int index, int size)
{
  return ((double) bin_to_dec(&(pj->chrom[index]), size) - 512.0)/100.0;
}


double bin_to_dec(int *chrom, int l)
{
  int i;
  double prod;

  prod = 0.0;

  for(i = 0; i < l; i++){
    prod += (chrom[i] == 0 ? 0.0 : pow((double)2.0, (double) i));
  }
  return prod;
}

void dec_to_bin(int ad, int *barray, int size)
{
  int i, t;

  t = ad;
  for(i = 0; i < size; i++){
    barray[i] = t%2;
    t = t/2;
  }
}

