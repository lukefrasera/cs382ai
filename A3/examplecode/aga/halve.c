#include <stdio.h>
#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"

void halve(opop,pop)
individual *opop,*pop;
{

  int size,i;

  size = sizeof(individual);
  sort(rank,opop,2*popsize);
  for(i =0;i<popsize;i++){
    struct_cp((char *) &pop[i], (char *) &opop[rank[i]],size);
  }
}
