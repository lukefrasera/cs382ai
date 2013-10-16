#include <math.h>
#include <stdio.h>

#include "const.h"
#include "type.h"
#include "exfunc.h"

#undef EXTERN
#define EXTERN    
/* EXTERN defined as extern everywhere except in this file so
   storage is allocated here ONCE  */

#include "extern.h"

main(argc,argv)
     int argc;
     char *argv[];
{
  IPTR tmp;
  
  gen = 0;
  if(argc != 2) erfunc("Usage: ga <inputfile name> ", argc);
  strcpy(Ifile, argv[1]);
/*  strcpy(ph_name, argv[2]);
  Cmult = (double)(atoi(argv[3]));*/

  initialize();
  while(gen < maxgen){
    gen++;
    generation[selector](op, np, gen);
    statistics(np);
    report(gen, np);
    tmp = op;
    op = np;
    np = tmp;
  }
}
