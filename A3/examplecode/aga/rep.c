#include <stdio.h>
#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"

void raw_stat();
void hole_dim();

void report(gen, pop)
     int gen;
     IPTR pop;
{ /* report generations stats */
  FILE *fp;
    
  if( (fp = fopen(fname,"a")) == NULL){
    printf("error in opening file %s \n",fname);
    exit(1);
  }else{
    raw_stat(fp, pop);
    fclose(fp);
  }
  raw_stat(stdout, pop);


}

void raw_stat(FILE *fp, IPTR pop)
{
  fprintf(fp," %3d, %lf, %lf, %lf, min=%5.2lf, smin=%5.2lf, ", gen, max, smax, avg, min, smin);
  fprintf(fp," %3d, %lf, %3d, ", biggen, bigmax, bigind);
  fprintf(fp," %lf\n", pop[maxi].fitness);
}

