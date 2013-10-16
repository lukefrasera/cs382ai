#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"
#include <stdio.h>

void initdata()
{ /* inittialize global params, 

     popsize:   population size
     lchrom :   chromosome lenght
     maxgen :   maximum no. of generations.
     pcross :   crossover probability
     pmut   :   mutation probability           */
  int i;
  FILE *inpfl;
  
  if( (inpfl = fopen(Ifile,"r")) == NULL){
    printf("error in opening file %s \n",fname);
    exit(1);
  }
  
  printf(" Enter population size - popsize-> "); 
  fscanf(inpfl,"%d",&popsize);
  printf("\n");
  printf(" Enter chromosome length - lchrom-> "); 
  fscanf(inpfl,"%d",&lchrom);
  printf("\n");
  printf(" Enter max. generations - maxgen-> "); 
  fscanf(inpfl,"%d",&maxgen);
  printf("\n");
  printf(" Enter how many variables (0-30)-> "); 
  fscanf(inpfl,"%d",&nvars);
  printf("\n");
  printf(" Enter Crossover type xType (lchrom - MX) -> "); 
  fscanf(inpfl,"%d",&xType);
  printf("\n");
  printf(" Enter crossover prob - pcross-> "); 
  fscanf(inpfl,"%lf",&pcross);
  printf("\n");
  printf(" Enter mutation prob - pmut-> "); 
  fscanf(inpfl,"%lf",&pmut);
  printf("\n");
  printf(" Enter file name for graph output -fname-> ");
  fscanf(inpfl,"%s",fname);
  printf("Save file is %s\n",fname);
  
  printf(" Enter kind of selection (0/1) -> ");
  fscanf(inpfl,"%d", &selector);
  printf(" Selection is %d\n", selector);

  printf(" Enter mating pool choice (0/1)-> ");
  fscanf(inpfl,"%d", &seller);
  printf(" Mating pool is chosen %d\n", seller);

  printf(" Enter Cmult (1.2 - 2.0)-> ");
  fscanf(inpfl,"%lf", &Cmult);
  printf(" Cmult is %lf\n", Cmult);



  fclose(inpfl);
  printf("\n");
  randomize();
  ncross = 0; nmut = 0;
}

void initpop()
{ /* initialize a random population */
  IPTR pj;
  int j, j1, j2;
  FILE *fp;
  double f1;
  
  op = &opop[0];
  np = &npop[0];
  
  for (j = 0; j < popsize; j++){
    pj = &(opop[j]);
    for (j1 = 0; j1 < lchrom; j1++){
      pj->chrom[j1] = flip(0.5); 
    }
    pj->fitness = f1  = eval(pj->chrom);
    
    pj->parent1 = pj->parent2 = 0;
  }
}

void initreport()
{
  FILE *fp;
  int i, k;

  printf("\n\nPopulation Size(popsize)  %d\n",popsize);
  printf("Cromosome length (lchrom)  %d\n",lchrom);
  printf("Maximum num of generations(maxgen)  %d\n",maxgen);
  printf("Crossover Probability (pcross)  %lf\n",pcross);
  printf("Mutation Probability (pmut)  %lf\n",pmut);
  printf("\n\t\tFirst Generation Stats  \n\n");
  printf("Maximum Fitness  %lf\n",max);
  printf("Average Fitness  %lf\n",avg);
  printf("Minimum Fitness  %lf\n",min);

  if( (fp = fopen(fname,"a")) == NULL){
    printf("error in opening file %s \n",fname);
    exit(1);
  }else{
    raw_stat(fp, op);
    /**fprintf(fp, " %3d %lf %lf %lf \n",gen, max, avg, min);*/
    fclose(fp);
  }
  raw_stat(stdout, op);
}

void initialize()
{ /* initialize everything */
  initdata();
  printf("after initfuncs\n");
  initfuncs();
  initpop();
  printf("after initPOP\n");

  statistics(op);
  printf("after STATS\n");
  scalepop(op);

  statistics(op);
  initreport();
}


void initfuncs()
{
  int i;

  for(i = 0; i < NBFUNC; i++) {
    generation[i] = gen0;
    seltor[i] = roulette;
  }
  generation[1] = gen1;
  generation[2] = gen0_scaled;
  generation[3] = gen1_scaled;
  seltor[1] = roulette2;
  seltor[2] = scaled_roulette;
}
