#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"


int roulette(pop, sumfitness, popsize)
     IPTR pop;
     double sumfitness;
     int popsize;
{ 

  /* select a single individual by roulette wheel selection */
  
  double rand,partsum;
  int j,i;

  partsum = 0.0; j = 0;
  rand = f_random() * sumfitness; 
  /* This sumfitness is actually SUMfitness
     passed to it by generation()  */
  
  j = -1;
  do{
    j++;
    partsum += pop[j].fitness;
  } while (partsum < rand && j < popsize - 1) ;
  
  return j;
}

int scaled_roulette(pop, scaled_sumfitness, popsize)
     IPTR pop;
     double scaled_sumfitness;
     int popsize;
{ 

  /* select a single individual by roulette wheel selection */
  
  double rand,partsum;
  int j,i;

  partsum = 0.0; j = 0;
  rand = f_random() * scaled_sumfitness;  /* scaled selection */
  
  j = -1;
  do{
    j++;
    partsum += pop[j].scaled_fitness;
  } while (partsum < rand && j < popsize - 1) ;
  
  return j;
}

int roulette2(IPTR pop, double sumfitness, int popsize)
{
  return rnd(0, popsize-1);
}


void sort(int *rk, IPTR pop,int size)
{
  int i,j;
  int temp,sorted;
  double fit[2*MAXPOP],temp1;
  
  for(i=0;i<size;i++){
    rk[i] = i;
    fit[i] = pop[i].fitness;
  }
  j = 0;
  sorted = 0;
  while(!sorted){
    sorted = 1;
    for(i=0;i<size-1;i++){
      if(fit[i] < fit[i+1]){
        temp = rk[i];
        rk[i] = rk[i+1];
        rk[i+1] = temp;
        temp1 = fit[i];
        fit[i] = fit[i+1];
        fit[i+1] = temp1;
        sorted = 0;
      }
    }
  }

}

void nosort(rk)
     int *rk;
{
  int i;
  for(i = 0; i < popsize; i++) rk[i] = i;
}

