#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"


int gen0(IPTR oldpop, IPTR newpop, int t)
{

  int i, p1, p2, c;

  IPTR pj, pjplus1, om1, om2;

  for(i = 0; i < popsize; i += 2){

    p1 = seltor[seller](oldpop, SUMfitness, popsize);
    p2 = seltor[seller](oldpop, SUMfitness, popsize);


    pj = &(newpop[i]);
    pjplus1 = &(newpop[i+1]);
    om1 = &(oldpop[p1]);
    om2 = &(oldpop[p2]);

    cross(om1, om2, pj, pjplus1);

    pj->fitness = eval(pj->chrom); 
    pj->parent1 = p1;
    pj->parent2 = p2;

    
    pjplus1->fitness = eval(pjplus1->chrom); 
    pjplus1->parent1 = p2;
    pjplus1->parent2 = p1;

  }

}


int gen1(IPTR oldpop, IPTR newpop, int t)
{

  int i, p1, p2, c;

  IPTR pj, pjplus1, om1, om2;

  for(i = popsize; i < 2*popsize; i += 2){

    p1 = seltor[seller](oldpop, SUMfitness, popsize);
    p2 = seltor[seller](oldpop, SUMfitness, popsize);
    /**
    fprintf(stderr, "selecting mates %d %d\n", p1, p2);
    **/
    pj = &(oldpop[i]);
    pjplus1 = &(oldpop[i+1]);
    om1 = &(oldpop[p1]);
    om2 = &(oldpop[p2]);

    cross(om1, om2, pj, pjplus1);

    pj->fitness = eval(pj->chrom); 
    pj->parent1 = p1;
    pj->parent2 = p2;

    pjplus1->fitness = eval(pjplus1->chrom); 
    pjplus1->parent1 = p2;
    pjplus1->parent2 = p1;
    
  }

  halve(oldpop, newpop);
}


int gen0_scaled(IPTR oldpop, IPTR newpop, int t)
{

  int i, p1, p2, c;

  IPTR pj, pjplus1, om1, om2;

  scalepop(oldpop);
  for(i = 0; i < popsize; i += 2){

    p1 = seltor[seller](oldpop, scaled_sumfitness, popsize);
    p2 = seltor[seller](oldpop, scaled_sumfitness, popsize);


    pj = &(newpop[i]);
    pjplus1 = &(newpop[i+1]);
    om1 = &(oldpop[p1]);
    om2 = &(oldpop[p2]);

    cross(om1, om2, pj, pjplus1);

    pj->fitness = eval_org(pj->chrom); 
    pj->parent1 = p1;
    pj->parent2 = p2;

    
    pjplus1->fitness = eval(pjplus1->chrom); 
    pjplus1->parent1 = p2;
    pjplus1->parent2 = p1;

  }

}

int gen1_scaled(IPTR oldpop, IPTR newpop, int t)
{

  int i, p1, p2, c;

  IPTR pj, pjplus1, om1, om2;

  scalepop(oldpop);

  for(i = popsize; i < 2*popsize; i += 2){

    p1 = seltor[seller](oldpop, scaled_sumfitness, popsize);
    p2 = seltor[seller](oldpop, scaled_sumfitness, popsize);
    /**
    fprintf(stderr, "selecting mates %d %d\n", p1, p2);
    **/
    pj = &(oldpop[i]);
    pjplus1 = &(oldpop[i+1]);
    om1 = &(oldpop[p1]);
    om2 = &(oldpop[p2]);

    cross(om1, om2, pj, pjplus1);

    pj->fitness = eval(pj->chrom); 
    pj->parent1 = p1;
    pj->parent2 = p2;

    pjplus1->fitness = eval(pjplus1->chrom); 
    pjplus1->parent1 = p2;
    pjplus1->parent2 = p1;
    
  }

  halve(oldpop, newpop);
}
