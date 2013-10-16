EXTERN INDIVIDUAL opop[2*MAXPOP], npop[2*MAXPOP];
EXTERN IPTR op, np;

EXTERN double SUMfitness, max, avg, min, bigmax, scaled_sumfitness;
EXTERN double scale_constA, scale_constB, smax, smin, Cmult;
EXTERN double  pcross, pmut, randomseed;
EXTERN int xType;
EXTERN int popsize, gen, lchrom, ncross, nmut;
EXTERN int rank[2*MAXPOP];

EXTERN int maxi, mini, biggen, bigind;
EXTERN int nvars, maxgen;

EXTERN char fname[30];
EXTERN char ph_name[30];
EXTERN char Ifile[30];
EXTERN double Maxconst;

EXTERN int (*seltor[NBFUNC])(IPTR, double, int);
EXTERN int (*generation[NBFUNC])(IPTR, IPTR, int);

EXTERN int selector, seller;

