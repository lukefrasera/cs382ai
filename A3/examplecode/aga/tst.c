#define RANCONST 1e-9

extern int flip(), rnd(), randomize();

double randomseed;

main(int argc, char *argv[])
{
  int i;

  randomize();
  for(i = 0; i < 10; i++){
    printf("flip %d  rnd(0, 10) %d\n", flip(0.5), rnd(0, 10));
  }
}
