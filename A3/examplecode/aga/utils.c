#include <stdio.h>
#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"

int erfunc(char *s, int val)
{
  fprintf(stderr, "%s %d\n", s, val);
  exit(1);
}

void struct_cp(to, from, size)
     char *to,*from;
     int size;
{
  int s;
  s = size;
  while(s--)
    *to++ = *from++;
}

