#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"

void cross(p1, p2, c1, c2)
     IPTR p1, p2, c1, c2;
{
/* p1,p2,c1,c2,m1,m2,mc1,mc2 */
  int *pi1,*pi2,*ci1,*ci2;
  int xp, i;

  pi1 = p1->chrom;
  pi2 = p2->chrom;
  ci1 = c1->chrom;
  ci2 = c2->chrom;
  
  if(flip(pcross)){
    ncross++;
    xp = rnd(0, lchrom - 1);
    for(i = 0; i < xp; i++){
      ci1[i] = muteX(pi1[i],pi2[i]);
      ci2[i] = muteX(pi2[i],pi1[i]);
    }
    for(i = xp; i < lchrom; i++){
      ci1[i] = muteX(pi2[i],pi1[i]);
      ci2[i] = muteX(pi1[i],pi2[i]);
    }
  } else {
    for(i = 0; i < lchrom; i++){
      ci1[i] = muteX(pi1[i],pi2[i]);
      ci2[i] = muteX(pi2[i],pi1[i]);
    }
  }
}

int muteX(pa, pb)
int pa, pb;
{
  return (flip(pmut) ? 1 - pa  : pa);
}


