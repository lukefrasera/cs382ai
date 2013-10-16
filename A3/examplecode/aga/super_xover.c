#include <stdio.h>

#include "const.h"
#include "type.h"
#include "extern.h"
#include "exfunc.h"

void cross(p1, p2, c1, c2)
     IPTR p1, p2, c1, c2;
{
/* p1,p2,c1,c2,m1,m2,mc1,mc2 */
  int *pi1,*pi2,*ci1,*ci2;
  int xp[MAX_SIZE], i, j, k;

  pi1 = p1->chrom;
  pi2 = p2->chrom;
  ci1 = c1->chrom;
  ci2 = c2->chrom;
  
  if(flip(pcross)){
    ncross++;
    if(xType == lchrom){
      for(i = 0; i < lchrom; i++){
	ci1[i] = (flip(0.5) ? pi1[i] : pi2[i]);
	ci2[i] = (flip(0.5) ? pi1[i] : pi2[i]);
      }
    } else {
      for(i = 0; i < xType; i++){
	xp[i] = rnd(0,lchrom);
      }
      do_sort(xp);
      j = 0;
      k=0;
      for(i=0;i<lchrom;i++){
	if(i == xp[j]) {
	  k++; j++;
	}
	ci1[i] = muteX(pi1[i], pi2[i], k%2);
	ci2[i] = muteX(pi2[i], pi1[i], k%2);
      }
    }
  } else {
    for(i = 0; i < lchrom; i++){
      ci1[i] = muteX(pi1[i], pi2[i], 0);
      ci2[i] = muteX(pi2[i], pi1[i], 0);
    }
  }
/**  chrom_write(p1, stdout); chrom_write(p2, stdout);
  printf("\n");
  chrom_write(c1, stdout); chrom_write(c2, stdout);
  printf("\n\n");
  **/
}

int muteX(pa, pb, flag)
int pa, pb, flag;
{
  return (flip(pmut) ? 1 - (flag ? pb : pa ) : (flag ? pb : pa ));

}


void do_sort(xp)
int *xp;
{
  int i,sorted,j;
  
  i=0;
  sorted = 0;
  while(!sorted){
    sorted = 1;
    for(i=0;i<xType-1;i++){
      if(xp[i] > xp[i+1]){
        j = xp[i];
        xp[i] = xp[i+1];
        xp[i+1] = j;
        sorted = 0;
      }
    }
  }
} 

