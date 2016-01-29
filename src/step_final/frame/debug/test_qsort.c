#include <stdio.h>
#include <stdlib.h>
int comp (const void * elem1, const void * elem2) 
{
  int f = *((int*)elem1);
  int s = *((int*)elem2);
  if (f > s) return  1;
  if (f < s) return -1;
  return 0;
}
int main(int argc, char* argv[]) 
{
  int i;
  int x[] = {4,5,2,3,1,0,9,8,6,7};

  qsort (x, sizeof(x)/sizeof(*x), sizeof(*x), comp);

  for (i = 0 ; i < 10 ; i++)
    printf ("%d ", x[i]);

  return 0;
}
