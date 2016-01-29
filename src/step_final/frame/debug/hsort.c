#include <stdio.h>

#define ARRAYLEN 4

void HeapAdjust(int a[],int s,int n)//¹¹³É¶Ñ
{
  int j,t;
  while(2*s+1<n) //µÚs¸ö½áµãÓÐÓÒ×ÓÊ÷ 
  {
	j=2*s+1 ;
	if((j+1)<n)
	{            
	  if(a[j]<a[j+1])//ÓÒ×ó×ÓÊ÷Ð¡ÓÚÓÒ×ÓÊ÷£¬ÔòÐèÒª±È½ÏÓÒ×ÓÊ÷
		j++; //ÐòºÅÔö¼Ó1£¬Ö¸ÏòÓÒ×ÓÊ÷ 
	}
	if(a[s]<a[j])//±È½ÏsÓëjÎªÐòºÅµÄÊý¾Ý
	{            
	  t=a[s];  //½»»»Êý¾Ý 
	  a[s]=a[j];
	  a[j]=t;            
	  s=j ;//¶Ñ±»ÆÆ»µ£¬ÐèÒªÖØÐ µ÷Õû
	}
	else //±È½Ï×óÓÒº¢×Ó¾ù´óÔò¶ÑÎ´ÆÆ»µ£¬²»ÔÙÐèÒªµ÷Õû
	  break;
  }
}
void HeapSort(int a[],int n)//¶ÑÅÅÐò
{
  int t,i;
  int j;
  for(i=n/2-1;i>=0;i--)    //½«a[0,n-1]½¨³É´ó¸ù¶Ñ
	HeapAdjust(a, i, n);

  for(i=0;i<ARRAYLEN;i++)
	printf("%d ",a[i]);
  printf("\n");

  for(i=n-1;i>0;i--)
  {
	t=a[0];//ÓëµÚi¸ö¼Ç ¼½»»»
	a[0] =a[i];
	a[i] =t;
	HeapAdjust(a,0,i);        //½«a[0]ÖÁa[i]ÖØÐ µ÷ÕûÎª¶Ñ
  }  
}
int main()
{
  int i,a[ARRAYLEN];
  for(i=0;i<ARRAYLEN;i++)
	a[i]=0;
  for(i=0;i<ARRAYLEN;i++)
	scanf("%d ",&a[i]);
  printf("origin:"); 
  for(i=0;i<ARRAYLEN;i++)
	printf("%d ",a[i]);
  printf("\n");
  HeapSort(a,ARRAYLEN);
  printf("Sorted:"); 
  for(i=0;i<ARRAYLEN;i++)
	printf("%d ",a[i]);
  printf("\n");
  getchar();getchar();
  getchar();
  return 0;   
}
