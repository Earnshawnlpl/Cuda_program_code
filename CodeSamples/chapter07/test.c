#include <stdio.h>
#include <stdlib.h>

int main()
{
    double a = 3.1415927;
    double b = 3.1415928;
    if(a == b)
    {
        printf("a and b are equal\n");
    }
    else
    {
        printf("a and b are not equal\n");
    }   
}