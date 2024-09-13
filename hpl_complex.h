
#include <math.h>

struct complex
{
    double r;
    double i;
};


#define zadd( z, x, y ) { (z).r = (x).r + (y).r; \
                              (z).i = (x).i + (y).i; }

// #define zadd_accum( z, x, y ) { (z).r += ((x).r + (y).r); \
//                               (z).i += ((x).i + (y).i); }

#define zsub( z, x, y ) { (z).r = (x).r - (y).r; \
                              (z).i = (x).i - (y).i; }


#define zmul( z, x, y ) { (z).r = (x).r * (y).r - (x).i * (y).i; \
                              (z).i = (x).r * (y).i + (x).i * (y).r; }



#define zdiv( z, x, y ) { double t = (y).r * (y).r + (y).i * (y).i;       \
                              (z).r = (  (x).r * (y).r + (x).i * (y).i ) / t; \
                              (z).i = (  (x).i * (y).r - (x).r * (y).i ) / t; }


#define zneg( z, x )    { (z).r = -(x).r; (z).i = -(x).i; }
#define zrec( z, x )    { double t = (x).r * (x).r + (x).i * (x).i; \
                                 (z).r = (x).r / t; (z).i = -(x).i / t; }


#define zabs( x )       ( fabs( (x).r ) + fabs( (x).i ) )
#define zmod( x )       ( sqrt( (x).r * (x).r + (x).i * (x).i ) )


#define zequ( x, y )    ( (x).r == (y).r && (x).i == (y).i ? 1 : 0 )
#define zneq( x, y )    ( (x).r != (y).r || (x).i != (y).i ? 1 : 0 )

