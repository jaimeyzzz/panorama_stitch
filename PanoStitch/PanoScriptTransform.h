#pragma once

#include <cmath>
#include <cstdio>
#include <stdarg.h>
#include <cstring>
#include <cstdlib>
#include <ctype.h>
#include <limits.h>

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef bzero
#define bzero(dest, len)   memset((dest), 0, (len))
#endif


//---------------------- Some useful math defines --------------------------

#ifndef PI
#define PI 3.14159265358979323846264338327950288
#endif
#ifndef HALF_PI
#define HALF_PI (PI*0.5)
#endif

#define EPSLN	1.0e-10

// Normalize an angle to +/-180degrees

#define NORM_ANGLE( x )      while( x >180.0 ) x -= 360.0; while( x < -180.0 ) x += 360.0;
#define NORM_ANGLE_RAD( x )  while( (x) >PI ) (x) -= 2 * PI; while( (x) < -PI ) (x) += 2 * PI;

// Convert degree to radian

#define DEG_TO_RAD( x )		( (x) * 2.0 * PI / 360.0 )

// and reverse

#define RAD_TO_DEG( x )		( (x) * 360.0 / ( 2.0 * PI ) )

// Convert double x to unsigned char/short c



#define	DBL_TO_UC( c, x )	if((x)>255.0) c=255U;								\
								else if ((x)<0.0) c = 0;							\
								else c = (unsigned char)floor((x)+0.5);

#define	DBL_TO_US( c, x )	if((x)>65535.0) c=65535U;							\
								else if ((x)<0.0) c = 0;							\
								else c = (unsigned short)floor((x)+0.5);

#define	DBL_TO_FL( c, x )	if((x)>1e+038) c=1e+038;							\
								else if ((x)<0.0) c = 0;							\
								else c = (float)(x);


//#define MAX_FISHEYE_FOV		179.0
//
//#define FAST_TRANSFORM_STEP_NORMAL	40
//#define FAST_TRANSFORM_STEP_MORPH	6
//#define FAST_TRANSFORM_STEP_NONE    0
//
//struct PTPoint
//{
//	double x;
//	double y;
//};
//
//typedef struct PTPoint PTPoint;
//
//#define CopyPTPoint( to, from )       memcpy( &to, &from, sizeof( PTPoint ))
//#define SamePTPoint( p, s )			  ((p).x == (s).x && (p).y == (s).y)
//
//struct PTLine
//{
//	PTPoint v[2];
//};
//
//typedef struct PTLine PTLine;
//
//
//struct PTTriangle
//{
//	PTPoint v[3];
//};
//
//typedef struct PTTriangle PTTriangle;
//
////----------------------- Structures -------------------------------------------
//
//struct remap_Prefs{								// Preferences Structure for remap
//	int    		magic;					//  File validity check, must be 30
//	int				from;					// Image format source image
//	int				to;						// Image format destination image
//	double			hfov;					// horizontal field of view /in degrees
//	double			vfov;					// vertical field of view (usually ignored)
//};
//
//typedef struct remap_Prefs rPrefs;
//
//struct perspective_Prefs{						//  Preferences structure for tool perspective
//	int			magic;					//  File validity check, must be 40
//	int				format;					//  rectilinear or fisheye?
//	double  		hfov;					//  Horizontal field of view (in degree)
//	double			x_alpha;				//  New viewing direction (x coordinate or angle)
//	double 			y_beta;					//  New viewing direction (y coordinate or angle)
//	double			gamma;					//  Angle of rotation
//	int				unit_is_cart;			//  true, if viewing direction is specified in coordinates
//	int				width;					//  new width
//	int				height;					//  new height
//};
//
//typedef struct perspective_Prefs pPrefs;
//
//
//struct optVars{									//  Indicate to optimizer which variables to optimize
//	int hfov;								//  optimize hfov? 0-no 1-yes , etc
//
//	// panotools uses these variables for two purposes: to 
//	// determine which variables are used for reference to another one
//	// and to determine which variables to optimize
//
//	int yaw;
//	int pitch;
//	int roll;
//	int a;
//	int b;
//	int c;
//	int d;
//	int e;
//	int shear_x;
//	int shear_y;
//	int tiltXopt;
//	int tiltYopt;
//	int tiltZopt;
//	int tiltScaleOpt;
//
//	int transXopt;
//	int transYopt;
//	int transZopt;
//	int transYawOpt;
//	int transPitchOpt;
//
//	int testP0opt;
//	int testP1opt;
//	int testP2opt;
//	int testP3opt;
//
//};
//
//typedef struct optVars optVars;
//
//
//enum{										// Enumerates for stBuf.seam
//	_middle,								// seam is placed in the middle of the overlap
//	_dest									// seam is places at the edge of the image to be inserted
//};
//
//enum{										// Enumerates for colcorrect
//	_colCorrectImage = 1,
//	_colCorrectBuffer = 2,
//	_colCorrectBoth = 3,
//};
//
//struct stitchBuffer{						// Used describe how images should be merged
//	char				srcName[256];		// Buffer should be merged to image; 0 if not.
//	char				destName[256];		// Converted image (ie pano) should be saved to buffer; 0 if not
//	int				feather;		// Width of feather
//	int				colcorrect;		// Should the images be color corrected?
//	int				seam;			// Where to put the seam (see above)
//	unsigned char                   psdOpacity;               // Opacity of the layer. Currently used only by PSD output. 0 trans, 255 opaque
//	unsigned char                   psdBlendingMode;          // blending mode (photoshop)
//};
//
//
//typedef struct stitchBuffer stBuf;
//
//
//#if 0
//struct controlPoint{							// Control Points to adjust images
//	int  num[2];							// Indices of Images 
//	int	 x[2];								// x - Coordinates 
//	int  y[2];								// y - Coordinates 
//	int  type;								// What to optimize: 0-r, 1-x, 2-y
//};
//#endif
//struct controlPoint{							// Control Points to adjust images
//	int  num[2];							// Indices of Images 
//	double x[2];								// x - Coordinates 
//	double y[2];								// y - Coordinates 
//	int  type;								// What to optimize: 0-r, 1-x, 2-y
//};
//
//typedef struct controlPoint controlPoint;
//
//struct CoordInfo{								// Real World 3D coordinates
//	int  num;								// auxilliary index
//	double x[3];
//	int  set[3];
//};
//
//typedef struct CoordInfo CoordInfo;
//
//// Some useful macros for vectors
//
//#define SCALAR_PRODUCT( v1, v2 )	( (v1)->x[0]*(v2)->x[0] + (v1)->x[1]*(v2)->x[1] + (v1)->x[2]*(v2)->x[2] ) 
//#define ABS_SQUARED( v )			SCALAR_PRODUCT( v, v )
//#define ABS_VECTOR( v )				sqrt( ABS_SQUARED( v ) )
//#define CROSS_PRODUCT( v1, v2, r )  { (r)->x[0] = (v1)->x[1] * (v2)->x[2] - (v1)->x[2]*(v2)->x[1];  \
//	(r)->x[1] = (v1)->x[2] * (v2)->x[0] - (v1)->x[0] * (v2)->x[2];	\
//	(r)->x[2] = (v1)->x[0] * (v2)->x[1] - (v1)->x[1] * (v2)->x[0]; }
//#define DIFF_VECTOR( v1, v2, r )  	{ 	(r)->x[0] = (v1)->x[0] - (v2)->x[0];  \
//	(r)->x[1] = (v1)->x[1] - (v2)->x[1];  \
//	(r)->x[2] = (v1)->x[2] - (v2)->x[2]; }
//#define DIST_VECTOR( v1, v2 )		sqrt( ((v1)->x[0] - (v2)->x[0]) * ((v1)->x[0] - (v2)->x[0]) + \
//	((v1)->x[1] - (v2)->x[1]) * ((v1)->x[1] - (v2)->x[1]) + \
//	((v1)->x[2] - (v2)->x[2]) * ((v1)->x[2] - (v2)->x[2]))
//
//struct transformCoord{							// 
//	int nump;								// Number of p-coordinates
//	CoordInfo  *p;							// Coordinates "as is"
//	int numr;								// Number of r-coordinates
//	CoordInfo  *r;							// Requested values for coordinates
//};
//
//typedef struct transformCoord transformCoord;
//
//struct  tMatrix{
//	double alpha;
//	double beta;
//	double gamma;
//	double x_shift[3];
//	double scale;
//};
//
//typedef struct tMatrix tMatrix;
//

void 	SetMatrix(double a, double b, double c, double m[3][3], int cl);

 int resize(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int shear(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int shearInv(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int horiz(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int vert(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int radial(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int radial_brown(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);

 //int tiltForward(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 //int tiltInverse(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);

 int persp_sphere(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int persp_rect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);


 int rect_pano(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int pano_rect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int pano_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_pano(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int sphere_cp_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int sphere_tp_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_sphere_cp(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int rect_sphere_tp(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int sphere_tp_rect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int sphere_cp_pano(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int rect_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_rect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int plane_transfer_to_camera(double x_dest, double y_dest, double * x_src, double * y_src, void * params);
 int plane_transfer_from_camera(double x_dest, double y_dest, double * x_src, double * y_src, void * params);
 int erect_sphere_tp(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int mirror_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int mercator_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_mercator(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int lambert_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_lambert(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_lambertazimuthal(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int lambertazimuthal_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_hammer(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int hammer_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int transmercator_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_transmercator(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int sinusoidal_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_sinusoidal(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int stereographic_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_stereographic(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int albersequalareaconic_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_albersequalareaconic(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int albersequalareaconic_distance(double *x_src, void* params);
 int millercylindrical_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_millercylindrical(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int panini_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_panini(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int equipanini_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_equipanini(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);

 int panini_general_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_panini_general(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);

 int arch_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_arch(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);

 int biplane_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_biplane(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int biplane_distance(double width, double b, void* params);
 int triplane_erect(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int erect_triplane(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int triplane_distance(double width, double b, void* params);

 int mirror_sphere_cp(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int mirror_pano(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int sphere_cp_mirror(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);

 int sphere_tp_pano(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int pano_sphere_tp(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int sphere_tp_mirror(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int mirror_sphere_tp(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int sphere_tp_equisolid(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int equisolid_sphere_tp(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int sphere_tp_orthographic(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int orthographic_sphere_tp(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);

 int sphere_tp_thoby(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);
 int thoby_sphere_tp(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);


 int rotate_erect(double x_dest, double y_dest, double* x_src, double* y_src, void* params);
 int inv_radial(double x_dest, double y_dest, double* x_src, double* y_src, void* params);
 int inv_radial_brown(double x_dest, double y_dest, double* x_src, double* y_src, void* params);

 int vertical(double x_dest, double y_dest, double* x_src, double* y_src, void* params);
 int inv_vertical(double x_dest, double y_dest, double* x_src, double* y_src, void* params);
 int deregister(double x_dest, double y_dest, double* x_src, double* y_src, void* params);
 int tmorph(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);

 int shift_scale_rotate(double x_dest, double  y_dest, double* x_src, double* y_src, void* params);


 void SetCorrectionRadius(double* rad);