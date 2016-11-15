// Vector3f.cpp: implementation of the CVector3f class.
//
//////////////////////////////////////////////////////////////////////
#include "Vector3f.h"


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

//得到两向量的叉乘 
/* 有关叉乘的说明(文字出处:http://www.gameres.com/Articles/Program/Visual/Other/shiliang.htm) 
叉乘:Vector1(x1,y1,z1),Vector2(x2,y2,z2): 
其结果是个矢量. 
方向是Vector1,Vector2构成的平面法线.再使用右手定则 
长度是Length(Vector1)*Length(Vector2)*sin(theta) 
theta是Vector1 & Vector2的夹角. 
所以,平行的矢量叉乘结果为0矢量(长为0,方向任意) 
计算结果矢量:(ox,oy,oz) 
ox = (y1 * z2) - (y2 * z1) 
oy = (z1 * x2) - (z2 * x1) 
oz = (x1 * y2) - (x2 * y1) 
用途:计算法向量 
*/ 
CVector3f Cross(CVector3f vVector1, CVector3f vVector2) 
{ 
	//定义一个容纳叉乘结果的向量 
	CVector3f vNormal; 

	//得到此向量在X轴上的投影值 
	vNormal.x = ((vVector1.y * vVector2.z) - (vVector1.z * vVector2.y)); 

	//得到此向量在Y轴上的投影值 
	vNormal.y = ((vVector1.z * vVector2.x) - (vVector1.x * vVector2.z)); 

	//得到此向量在Z轴上的投影值 
	vNormal.z = ((vVector1.x * vVector2.y) - (vVector1.y * vVector2.x)); 

	//返回此向量 
	return vNormal; 
} 

//得到一个向量的绝对长度 
double Magnitude(CVector3f vNormal) 
{ 
	return sqrt( (vNormal.x * vNormal.x) + (vNormal.y * vNormal.y) + (vNormal.z * vNormal.z) ); 
} 

//将一个向量单位化 
CVector3f Normalize(CVector3f vNormal) 
{ 
	//得到此向量的绝对长度 
	double magnitude = Magnitude(vNormal); 

	//让每个分量分别除以此长度 
	vNormal.x /= magnitude;  
	vNormal.y /= magnitude;  
	vNormal.z /= magnitude;  

	//返回此向量 
	return vNormal;  
} 

//得到一个三点决定的平面的垂直向量(经过单位化) 
CVector3f Normal(CVector3f vPolygon[]) 
{ 
	//得到两条边的向量 
	CVector3f vVector1 = vPolygon[2] - vPolygon[0]; 
	CVector3f vVector2 = vPolygon[1] - vPolygon[0]; 

	//得到这两向量的叉乘 
	CVector3f vNormal = Cross(vVector1, vVector2); 

	//单位化 
	vNormal.x = Normalize(vNormal).x;   
	vNormal.y = Normalize(vNormal).y; 
	vNormal.z = Normalize(vNormal).z; 
  
	//返回此变量 
	return vNormal;  
} 

//得到两点间的距离 
double Distance(CVector3f vPoint1, CVector3f vPoint2) 
{ 
	double distance = sqrt( (vPoint2.x - vPoint1.x) * (vPoint2.x - vPoint1.x) + 
						(vPoint2.y - vPoint1.y) * (vPoint2.y - vPoint1.y) + 
						(vPoint2.z - vPoint1.z) * (vPoint2.z - vPoint1.z) ); 

	return (double)distance; 
} 

//得到两向量的点积 
/*有关点积的说明(文字出处:http://www.gameres.com/Articles/Program/Visual/Other/shiliang.htm) 
两个矢量的点积是个标量. 
中学物理的力做功就是矢量点积的例子:W=|F|.|S|.cos(theta) 

二矢量点积: 
Vector1:(x1,y1,z1) Vector2(x2,y2,z2) 
DotProduct=x1*x2+y1*y2+z1*z2 

很重要的应用: 
1.求二矢量余弦: 
由我们最熟悉的力做功: 
cos(theta)=F.S/(|F|.|S|) 
可以判断二矢量的方向情况: cos=1同向,cos=-1相反,cos=0直角 
曲面消隐(Cull face)时判断物体表面是否可见:(法线和视线矢量的方向问题)cos>0不可见,cos<0可见 
OpenGL就是这么做的。 

2.Lambert定理求光照强度也用点积: 
Light=K.I.cos(theta) 
K,I为常数,theta是平面法线与入射光线夹角 
*/ 
#define MAX(a, b) ((a < b) ? (b) : (a))
#define MIN(a, b) ((a < b) ? (a) : (b))
double Dot(CVector3f vVector1, CVector3f vVector2) 
{ 
	return ( (vVector1.x * vVector2.x) + (vVector1.y * vVector2.y) + (vVector1.z * vVector2.z) ); 
} 

//得到两向量的夹角 
double AngleBetweenVectors(CVector3f Vector1, CVector3f Vector2) 
{ 
	//得到两向量的点积 
	double dotProduct = Dot(Vector1, Vector2); 

	//得到两向量长度的乘积 
	double vectorsMagnitude = Magnitude(Vector1) * Magnitude(Vector2) ; 

	//得到两向量夹角 
	double angle = acos( MAX(MIN(dotProduct / vectorsMagnitude, 1.0), -1.0) ); 

	//返回角度值 
	return( angle ); 
}

