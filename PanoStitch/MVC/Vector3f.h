// Vector3f.h: interface for the CVector3f class.
//
//////////////////////////////////////////////////////////////////////

#ifndef VECTOR_3F
#define VECTOR_3F


#include <math.h>


class CVector3f  
{
public: 
	//无参构造函数 
	CVector3f() 
	{ 
		x = y = z = 0; 
	}; 
	//有参构造函数 
	CVector3f(const double inx,const double iny,const double inz) 
	{ 
		//分别将参数赋值给三个成员变量 
		x = inx; 
		y = iny; 
		z = inz;  
	}; 
	//析构函数 
	~CVector3f(){}; 
  
	//重载=操作符,实现两向量变量的赋值 
	CVector3f& operator=(const CVector3f& inVet) 
	{ 
		x = inVet.x; 
		y = inVet.y; 
		z = inVet.z; 
		return *this; 
	}; 

	//重载+操作符,实现两向量变量的相加 
	CVector3f operator+(CVector3f vVector) 
	{ 
		//返回相加的结果 
		return CVector3f(vVector.x + x, vVector.y + y, vVector.z + z); 
	} 
	CVector3f operator+=(CVector3f vVector)
	{
		return CVector3f(vVector.x + x, vVector.y + y, vVector.z + z); 
	}

	//重载-操作符,实现两向量变量的相减 
	CVector3f operator-(CVector3f vVector) 
	{ 
		//返回相减的结果 
		return CVector3f(x - vVector.x, y - vVector.y, z - vVector.z); 
	} 

	//重载*操作符,实现一个向量变量和一个浮点数的乘法 
	CVector3f operator*(double num) 
	{ 
		//返回缩放了的向量 
		return CVector3f(x * num, y * num, z * num); 
	} 

	//重载/操作符,实现一个向量变量和一个浮点数的除法 
	CVector3f operator/(double num) 
	{ 
		//返回缩放了的向量 
		return CVector3f(x / num, y / num, z / num); 
	} 

	//重载==操作符
	bool operator==(CVector3f vVector)
	{
		if (x==vVector.x && y==vVector.y && z==vVector.z)
			return true;
		else 
			return false;
	}
  
	//向量绕x轴旋转,参数sintheta为旋转角度的正弦值,参数costheta为旋转角度的余弦值 
	void RotateX(double theta) 
	{
		double sintheta = sin(theta);
		double costheta = cos(theta);
		double sin_beta, cos_bata; 
		sin_beta = z * costheta + y * sintheta; 
		cos_bata = y * costheta - z * sintheta; 
		z = sin_beta;
		y = cos_bata; 
	}; 

	//向量绕y轴旋转,参数sintheta为旋转角度的正弦值,参数costheta为旋转角度的余弦值 
	void RotateY(double theta) 
	{ 
		double sintheta = sin(theta);
		double costheta = cos(theta);
		double sin_beta, cos_bata; 
		sin_beta = z * costheta + x * sintheta; 
		cos_bata = x * costheta - z * sintheta;
		z = sin_beta;
		x = cos_bata; 
	}; 

	//向量绕z轴旋转,参数sintheta为旋转角度的正弦值,参数costheta为旋转角度的余弦值 
	void RotateZ(double theta) 
	{ 
		double sintheta = sin(theta);
		double costheta = cos(theta);
		double sin_beta, cos_bata; 
		sin_beta = y * costheta + x * sintheta; 
		cos_bata = x * costheta - y * sintheta; 
		y = sin_beta;
		x = cos_bata; 
	}; 
  
	//缩放一个向量,参数scale为缩放的比例 
	void Zoom(double scale) 
	{ 
		x *= scale; 
		y *= scale; 
		z *= scale; 
	}; 
  
	//平移一个向量 
	void Move(CVector3f inVect) 
	{ 
		x += inVect.x; 
		y += inVect.y; 
		z += inVect.z; 
	}; 

public:  
	double x;//成员变量x,向量在x轴上的分量 
	double y;//成员变量y,向量在y轴上的分量 
	double z;//成员变量z,向量在z轴上的分量 

};

//得到两向量的叉乘 
CVector3f Cross(CVector3f vVector1, CVector3f vVector2); 

//得到一个向量的绝对长度 
double Magnitude(CVector3f vNormal); 

//将一个向量单位化 
CVector3f Normalize(CVector3f vNormal); 

//得到一个三点决定的平面的垂直向量(经过单位化) 
CVector3f Normal(CVector3f vPolygon[]); 

//得到空间中两点的距离 
double Distance(CVector3f vPoint1, CVector3f vPoint2); 

//得到两向量的点积 
double Dot(CVector3f vVector1, CVector3f vVector2); 

//得到两向量的夹角 
double AngleBetweenVectors(CVector3f Vector1, CVector3f Vector2); 

#endif // !defined(AFX_VECTOR3F_H__464A32D5_CD8B_410F_B4D4_DECAA91AAC91__INCLUDED_)
