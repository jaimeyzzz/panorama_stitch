#include "MVCTriangulate.h"
#include "../PanoStitch.h"

using namespace cv;
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria> Mesher;

typedef CDT::Vertex_handle Vertex_handle;
typedef K::Point_2   CGPoint;

void MVCTriangulate(std::vector<cv::Point>& pts, std::vector<int>& tris, const cv::Mat & mask) {
	vector<CGPoint> vertices;
	for (int i = 0; i < pts.size(); i++)
		vertices.push_back(CGPoint(pts[i].x, pts[i].y));
	CDT cdt;

	for (int i = 0; i< vertices.size(); i++) {
		Vertex_handle v_pt1 = cdt.insert(vertices[i]);
		Vertex_handle v_pt2 = cdt.insert(vertices[(i + 1) % vertices.size()]);

		cdt.insert_constraint(v_pt1, v_pt2);
	}
	assert(cdt.is_valid());

	Mesher mesher(cdt);
	mesher.refine_mesh();

	CDT::Finite_vertices_iterator vit;
	pts.clear();
	for (vit = cdt.finite_vertices_begin(); vit != cdt.finite_vertices_end(); vit++) {
		pts.push_back(Point(vit->point().hx(), vit->point().hy()));
	}

#ifdef DEBUG_MODE
	Mat debugMeshResult = Mat::zeros(mask.rows, mask.cols, CV_8UC4);
#endif
	CDT::Finite_faces_iterator fit;
	for (fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); fit++) {
		Point ** vers = new Point*[1];
		vers[0] = new Point[3];
		bool out = false;
		double cx = 0.0, cy = 0.0;
		for (int k = 0; k < 3; k++) {
			vers[0][k] = Point(fit->vertex(k)->point().hx(), fit->vertex(k)->point().hy());
			cx += vers[0][k].x, cy += vers[0][k].y;
			if (mask.at<uchar>(vers[0][k].y, vers[0][k].x) == 0) out = true;
		}
		cx /= 3, cy /= 3;
		if (mask.at<uchar>(cvRound(cy), cvRound(cx)) == 0) out = true;
		if (out) continue;
		int* npts = new int[1];
		npts[0] = 3;

#ifdef DEBUG_MODE		
		fillPoly(debugMeshResult, (const Point**)vers, npts, 1, Scalar(255, 255, 255, 255));
		polylines(debugMeshResult, (const Point**)vers, npts, 3, true, Scalar(255, 64, 0, 255));
#endif

		for (int k = 0; k < 3; k++) {
			Point pt(fit->vertex(k)->point().hx(), fit->vertex(k)->point().hy());
			int index = find(pts.begin(), pts.end(), pt) - pts.begin();
			tris.push_back(index);
		}
	}
#ifdef DEBUG_MODE
	imwrite("DEBUG_MODE\\mvc_triangulate_mesh.png", debugMeshResult);
	PanoStitch::showImage("MVC Triangulate Mesh", debugMeshResult, 0.5);
#endif
}