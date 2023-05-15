#include <iostream>
#include <stdio.h>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/Color.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <bits/stdc++.h> //or set the correct path using #include "bits/stdc++.h>"

using namespace cv;
using namespace std;
struct info
{
    CGAL::Color c = CGAL::Color(0, 0, 0, 0);
    float val = -10;
    int id = -1;
    int parent = -1;
    int pro = 0;
} inf1;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Triangulation_face_base_with_info_2<info, K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Triangulation_2<K, Tds> Triangulation;
typedef Triangulation::Face_handle Face_handle;
typedef CGAL::Triangulation_2<K, Tds>::Point point;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Delaunay;
Delaunay t;
char inputname[100], argname[100];
int minx = 9999, miny = 9999, maxx = 0, maxy = 0, filledcolsi = 0, trii = 0, init = 0, cset = 0;
cv::Mat image, img_skel, filled, img1, filli;
std::vector<point> foreground;
CGAL::Color brushcol, filledcols[10000];
point tri[1000][3];

class Data
{
public:
    Delaunay::Face_handle fh;
    float priority;
};
bool operator<(const Data &lhs, const Data &rhs);
bool operator<(const Data &lhs, const Data &rhs)
{
    return lhs.priority < rhs.priority;
}
struct LessThanByPriority
{
    bool operator()(const Data &lhs, const Data &rhs) const
    {
        return lhs.priority < rhs.priority;
    }
};
std::priority_queue<Data, std::vector<Data>, LessThanByPriority> pq, pq1;
float distance(point a, point b)
{
    return sqrt(((b.x() - a.x()) * (b.x() - a.x())) + ((b.y() - a.y()) * (b.y() - a.y())));
}
void fillcolor(Delaunay::Face_handle fh, float priority, void *param, int parent)
{
    if (fh->info().val <= priority && fh->info().id != -1 && (fh->info().parent != parent || fh->info().c != brushcol) && fh->info().val != 99999)
    {
        fh->info().pro = 1;
        Mat &img = *((Mat *)(param));
        fh->info().parent = parent;
        cv::Point rook_points[1][20];
        fh->info().c = brushcol;
        for (int i = 0; i < 3; i++)
            rook_points[0][i] = cv::Point(fh->vertex(i)->point().x(), fh->vertex(i)->point().y());
        const Point *ppt[1] = {rook_points[0]};
        int npt[] = {3};
        fillPoly(img, ppt, npt, 1, cv::Scalar(brushcol.red(), brushcol.green(), brushcol.blue()), 8);
        fh->info().val = priority;
        Data d1;
        for (int i = 0; i < 3; i++)
        {
            d1.fh = fh->neighbor(i);
            d1.priority = distance(fh->vertex((i + 1) % 3)->point(), fh->vertex((i + 2) % 3)->point());
            if (d1.priority > priority)
                d1.priority = priority;
            if (init == 0 || d1.priority > 5)
                if (d1.priority > 2)
                    pq.push(d1);
        }
    }
}
void thinningIteration(cv::Mat &img, int iter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);
    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);
    int nRows = img.rows;
    int nCols = img.cols;
    if (img.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;
    uchar *pDst;
    pAbove = NULL;
    pCurr = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);
    for (y = 1; y < img.rows - 1; ++y)
    {
        pAbove = pCurr;
        pCurr = pBelow;
        pBelow = img.ptr<uchar>(y + 1);
        pDst = marker.ptr<uchar>(y);
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);
        for (x = 1; x < img.cols - 1; ++x)
        {
            nw = no;
            no = ne;
            ne = &(pAbove[x + 1]);
            we = me;
            me = ea;
            ea = &(pCurr[x + 1]);
            sw = so;
            so = se;
            se = &(pBelow[x + 1]);
            int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                    (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                    (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                    (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);
            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
        }
    }
    img &= ~marker;
}
void thinning(const cv::Mat &src, cv::Mat &dst)
{
    dst = src.clone();
    dst /= 255;
    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;
    do
    {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } while (cv::countNonZero(diff) > 0);
    dst *= 255;
}
void simplify()
{
    float sp;
    filled = image.clone();
    Mat im = image.clone();
    Mat oim;
    oim = image.clone();
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
        {
            if ((image.at<cv::Vec3b>(i, j)[0] < 10 && image.at<cv::Vec3b>(i, j)[1] < 10 && image.at<cv::Vec3b>(i, j)[2] < 10) || (image.at<cv::Vec3b>(i, j)[0] > 200 && image.at<cv::Vec3b>(i, j)[1] > 200 && image.at<cv::Vec3b>(i, j)[2] > 200))
            {
                im.at<Vec3b>(i, j) = {0, 0, 0};
                filled.at<Vec3b>(i, j) = {0, 0, 0};
            }
            else
                filled.at<Vec3b>(i, j) = {255, 255, 255};
        }
    copyMakeBorder(im, im, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(255, 255, 255));
    strcpy(inputname, argname);
    filli = image.clone();
    int sx, lx, sy, ly;
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
        {
            if (im.at<cv::Vec3b>(i, j) == Vec3b(0, 0, 0))
            {
                float min = 9999, min1 = 9999;
                Vec3b c, c1;
                c = Vec3b(0, 0, 0);
                sx = i - 30;
                if (sx < 0)
                    sx = 0;
                lx = i + 30;
                if (lx > image.rows - 1)
                    lx = image.rows;
                sy = j - 30;
                if (sy < 0)
                    sy = 0;
                ly = j + 30;
                if (ly > image.cols - 1)
                    ly = image.cols;
                for (int i1 = sx; i1 < lx; i1++)
                    for (int j1 = sy; j1 < ly; j1++)
                    {
                        if (im.at<cv::Vec3b>(i1, j1) != Vec3b(0, 0, 0))
                            if (distance(point(i, j), point(i1, j1)) < min)
                            {
                                min = distance(point(i, j), point(i1, j1));
                                c = im.at<cv::Vec3b>(i1, j1);
                            }
                    }
                for (int i1 = sx; i1 < lx; i1++)
                    for (int j1 = sy; j1 < ly; j1++)
                        if (im.at<cv::Vec3b>(i1, j1) != Vec3b(0, 0, 0))
                            if (distance(point(i, j), point(i1, j1)) < min1 && im.at<cv::Vec3b>(i1, j1) != c)
                                min1 = distance(point(i, j), point(i1, j1));
                if (abs(min - min1) < 1.5)
                    filli.at<Vec3b>(i, j) = {0, 0, 0};
                else
                    filli.at<Vec3b>(i, j) = {255, 255, 255};
            }
            else
                filli.at<Vec3b>(i, j) = {255, 255, 255};
        }
    cv::imshow("Result", filli);
}
void onmouse(int event, int x, int y, int flags, void *param)
{
    Mat &img = *((Mat *)(param));

    if (event == 1) // left button
    {
        if (cset == 0)
            brushcol = CGAL::Color(rand() % 200 + 10, rand() % 200 + 10, rand() % 200 + 10);
        cset = 0;
        Face_handle fh = t.locate(point(x, y));
        fh->info().val = 99999;
        fh->info().parent = fh->info().id;
        fh->info().c = brushcol;
        filledcols[filledcolsi++] = brushcol;
        cv::Point rook_points[1][20];
        for (int i = 0; i < 3; i++)
            rook_points[0][i] = cv::Point(fh->vertex(i)->point().x(), fh->vertex(i)->point().y());
        const Point *ppt[1] = {rook_points[0]};
        int npt[] = {3};
        fillPoly(img, ppt, npt, 1, cv::Scalar(brushcol.red(), brushcol.green(), brushcol.blue()), 8);
        for (int i = 0; i < 3; i++)
            line(img, cv::Point(fh->vertex((i + 1) % 3)->point().x(), fh->vertex((i + 1) % 3)->point().y()),
                 cv::Point(fh->vertex((i + 2) % 3)->point().x(), fh->vertex((i + 2) % 3)->point().y()), cv::Scalar(brushcol.red(), brushcol.green(), brushcol.blue()), 3, cv::LINE_8);
        Delaunay::Face_handle fh1 = fh->neighbor(0);
        tri[trii][0] = fh->vertex(0)->point();
        tri[trii][1] = fh->vertex(1)->point();
        tri[trii][2] = fh->vertex(2)->point();
        trii++;
        Data d1;
        for (int i = 0; i < 3; i++)
        {
            d1.fh = fh->neighbor(i);
            d1.priority = distance(fh->vertex((i + 1) % 3)->point(), fh->vertex((i + 2) % 3)->point());
            if (d1.priority > 5)
                pq.push(d1);
        }
        int parent = fh->info().id;
        while (pq.empty() != 1)
        {
            Data d2 = pq.top();
            pq.pop();
            fillcolor(d2.fh, d2.priority, &img, parent);
        }
        for (int i = 0; i != foreground.size(); ++i)
            img.at<cv::Vec3b>(foreground.at(i).x(), foreground.at(i).y()) = {0, 0, 0};
        cv::imshow("Display Image", img);
        simplify();
    }
    if (event == 2) // right button
    {
        cv::Mat temim = img.clone();
        for (int i = 0; i < trii; i++)
            for (int j = 0; j < 3; j++) // blue triangles
                line(temim, cv::Point(tri[i][(j + 1) % 3].x(), tri[i][(j + 1) % 3].y()),
                     cv::Point(tri[i][(j + 2) % 3].x(), tri[i][(j + 2) % 3].y()), cv::Scalar(0, 0, 255), 2, cv::LINE_8);
        cv::Vec3b col = img.at<cv::Vec3b>(y, x); // color setting using reverse coords (rand setting)
        cset = 1;
        brushcol = CGAL::Color(col[0], col[1], col[2]);
        cv::imshow("Display Image", temim);
    }
    if (event == 3) // middle button
    {
        simplify();
        Mat im_gray;
        Mat blank(img1.rows, img1.cols, CV_8UC3, Scalar(255, 255, 255));
        img_skel = blank.clone();
        cvtColor(filli, im_gray, cv::COLOR_RGB2GRAY);
        Mat img_bw = im_gray > 128;
        cv::subtract(cv::Scalar::all(255), img_bw, img_bw);
        thinning(img_bw, img_skel);
        Mat sub_mat = Mat::ones(img_skel.size(), img_skel.type()) * 255;
        subtract(sub_mat, img_skel, img_skel);
        bitwise_and(img_bw, img_skel, img_bw);
        cv::imwrite("result.png", img_skel);
        exit(0);
    }
}
void initialize()
{
    init = 1;
    Delaunay::Finite_faces_iterator it;
    for (it = t.finite_faces_begin(); it != t.finite_faces_end(); it++)
    {
        Delaunay::Face_handle f = it;
        Data d1;
        d1.fh = f;
        d1.priority = distance(f->vertex(0)->point(), CGAL::circumcenter(f->vertex(0)->point(), f->vertex(1)->point(), f->vertex(2)->point()));
        pq1.push(d1);
    }
    while (1)
    {
        Data d1 = pq1.top();
        pq1.pop();
        if (d1.priority < 5 || pq1.size() == 0)
            break;
        if (d1.fh->info().pro == 0)
            if (d1.priority > 5)
            {
                Delaunay::Face_handle fh = d1.fh;
                brushcol = CGAL::Color(rand() % 200 + 10, rand() % 200 + 10, rand() % 200 + 10);
                filledcols[filledcolsi++] = brushcol;
                for (int i = 0; i < 3; i++)
                {
                    d1.fh = fh->neighbor(i);
                    d1.priority = distance(fh->vertex((i + 1) % 3)->point(), fh->vertex((i + 2) % 3)->point());
                    if (d1.priority > 5)
                        pq.push(d1);
                }
                int parent = fh->info().id;
                int cou = 0;
                while (pq.empty() != 1)
                {
                    cou++;
                    Data d2 = pq.top();
                    pq.pop();
                    fillcolor(d2.fh, d2.priority, &image, parent);
                }
                if (cou > 3)
                {
                    for (int i = 0; i < 3; i++)
                        tri[trii][i] = fh->vertex(i)->point();
                    trii++;
                }
                d1.fh->info().pro = 1;
            }
    }
    init = 0;
}

//
// int main(int argc, char** argv ) {
//
//// image load and windows creation
//    image = cv::imread( argv[1], 1 );
//    strcpy(argname,argv[1]);
//    if ( !image.data )
//    {
//        printf("No image data \n");
//        return -1;
//    }
//    int s1=image.rows,s2=image.cols,id=0;
//    double p1=600.0/s1;
//    double p2=600.0/s2;
//    if(p1<p2)
//        p1=p2;
//    cv::resize(image,image,cv::Size(p1*image.cols,p1*image.rows));
//    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE|cv::WINDOW_GUI_NORMAL );
////cv::namedWindow("Display Image",cv::WINDOW_GUI_NORMAL  );
//    setMouseCallback("Display Image", onmouse, &image);
//    cv::imshow("Display Image", image);
//    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE|cv::WINDOW_GUI_NORMAL);
//    cv::imshow("Result", image);
//    brushcol=CGAL::Color(0,0,255);
//    for(int i=0; i<image.rows; i++)
//        for(int j=0; j<image.cols; j++)
//            // remove all pixels not red(colored)
//            if(image.at<cv::Vec3b>(i,j)[0]<240)
//            {
//                t.insert(point(j,i));
//                foreground.push_back(point(i,j));
//                image.at<cv::Vec3b>(i,j)={0,0,0};
//            }
//            // paint to black
//            else
//                image.at<cv::Vec3b>(i,j)={255,255,255};
//    t.insert(point(0,0));
//    t.insert(point(0,image.rows-1));
//    t.insert(point(image.cols-1,0));
//    t.insert(point(image.cols-1,image.rows-1));
//
//// data restructure and drawing
//    Delaunay::Finite_faces_iterator it;
//    for (it = t.finite_faces_begin(); it != t.finite_faces_end(); it++)
//    {
//        Delaunay::Face_handle f=it;
//        inf1.id=id;
//        inf1.parent=id;
//        inf1.pro=0;
//        id++;
//        f->info() = inf1;
//    }
//    initialize();
//    cv::imshow("Display Image", image);
//    cv::imshow("Result", image);
//    simplify();
//    cv::waitKey();
//    return 0;
//}

cv::Vec3b WHITE = {255, 255, 255};
cv::Vec3b BLACK = {0, 0, 0};
cv::Vec3b BLUE = {255, 0, 0};
cv::Vec3b GREEN = {0, 255, 0};
cv::Vec3b RED = {0, 0, 255};
// #define Debug

int cellDetect(cv::Mat &img, int x, int y)
{
    int neighbour = 0;
    if (img.at<cv::Vec3b>(x, y) == BLACK)
    {
        for (int i = x - 1; i < x + 2; i++)
        {
            for (int j = y - 1; j < y + 2; j++)
            {
                if (img.at<cv::Vec3b>(i, j) == BLACK)
                    neighbour++;
            }
        }
        if (neighbour < 2)
            return 0;
        else if (neighbour > 3)
            return 2; // junctions
        else
            return 1; // end points
    }
    return 0;
}

/// <summary>
/// <para>Count neighbours starting from the centre black cell</para>
/// </summary>
/// <param name="img">Matrix</param>
/// <param name="x"></param>
/// <param name="y"></param>
int countNeighbours(const cv::Mat &img, int x, int y)
{

    if (img.at<cv::Vec3b>(x, y) != BLACK && img.at<cv::Vec3b>(x, y) != RED && img.at<cv::Vec3b>(x, y) != GREEN)
        return 0;

    int neighbourcount = 0;
    for (int ni = x - 1; ni < x + 2; ni++)
    {
        for (int nj = y - 1; nj < y + 2; nj++)
        {
            if (ni == x && nj == y)
                continue;
            else if (img.at<cv::Vec3b>(ni, nj) == BLACK)
                neighbourcount++;
        }
    }
    return neighbourcount;
}

/// <summary>
/// <para>detech (x,y) surrounding whether exist junctions or endpoints</para>
/// </summary>
/// <param name="img">Matrix</param>
/// <param name="x"></param>
/// <param name="y"></param>
bool detechNeigbourVertices(const cv::Mat &img, int x, int y)
{

    for (int ti = x - 1; ti < x + 2; ti++)
    {
        for (int tj = y - 1; tj < y + 2; tj++)
        {
            if (img.at<cv::Vec3b>(ti, tj) == RED || img.at<cv::Vec3b>(ti, tj) == GREEN)
                return true;
        }
    }
    return false;
}

class Vertex
{
public:
    Vertex(cv::Point p, int d) : pos(p), degree(d){};
    cv::Point pos;
    int degree;
};

class Edge
{
public:
    // Edge(int id, cv::Point s, cv::Point e) :index(id), start(s), end(e) {};
    // int index = 0;
    cv::Point start;
    cv::Point end;
};

class Curve
{
public:
    // int index = 0;
    cv::Point start;
    cv::Point end;
    std::vector<cv::Point> pixels;
};

enum
{
    VISITED = 0,
    UNVISITED = 1
};

std::vector<cv::Point> junctions;
std::vector<cv::Point> endpoints;
std::vector<Vertex> vertices;
cv::Mat verticesflag;
std::vector<Curve> curves;
std::vector<Edge> edges;

int junctionscount = 0;
int endpointscount = 0;
int verticescount = 0;
int markscount = 0;

/// <summary>
/// <para>Prepare the topology structure:</para>
/// <para>mark all stroke pixels, store all junctions and endpoints</para>
/// </summary>
/// <param name="img"></param>
void preprocess(cv::Mat &img)
{

    // initial no-black pixels are visited(0)
    verticesflag = cv::Mat::zeros(img.size(), CV_8UC1); // 1 channel

    for (int i = 1; i < img.rows - 1; i++)
    {
        for (int j = 1; j < img.cols - 1; j++)
        {

            if (img.at<cv::Vec3b>(i, j) != BLACK)
                continue;
            verticesflag.ptr<uchar>(i)[j] = UNVISITED;
            markscount++;
            int cn = countNeighbours(img, i, j);
            cv::Point tempP(i, j);
            Vertex tempV = Vertex(tempP, cn);

            if (cn > 2)
            { // 2 more neighbours

                uchar mask = 0; // [0: nw ...]

                // set 3*3 mark except "me"
                mask |= (int)(img.at<cv::Vec3b>(i - 1, j - 1) == BLACK) << 7; // nw
                mask |= (int)(img.at<cv::Vec3b>(i - 1, j) == BLACK) << 6;     // no
                mask |= (int)(img.at<cv::Vec3b>(i - 1, j + 1) == BLACK) << 5; // ne
                mask |= (int)(img.at<cv::Vec3b>(i, j - 1) == BLACK) << 4;     // we
                mask |= (int)(img.at<cv::Vec3b>(i, j + 1) == BLACK) << 3;     // ea
                mask |= (int)(img.at<cv::Vec3b>(i + 1, j - 1) == BLACK) << 2; // sw
                mask |= (int)(img.at<cv::Vec3b>(i + 1, j) == BLACK) << 1;     // so
                mask |= (int)(img.at<cv::Vec3b>(i + 1, j + 1) == BLACK);      // se

                // match 3 junction patterns
                if (((mask & 0x8A) == 0x8A) || ((mask & 0x32) == 0x32) || ((mask & 0x4C) == 0x4C) || ((mask & 0x51) == 0x51))
                {
                    if (!detechNeigbourVertices(img, i, j))
                    {
                        junctionscount++;
                        img.at<cv::Vec3b>(i, j) = RED;
                        junctions.push_back(tempP);
                        vertices.push_back(tempV);
                    }
                }
                if (((mask & 0x1A) == 0x1A) || ((mask & 0x58) == 0x58) || ((mask & 0x52) == 0x52) || ((mask & 0x4A) == 0x4A))
                {
                    if (!detechNeigbourVertices(img, i, j))
                    {
                        junctionscount++;
                        img.at<cv::Vec3b>(i, j) = RED;
                        junctions.push_back(tempP);
                        vertices.push_back(tempV);
                    }
                }
                if (((mask & 0x31) == 0x31) || ((mask & 0x45) == 0x45) || ((mask & 0xA2) == 0xA2) || ((mask & 0x8C) == 0x8C))
                {
                    if (!detechNeigbourVertices(img, i, j))
                    {
                        junctionscount++;
                        img.at<cv::Vec3b>(i, j) = RED;
                        junctions.push_back(tempP);
                        vertices.push_back(tempV);
                    }
                }
            }
            else if (countNeighbours(img, i, j) == 1)
            {
                if (!detechNeigbourVertices(img, i, j))
                {
                    endpointscount++;
                    img.at<cv::Vec3b>(i, j) = GREEN;
                    // img.at<cv::Vec3b>(i, j) = RED;
                    endpoints.push_back(tempP);
                    vertices.push_back(tempV);
                }
            }
        }
    }

#ifdef Debug
    for (int d = 0; d < junctions.size(); d++)
    {
        cout << junctions[d] << " junction has " << countNeighbours(img, junctions[d].x, junctions[d].y) << " neighbours" << endl;
        circle(img, Point(junctions[d].y, junctions[d].x), 4, Scalar(255, 0, 0), -1); //-1 means filling
    }
    for (int d = 0; d < endpoints.size(); d++)
    {
        cout << endpoints[d] << " endpoint has " << countNeighbours(img, endpoints[d].x, endpoints[d].y) << " neighbours" << endl;
        circle(img, Point(endpoints[d].y, endpoints[d].x), 4, Scalar(0, 255, 0), -1); //-1 means filling
    }
    std::cout << junctions.size() << " red junctions and " << endpoints.size() << " endpoints have been painted" << std::endl;
#endif
    cout << "There are " << markscount << " numbers of 1" << endl;
    for (int d = 0; d < vertices.size(); d++)
    {
        cout << "The number " << d << " (" << vertices[d].pos.x << ", " << vertices[d].pos.y << ")"
             << " Vertice has " << vertices[d].degree << " neighbours" << endl;
    }
}

bool isSurroundNeighbour(cv::Point p1, cv::Point p2)
{
    if (abs(p1.x - p2.x) + abs(p1.y - p2.y) <= 1)
        return true;
    return false;
}

bool findExistIn(const std::vector<cv::Point> &arr, const cv::Point &p)
{
    for (int i = 0; i < arr.size(); i++)
        if (arr[i].x == p.x && arr[i].y == p.y)
            return true;
    return false;
}

bool findNearestVertices(cv::Point &p)
{
    for (int i = p.x - 1; i < p.x + 2; i++)
    {
        for (int j = p.y - 1; j < p.y + 2; j++)
        {
            if (findExistIn(junctions, Point(i, j)) || findExistIn(endpoints, Point(i, j)))
            {
                p = Point(i, j);
                return true;
            }
        }
    }
    return false;
}

bool checkTopologyData(const cv::Mat &img)
{

    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);
    // check preprocess
    if (verticesflag.empty())
    {
        std::cerr << "ERROR: vertexFlag DATA EMPTY! " << std::endl;
        return false;
    }
    else if (junctions.size() < 1 || endpoints.size() < 1)
        return false;
    // int nRows = img.rows;
    // int nCols = img.cols;
    // if (img.isContinuous()) {
    //     nCols *= nRows;
    //     nRows = 1;
    // }

    return true;
}

std::queue<cv::Point> mainqueue; // execute Queue
void createTopology(cv::Mat &img)
{

    if (!checkTopologyData(img))
        return;
    else
        std::cout << "ERROR: vertexFlag DATA EMPTY! " << std::endl;

    int cx = vertices[0].pos.x;
    int cy = vertices[0].pos.y;
    mainqueue.push(Point(cx, cy));

    while (!mainqueue.empty())
    {

        Curve curCurve;
        Edge curEdge;
        cv::Point nextpos;
        cv::Point curPos = mainqueue.front();
        mainqueue.pop();

        if (findExistIn(junctions, curPos) || findExistIn(endpoints, curPos))
        {
            if (curCurve.pixels.size() < 1)
            {
                curCurve.start = curPos;
                curEdge.start = curPos;
                curCurve.pixels.push_back(curPos);
                for (int i = curPos.x - 1; i < curPos.x + 2; i++)
                {
                    for (int j = curPos.y - 1; j < curPos.y + 2; j++)
                    {
                        if (verticesflag.ptr<uchar>(i)[j] == UNVISITED)
                            mainqueue.push(Point(i, j));
                    }
                }
            }
            else
            {
                curCurve.end = curPos;
                curEdge.end = curPos;
                curves.push_back(curCurve);
                edges.push_back(curEdge);
            }
        }
        else
        {
            if (findNearestVertices(nextpos))
            {
                curCurve.pixels.push_back(curPos);
                mainqueue.push(nextpos);
            }

            while (verticesflag.ptr<uchar>(curPos.x)[curPos.y] == UNVISITED)
            {

                for (int i = curPos.x - 1; i < curPos.x + 2; i++)
                {
                    for (int j = curPos.y - 1; j < curPos.y + 2; j++)
                    {
                        if (findExistIn(junctions, Point(i, j)) || findExistIn(endpoints, Point(i, j)))
                        {
                            if (i == startpos.x && j == startpos.y)
                            {
                                curCurve.pixels.push_back(startpos);
                                curCurve.start = startpos;
                                curEdge.start = startpos;
                            }
                            else
                            {
                                Point v2(i, j);
                                curCurve.pixels.push_back(v2);
                                curCurve.end = v2;
                                curEdge.end = v2;
                                curves.push_back(curCurve);
                                edges.push_back(curEdge);
                                // clear
                                curCurve.pixels.clear();
                                curCurve.start = Point(0, 0);
                                curCurve.end = Point(0, 0);
                                curEdge.start = Point(0, 0);
                                curEdge.end = Point(0, 0);
                            }
                        }
                        else
                        {
                            if (verticesflag.ptr<uchar>(i)[j] == UNVISITED)
                            {
                                curCurve.pixels.push_back(curPos);
                                nextpos = Point(i, j);
                            }
                        }
                    }
                }
            }
        }

        verticesflag.ptr<uchar>(curPos.x)[curPos.y] = VISITED;
    }

#ifdef Debug

#endif
}

#ifdef DEBUG
void draw_topology_answer(cv::Mat &img)
{
    line(img, Point(junctions[0].y, junctions[0].x), Point(junctions[1].y, junctions[1].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[5].y, junctions[5].x), Point(junctions[1].y, junctions[1].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[0].y, junctions[0].x), Point(junctions[2].y, junctions[2].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[3].y, junctions[3].x), Point(junctions[2].y, junctions[2].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[3].y, junctions[3].x), Point(junctions[2].y, junctions[2].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[3].y, junctions[3].x), Point(junctions[4].y, junctions[4].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[4].y, junctions[4].x), Point(junctions[7].y, junctions[7].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[6].y, junctions[6].x), Point(junctions[5].y, junctions[5].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[7].y, junctions[7].x), Point(junctions[8].y, junctions[8].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[10].y, junctions[10].x), Point(junctions[8].y, junctions[8].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[6].y, junctions[6].x), Point(junctions[9].y, junctions[9].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[11].y, junctions[11].x), Point(junctions[9].y, junctions[9].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[11].y, junctions[11].x), Point(junctions[12].y, junctions[12].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[10].y, junctions[10].x), Point(junctions[12].y, junctions[12].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[6].y, junctions[6].x), Point(endpoints[0].y, endpoints[0].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[10].y, junctions[10].x), Point(endpoints[1].y, endpoints[1].x), Scalar(0, 0, 255), 2);
    line(img, Point(junctions[12].y, junctions[12].x), Point(endpoints[2].y, endpoints[2].x), Scalar(0, 0, 255), 2);
}
#endif // DEBUG

double lambda = 0.f, miu = 0.f;
int lambda_slider = 0, miu_slider = 0;
const float lambda_slider_max = 10, miu_slider_max = 10;
// scrollbar callback
static void on_lambda_trackbar(int, void *)
{
    lambda = lambda_slider / lambda_slider_max;
    std::cout << "lambda: " << lambda << std::endl;
}
static void on_miu_trackbar(int, void *)
{
    miu = (double)miu_slider / miu_slider_max;
    std::cout << "miu: " << miu << std::endl;
}

void on_mouse(int event, int x, int y, int flags, void *param)
{
    Mat &img = *((Mat *)(param));

    if (event == 3) // middle button
    {
        // preprocess(img);
        cv::imwrite("testresult.png", img);
        exit(0);
    }
}

int main(int argc, char **argv)
{

    image = cv::imread(argv[1], 1);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    int s1 = image.rows, s2 = image.cols, id = 0;
    double p1 = 600.0 / s1;
    double p2 = 600.0 / s2;
    if (p1 < p2)
        p1 = p2;
    cv::resize(image, image, cv::Size(p1 * image.cols, p1 * image.rows));
    cv::namedWindow("Show Image", cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
    setMouseCallback("Show Image", on_mouse, &image);
    // brushcol = CGAL::Color(0, 0, 255);

    // Waiting for implement
    char TrackbarName[50];
    sprintf(TrackbarName, "lambda: ", lambda);
    createTrackbar(TrackbarName, "Show Image", &lambda_slider, lambda_slider_max, on_lambda_trackbar);
    char TrackbarName1[50];
    sprintf(TrackbarName1, "miu: ", miu);
    createTrackbar(TrackbarName1, "Show Image", &miu_slider, miu_slider_max, on_miu_trackbar);

    preprocess(image);
    createTopology(image); // draw_topology(image);
    cv::imshow("Show Image", image);

    cv::waitKey();
    return 0;
}

/*
if (curCurve.pixels.size() < 1) {

    if (findExistIn(junctions, curPos) || findExistIn(endpoints, curPos)) { // vertex
        for (int i = curPos.x - 1; i < curPos.x + 2; i++) {
            for (int j = curPos.y - 1; j < curPos.y + 2; j++) {
                if (vertexFlags.ptr<uchar>(i)[j] == UNVISITED) {
                    vertexFlags.ptr<uchar>(i)[j] = VISITED;
                    Q.push(Point(i, j));
                }
            }
        }
        //curCurve.pixels.push_back(curPos); // start point
        //curCurve.v1 = curPos;
        //curEdge.v1 = curPos;
    }
    else { // vertex neighbour
        //for (int i = curPos.x - 1; i < curPos.x + 2; i++) {
        //    for (int j = curPos.y - 1; j < curPos.y + 2; j++) {
        //        if (findExistIn(junctions, Point(i,j)) || findExistIn(endpoints, Point(i, j))) {
        //            curCurve.pixels.push_back(Point(i,j)); // may be the start point
        //            curCurve.v1 = Point(i, j);
        //            curEdge.v1 = Point(i, j);
        //        }
        //    }
        //}
        curCurve.pixels.push_back()
            curCurve.pixels.push_back(curPos); // may be the second point
    }
    cout << "----" << curves.size() << " || " << edges.size() << "----" << endl;

    // traverse single stroke
    while (!curCurve.pixels.empty()) {

        Point nextPos;
        Point tempPos = curCurve.pixels.back();

        vertexFlags.ptr<uchar>(tempPos.x)[tempPos.y] = VISITED; // set 0 when arrived
        bool detechVertex = false;
        Point neighborVertex;

        for (int i = tempPos.x - 1; i < tempPos.x + 2; i++) {
            for (int j = tempPos.y - 1; j < tempPos.y + 2; j++) {

                if (vertexFlags.ptr<uchar>(i)[j] == UNVISITED) {
                    nextPos.x = i;
                    nextPos.y = j;
                    break;
                }
                else if (findExistIn(junctions, Point(i, j)) || findExistIn(endpoints, Point(i, j))) {
                    if ()
                        if (i == tempPos.x && j == tempPos.y) {
                            if (!Q.empty()) {
                                nextPos = Q.front();
                                Q.pop();
                            }
                        }
                        else if (curCurve.v1.x != i && curCurve.v1.y != j) {
                            detechVertex = true;
                            neighborVertex.x = i;
                            neighborVertex.y = j;
                            break;
                        }
                }
            }
            if (detechVertex)
                break;
        }
        if (detechVertex) {
            curCurve.v2 = neighborVertex;
            curEdge.v2 = neighborVertex;
            curCurve.pixels.push_back(tempPos);
            curCurve.pixels.push_back(neighborVertex);
            cout << tempPos << endl;
            cout << neighborVertex << endl;
            curves.push_back(curCurve);
            edges.push_back(curEdge);

            curCurve.pixels.clear();
            Q.push(neighborVertex);
        }
        else {
            cout << tempPos << endl;
            curCurve.pixels.push_back(nextPos);
            tempPos.x = nextPos.x;
            tempPos.y = nextPos.y;
        }



    }
}*/

//
//// output color
// int pc = 0, bc = 0, tpc = 0;
// for (int i = 0; i < image.rows; i++) {
//     for (int j = 0; j < image.cols; j++) {
//         if (image.at<cv::Vec3b>(i, j)[0] < 255 || image.at<cv::Vec3b>(i, j)[1] < 255 || image.at<cv::Vec3b>(i, j)[2] < 255) {
//             pc++;
//             if (image.at<cv::Vec3b>(i, j) == BLACK)
//             {
//                 bc++;
//             }
//         }
//     }
// }
// std::cout << pc << " points have been painted" << std::endl;
// std::cout << bc << " black points have been painted" << std::endl;

//
// std::vector<int> myvec = { 10, 5, 7, 8 };
// cout << "test size: " << myvec.size() << endl;
// cout << "test empty: " << myvec.empty() << endl;
// cout << "test capacity" << myvec.capacity() << endl;
//
// myvec.clear();
// cout << "test size: " << myvec.size() << endl;
// cout << "test empty: " << myvec.empty() << endl;
// cout << "test capacity" << myvec.capacity() << endl;
//
// myvec.swap(vector<int>());
// cout << "test size: " << myvec.size() << endl;
// cout << "test empty: " << myvec.empty() << endl;
// cout << "test capacity" << myvec.capacity() << endl;
