#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/Color.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <bits/stdc++.h>
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
    if (event == 1)
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
    if (event == 2)
    {
        cv::Mat temim = img.clone();
        for (int i = 0; i < trii; i++)
            for (int j = 0; j < 3; j++)
                line(temim, cv::Point(tri[i][(j + 1) % 3].x(), tri[i][(j + 1) % 3].y()),
                     cv::Point(tri[i][(j + 2) % 3].x(), tri[i][(j + 2) % 3].y()), cv::Scalar(0, 0, 255), 2, cv::LINE_8);
        cv::Vec3b col = img.at<cv::Vec3b>(y, x);
        cset = 1;
        brushcol = CGAL::Color(col[0], col[1], col[2]);
        cv::imshow("Display Image", temim);
    }
    if (event == 3)
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
int main(int argc, char **argv)
{
    image = cv::imread(argv[1], 1);
    strcpy(argname, argv[1]);
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
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
    // cv::namedWindow("Display Image",cv::WINDOW_GUI_NORMAL  );
    setMouseCallback("Display Image", onmouse, &image);
    cv::imshow("Display Image", image);
    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
    cv::imshow("Result", image);
    brushcol = CGAL::Color(0, 0, 255);
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            if (image.at<cv::Vec3b>(i, j)[0] < 240)
            {
                t.insert(point(j, i));
                foreground.push_back(point(i, j));
                image.at<cv::Vec3b>(i, j) = {0, 0, 0};
            }
            else
                image.at<cv::Vec3b>(i, j) = {255, 255, 255};
    t.insert(point(0, 0));
    t.insert(point(0, image.rows - 1));
    t.insert(point(image.cols - 1, 0));
    t.insert(point(image.cols - 1, image.rows - 1));
    Delaunay::Finite_faces_iterator it;
    for (it = t.finite_faces_begin(); it != t.finite_faces_end(); it++)
    {
        Delaunay::Face_handle f = it;
        inf1.id = id;
        inf1.parent = id;
        inf1.pro = 0;
        id++;
        f->info() = inf1;
    }
    initialize();
    cv::imshow("Display Image", image);
    cv::imshow("Result", image);
    simplify();
    cv::waitKey();
    return 0;
}
