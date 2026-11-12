// Not compiled while making the tutorial. Needs PCL (sudo apt install libpcl-dev).
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

using Point = pcl::PointXYZI;
using Cloud = pcl::PointCloud<Point>;

std::vector<pcl::PointIndices> detectObstacles(const Cloud::Ptr &cloud, Cloud::Ptr &obstacles)
{
    Cloud::Ptr down(new Cloud);
    pcl::VoxelGrid<Point> voxel;                       // 1. voxel downsampling
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(0.2f, 0.2f, 0.2f);
    voxel.filter(*down);

    pcl::SACSegmentation<Point> seg;                   // 2. RANSAC ground plane
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.15);
    seg.setInputCloud(down);
    pcl::PointIndices::Ptr ground(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr plane(new pcl::ModelCoefficients);
    seg.segment(*ground, *plane);

    pcl::ExtractIndices<Point> extract;                // 3. keep everything except the ground
    extract.setInputCloud(down);
    extract.setIndices(ground);
    extract.setNegative(true);
    extract.filter(*obstacles);

    pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
    tree->setInputCloud(obstacles);
    std::vector<pcl::PointIndices> clusters;           // 4. Euclidean clustering
    pcl::EuclideanClusterExtraction<Point> ec;
    ec.setClusterTolerance(0.6);
    ec.setMinClusterSize(8);
    ec.setMaxClusterSize(50000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles);
    ec.extract(clusters);
    return clusters;
}
