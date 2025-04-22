#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class EdgePointsNode {
public:
    EdgePointsNode(ros::NodeHandle& nh)
        : it_(nh)
    {
        sub1_ = it_.subscribe("/camera1/image", 1, &EdgePointsNode::imageCallback1, this);
        sub2_ = it_.subscribe("/camera2/image", 1, &EdgePointsNode::imageCallback2, this);

        pub1_ = it_.advertise("/edgepoints/image1", 1);
        pub2_ = it_.advertise("/edgepoints/image2", 1);
    }

    void imageCallback1(const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "mono8");
            pub1_.publish(cv_ptr->toImageMsg());
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void imageCallback2(const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "mono8");
            pub2_.publish(cv_ptr->toImageMsg());
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

private:
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub1_, sub2_;
    image_transport::Publisher pub1_, pub2_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "edgepoints_node");
    ros::NodeHandle nh;

    EdgePointsNode node(nh);
    ros::spin();
    return 0;
}