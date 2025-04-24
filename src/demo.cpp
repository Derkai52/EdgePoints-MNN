#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "edgepoints.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <filesystem>
#include <iostream>

std::vector<cv::DMatch> mnn_matcher(const torch::Tensor& desc1, const torch::Tensor& desc2, float threshold = 0.9f) {
    torch::Tensor sim = torch::matmul(desc1, desc2.transpose(0, 1));  // [N1 x N2]

    // 将小于 threshold 的相似度置 0
    sim = torch::where(sim < threshold, torch::zeros_like(sim), sim);

    // 获取最近邻索引
    torch::Tensor nn12 = sim.argmax(1);  // 每一行最大值索引
    torch::Tensor nn21 = sim.argmax(0);  // 每一列最大值索引

    torch::Tensor ids1 = torch::arange(sim.size(0), torch::kLong);

    // 转到 CPU 上处理（避免 GPU 上 item<int64_t> 报错）
    nn12 = nn12.cpu();
    nn21 = nn21.cpu();
    ids1 = ids1.cpu();

    std::vector<cv::DMatch> matches;
    for (int64_t i = 0; i < ids1.size(0); ++i) {
        int64_t j = nn12[i].item<int64_t>();
        if (nn21[j].item<int64_t>() == i) {
            matches.emplace_back(cv::DMatch(i, j, 0));
        }
    }

    return matches;
}

cv::Mat draw_matches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kpts1,
    const cv::Mat& img2, const std::vector<cv::KeyPoint>& kpts2,
    const std::vector<cv::DMatch>& matches) {
    cv::Mat img_match;
    cv::drawMatches(img1, kpts1, img2, kpts2, matches, img_match,
    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // cv::imshow("MNN Matches", img_match);
    // cv::waitKey(1);
    return img_match;
}

class EdgePointsNode {
public:
    EdgePointsNode(ros::NodeHandle& nh)
        : it_(nh)
    {
        sub1_ = it_.subscribe("/hik_camera_1/image", 1, &EdgePointsNode::imageCallback1, this);
        sub2_ = it_.subscribe("/hik_camera_2/image", 1, &EdgePointsNode::imageCallback2, this);

        pub1_ = it_.advertise("/edgepoints/image1", 1);
        pub2_ = it_.advertise("/edgepoints/image2", 1);
        // Synchronize to ensure all operations are complete before starting the timer
        torch::cuda::synchronize();
    }

    cv::Mat image1_;
    cv::Mat image2_;
    bool image1_ready_ = false;
    bool image2_ready_ = false;
    EdgePoints edgepoints;
    std::vector<cv::DMatch> matches, inliers;
    std::vector<cv::KeyPoint> k1,k2;

    torch::Tensor desc1, keypoints1, heatmap1, idx1;
    torch::Tensor desc2, keypoints2, heatmap2, idx2;

    void imageCallback1(const sensor_msgs::ImageConstPtr& msg) {
        try {
            // image1_ = cv_bridge::toCvShare(msg, "mono8")->image.clone();
            image1_ = cv::imread("/home/emnavi/image1.png");
            image1_ready_ = true;
            tryMatch();
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void imageCallback2(const sensor_msgs::ImageConstPtr& msg) {
        try {
            // image2_ = cv_bridge::toCvShare(msg, "mono8")->image.clone();
            image2_ = cv::imread("/home/emnavi/image2.png");
            image2_ready_ = true;
            tryMatch();
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void tryMatch() {
        if (image1_ready_ && image2_ready_) {
            cv::resize(image1_, image1_, cv::Size(640, 480));
            cv::resize(image2_, image2_, cv::Size(640, 480));

            // 开始检测
            std::cout<<"Beginning inference benchmark"<<std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            edgepoints.detectAndCompute(image1_, keypoints1, desc1);
            edgepoints.detectAndCompute(image2_, keypoints2, desc2);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout<<"Inference benchmark done "<< duration.count() << " ms"<< std::endl;
            torch::cuda::synchronize();

            TensorToKeypoints(keypoints1, k1);
            TensorToKeypoints(keypoints2, k2);
            cv::Mat output;
            cv::drawKeypoints(image1_, k1, output, cv::Scalar(0, 255, 0));
            cv::imshow("KeyPoints", output);

            auto matches = mnn_matcher(desc1, desc2);
            output = draw_matches(image1_, k1, image2_, k2, matches);

            // 发布
            std_msgs::Header header;
            header.stamp = ros::Time::now();
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", output).toImageMsg();
            pub1_.publish(msg);
    
            image1_ready_ = false;
            image2_ready_ = false;
        }
    }


private:
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub1_, sub2_;
    image_transport::Publisher pub1_, pub2_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "edgepoints");
    ros::NodeHandle nh;

    EdgePointsNode node(nh);
    ros::spin();
    return 0;
}