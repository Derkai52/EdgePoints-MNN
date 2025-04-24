#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "edgepoints.h"
#include <yaml-cpp/yaml.h>

// #include "../estimator/parameters.h"

using namespace nvinfer1;


torch::Tensor simple_nms(const torch::Tensor& scores, int nms_radius) {
    TORCH_CHECK(nms_radius >= 0, "NMS radius must be non-negative.");

    auto max_pool = [&](const torch::Tensor& x) {
        return torch::nn::functional::max_pool2d(
            x,
            torch::nn::functional::MaxPool2dFuncOptions(nms_radius * 2 + 1)
                .stride(1)
                .padding(nms_radius)
        );
    };

    auto zeros = torch::zeros_like(scores);
    auto max_mask = scores == max_pool(scores);

    for (int i = 0; i < 2; ++i) {
        auto supp_mask = max_pool(max_mask.to(torch::kFloat)) > 0;
        auto supp_scores = torch::where(supp_mask, zeros, scores);
        auto new_max_mask = supp_scores == max_pool(supp_scores);
        max_mask = max_mask | (new_max_mask & (~supp_mask));
    }

    return torch::where(max_mask, scores, zeros);
}


EdgePoints::EdgePoints():dev(torch::kCUDA)
{
    std::string config_path = "/home/emnavi/ws_edgepoints/src/EdgePoints-MNN/launch/config.yaml";
    std::string engineFilePath = "/home/emnavi/ws_edgepoints/src/EdgePoints-MNN/model/EdgePoint.engine";
    // std::string engineFilePath = "/home/emnavi/ws_edgepoints/src/EdgePoints-MNN/model/xfeat.engine";
    YAML::Node config = YAML::LoadFile(config_path);

    // XFeat params
    inputH = config["image_height"].as<int>();
    inputW = config["image_width"].as<int>();
    top_k = config["max_keypoints"].as<int>();

    // NMS params
    threshold = config["threshold"].as<float>();
    kernel_size = config["kernel_size"].as<int>();

    //Softmax params
    softmaxTemp = config["softmaxTemp"].as<float>();

    // Load and initialize the engine
    loadEngine(engineFilePath);
    context = std::unique_ptr<IExecutionContext, DestroyObjects> (engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }

    // Image height and width after image preprocessing to make it compatible with the TensorRT engine.
    _H = (inputH/32)*32;
    _W = (inputW/32)*32;

    //Size of output of TensorRT engine
    outputH = _H/8;
    outputW = _W/8;

    //Scale correction factor
    rh = static_cast<float>(inputH) / static_cast<float>(_H);
    rw = static_cast<float>(inputW) / static_cast<float>(_W);

    //Get engine bindings
    inputIndex = engine->getBindingIndex("image");
    descIndex = engine->getBindingIndex("descriptor");
    scoresIndex = engine->getBindingIndex("scores");
    // heatmapIndex = engine->getBindingIndex("heatmap");

    //Sparse interpolator for post-processing outputs
    _nearest = InterpolateSparse2D("nearest");
	bilinear = InterpolateSparse2D("bilinear");

}

void EdgePoints::detectAndCompute(const cv::Mat& img, torch::Tensor& keypoints, torch::Tensor& descriptors)
{

    // Preprocess input image and convert to Tensor on GPU
    torch::Tensor input_Data = preprocessImages(img);

    batchSize = input_Data.size(0);

    // Variables to store output from TensorRT engine
    // featsData = torch::empty({batchSize,64,outputH,outputW}, torch::device(dev).dtype(torch::kFloat32));
    // keypointsData = torch::empty({batchSize, 65, outputH, outputW}, torch::device(dev).dtype(torch::kFloat32));
    // heatmapData = torch::empty({batchSize, 1, outputH, outputW}, torch::device(dev).dtype(torch::kFloat32));
    descData = torch::empty({1, 64, 60, 80}, torch::device(dev).dtype(torch::kFloat32));
    scoresData  = torch::empty({1, 1, 480, 640}, torch::device(dev).dtype(torch::kFloat32));

    // Create buffer to store input and outputs of TensorRT engine
    void* buffers[3]; 
    buffers[inputIndex] = input_Data.data_ptr();
    buffers[scoresIndex] = descData.data_ptr();
    buffers[descIndex] = scoresData.data_ptr();

    // buffers[inputIndex] = input_Data.data_ptr();
    // buffers[scoresIndex] = featsData.data_ptr();
    // buffers[keypointsIndex] = keypointsData.data_ptr();
    // buffers[heatmapIndex] = heatmapData.data_ptr();

    // Run inference on TensorRT engine
    context->executeV2(buffers);

    descData = torch::nn::functional::normalize(descData,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    /////////////////////////////////////// detect_keypoints ///////////////////////////

    auto b = scoresData.size(0);
    auto h = scoresData.size(2);
    auto w = scoresData.size(3);
    auto scores_nograd = scoresData.detach();

    auto nms_scores = simple_nms(scores_nograd, 2);
    // std::cout << nms_scores << std::endl;

    // remove border
    int radius = 2;
    nms_scores.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, radius + 1)}, 0);
    nms_scores.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, radius + 1)}, 0);
    nms_scores.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(h - radius, h)}, 0);
    nms_scores.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(w - radius, w)}, 0);



    int top_k = 1000;
    int n_limit=20000;
    float scores_th = 0.2;
    // Flatten for topk or mask-based selection
    auto scores_flat = scores_nograd.view({b, -1});
    auto nms_flat = nms_scores.view({b, -1});
    std::vector<torch::Tensor> indices_keypoints;



    if (top_k > 0) {
        auto topk = std::get<1>(torch::topk(nms_scores.view({b, -1}), top_k, 1, true, true));
        indices_keypoints = topk.unbind(0);
    } else {
        auto masks = nms_scores > scores_th;
        auto scores_view = scores_nograd.view({b, -1});
        for (int i = 0; i < b; ++i) {
            auto mask = masks[i].view({-1});
            auto indices = torch::nonzero(mask).squeeze();
            if (indices.numel() > n_limit) {
                auto kpt_scores = scores_view[i].index_select(0, indices);
                auto sorted = std::get<1>(kpt_scores.sort(-1, /*descending=*/true));
                indices = indices.index_select(0, sorted.slice(0, 0, n_limit));
            }
            indices_keypoints.push_back(indices);
        }
    }

    std::vector<torch::Tensor> keypoints_vec, scoredispersitys_vec, kptscores_vec;

    // sub
    int kernel_size = 2 * radius + 1;
    float temperature = 0.1f;
    torch::Tensor x = torch::linspace(-radius, radius, kernel_size);
    std::vector<torch::Tensor> mesh = torch::meshgrid({x, x}, /*indexing=*/"ij");
    torch::Tensor mesh_y = mesh[0];  // shape: [kernel_size, kernel_size]
    torch::Tensor mesh_x = mesh[1];  // shape: [kernel_size, kernel_size]

    // Step 3: stack and reshape (H, W) => [kernel_size*kernel_size, 2]
    torch::Tensor hw_grid = torch::stack({mesh_y, mesh_x}, /*dim=*/-1)  // shape: [k, k, 2]
                                    .reshape({-1, 2});                  // shape: [k*k, 2]

    auto patches = torch::nn::functional::unfold(scoresData, torch::nn::functional::UnfoldFuncOptions({kernel_size, kernel_size}).padding(radius));

    hw_grid = hw_grid.to(patches.device());
    for (int i = 0; i < b; ++i) {
        auto patch = patches[i].transpose(0, 1);  // (H*W) x kernel^2
        auto indices_kpt = indices_keypoints[i];  // [M]
        auto patch_scores = patch.index_select(0, indices_kpt);  // M x (kernel^2)
        auto max_v = std::get<0>(patch_scores.max(1, true)).detach();  // M x 1
        auto x_exp = (patch_scores - max_v) / temperature;
        x_exp = x_exp.exp();  // M x (kernel^2)

        auto xy_residual = torch::matmul(x_exp, hw_grid) / x_exp.sum(1, true);  // M x 2
        auto norm_dist2 = torch::norm((hw_grid.unsqueeze(0) - xy_residual.unsqueeze(1)) / radius, 2, -1).pow(2);
        auto scoredispersity = (x_exp * norm_dist2).sum(1) / x_exp.sum(1);  // [M]

        auto xy_base = torch::stack({indices_kpt.remainder(w), indices_kpt.div(w)}, 1).to(torch::kFloat);
        auto keypoints_xy = xy_base + xy_residual;
        keypoints_xy = keypoints_xy / torch::tensor({(float)(w-1), (float)(h-1)}).to(keypoints_xy.device()) * 2 - 1;

        auto kptscore = torch::nn::functional::grid_sample(
            scoresData[i].unsqueeze(0),  // 1xCxHxW
            keypoints_xy.view({1, 1, -1, 2}),
            torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(true)
        )[0][0][0];  // CxN -> just first channel

        keypoints_vec.push_back(keypoints_xy);
        scoredispersitys_vec.push_back(scoredispersity);
        kptscores_vec.push_back(kptscore);
    }


    keypoints = torch::cat(keypoints_vec);
    torch::Tensor scoredispersitys = torch::cat(scoredispersitys_vec);
    torch::Tensor kptscores = torch::cat(kptscores_vec);



    ////////////   Sample_descriptor   /////////////////////////////////////////////////////////
    std::vector<torch::Tensor> descriptors_v;
    int64_t batch_size = descData.size(0);
    int64_t channel = descData.size(1);
    int64_t height = descData.size(2);
    int64_t width = descData.size(3);

    
    for (int64_t i = 0; i < batch_size; ++i) {
        torch::Tensor kpts = keypoints; // Nx2, normalized [-1, 1] (x, y)

        // Reshape to 1x1xN x 2 (NCHW format with N points)
        torch::Tensor grid = kpts.view({1, 1, -1, 2}); // 1x1xN x 2

        // Select one image feature map: 1xC x H x W
        torch::Tensor fmap = descData[i].unsqueeze(0);  // 1 x C x H x W

        // Sample descriptors using bilinear interpolation
        torch::Tensor desc = torch::nn::functional::grid_sample(
            fmap,
            grid,
            torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(true)
        );  // Output: 1 x C x 1 x N

        // Remove extra dims -> C x N
        desc = desc.squeeze(0).squeeze(1);  // C x N

        // Normalize along channel dimension
        desc = torch::nn::functional::normalize(desc, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));

        // Transpose to N x C
        descriptors_v.push_back(desc.t());  // N x C
    }
    descriptors = torch::cat(descriptors_v);


    torch::Tensor scale = torch::tensor({float(img.cols - 1), float(img.rows - 1)}, keypoints.options());
    keypoints = (keypoints + 1.0) / 2.0 * scale;  // Now in [0, W-1] x [0, H-1]

    keypoints.cpu();
    descriptors.cpu();
}


torch::Tensor EdgePoints::preprocessImages(const cv::Mat& img)
{
    torch::Tensor img_tensor = MatToTensor(img);

    // img_tensor = torch::nn::functional::interpolate(
    //     img_tensor,
    //     torch::nn::functional::InterpolateFuncOptions()
    //         .size(std::vector<int64_t>{_H, _W})
    //         .mode(torch::kBilinear)
    //         .align_corners(false)
    // );


    return img_tensor;
}

std::vector<char> EdgePoints::readEngineFile(const std::string& engineFilePath)
{
    std::ifstream file(engineFilePath, std::ios::binary | std::ios::ate);
    if(!file.is_open()){
        throw std::runtime_error("Unable to open engine file: " + engineFilePath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0,std::ios::beg);

    std::vector<char> buffer(size);
    if(!file.read(buffer.data(),size)){
        throw std::runtime_error("Unable to read engine file: " + engineFilePath);
    }
    return buffer;
}

void EdgePoints::loadEngine(const std::string& engineFilePath)
{
    std::vector<char> engineData = readEngineFile(engineFilePath);

    runtime = std::unique_ptr<IRuntime,DestroyObjects>(createInferRuntime(gLogger));

    if(!runtime){
        throw std::runtime_error("Unable to create TensorRT runtime");
    }

    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* rawEngine = runtime->deserializeCudaEngine(engineData.data(),engineData.size());

    if(!rawEngine)
    {
        throw std::runtime_error("Unable to deserialize TensorRT engine");
    }
    engine = std::unique_ptr<ICudaEngine, DestroyObjects>(rawEngine);
}

inline torch::Tensor EdgePoints::MatToTensor(const cv::Mat& img)
{
    cv::Mat floatMat;
    img.convertTo(floatMat,CV_32F);

    CV_Assert(floatMat.isContinuous());

    int channels = floatMat.channels();
    int height = floatMat.rows;
    int width = floatMat.cols;

    torch::Tensor img_tensor = torch::from_blob(floatMat.data, {1, height, width, channels}, torch::kFloat32);
    img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();
    img_tensor = img_tensor.div(255.0);                          // 加上归一化
    img_tensor = img_tensor.to(dev);

    return img_tensor;
}

torch::Tensor EdgePoints::NMS(const torch::Tensor& x, float threshold, int kernel_size)
{
    auto options = torch::TensorOptions().dtype(torch::kLong).device(dev);

    int B = x.size(0);
    int H = x.size(2);
    int W = x.size(3);
    int pad = kernel_size / 2;

    //Perform MaxPool2d
    auto local_max = torch::max_pool2d(x,kernel_size, 1, pad);
    // Compare x with local_max and threshold
    auto pos = (x == local_max) & (x > threshold);
    // Get the positions of the positive elements
    std::vector<torch::Tensor> pos_batched;
    pos_batched.reserve(B);
    for(int i = 0; i < B; i++)
    {
        pos_batched.emplace_back(pos[i].nonzero().slice(/*dim=*/1, /*start=*/1, /*end=*/torch::indexing::None).flip(-1));
    }
    // Find the maximum number of keypoints to pad the tensor
    int pad_val = 0;
    for(const auto& tensor : pos_batched)
    {
        pad_val = std::max(pad_val, static_cast<int>(tensor.size(0)));
    }
    //Pad keypoints and build (B, N, 2) Tensor
    auto pos_tensor = torch::zeros({B, pad_val, 2}, options);
    for(int b = 0; b < B; b++)
    {
        pos_tensor[b].narrow(0, 0, pos_batched[b].size(0)) = pos_batched[b];
    }
    return pos_tensor;
}

torch::Tensor EdgePoints::get_kpts_heatmap(const torch::Tensor& kpts, float softmax_temp)
{
    //Apply softmax to the input tensor with temperature
    auto scores = torch::softmax(kpts * softmax_temp, 1).narrow(1,0,64);

    //Get dimension
    int B = scores.size(0);
    int H = scores.size(2);
    int W = scores.size(3);
  
    //Perform reshaping and permutation
    auto heatmap = scores.permute({0,2,3,1}).reshape({B, H, W, 8, 8});
    heatmap = heatmap.permute({0,1,3,2,4}).reshape({B, 1, H*8, W*8});

    return heatmap;
}

void EdgePoints::match(const torch::Tensor& feats1, const torch::Tensor& feats2, torch::Tensor& idx1, torch::Tensor& idx2, double min_cossim) 
{
    auto cossim = torch::matmul(feats1, feats2.t());
    auto cossim_t = torch::matmul(feats2, feats1.t());

    auto match12 = std::get<1>(cossim.max(1));
    auto match21 = std::get<1>(cossim_t.max(1));

    idx1 = torch::arange(match12.size(0), cossim.options().device(match12.device()));
    auto mutual = match21.index({match12}) == idx1;

    if (min_cossim > 0) {
        cossim = std::get<0>(cossim.max(1));
        auto good = cossim > min_cossim;
        idx1 = idx1.index({mutual & good});
        idx2 = match12.index({mutual & good});
    } 
    else 
    {
        idx1 = idx1.index({mutual});
        idx2 = match12.index({mutual});
    }
}

void EdgePoints::create_xy(int h, int w, torch::Tensor& xy) 
{
    auto y = torch::arange(h, dev).view({-1, 1});
    auto x = torch::arange(w, dev).view({1, -1});
    xy = torch::cat({x.repeat({h, 1}).unsqueeze(-1), y.repeat({1, w}).unsqueeze(-1)}, -1).view({-1, 2});
}