## EdgePoints TensorRT C++实现

本示例为C++复现版本，并在 Nvidia Jetson Orin NX 16GB 平台部署

![](./match_result.png)

- ROS Noetic
- JetPack 5.1.3
- Torch 2.1.0

### 下载
```bash
mkdir -p ws_edgepoints/src
cd ws_edgepoints/src
git clone https://github.com/Derkai52/EdgePoints-MNN.git

catkin_make
```

### 导出模型
```bash
/usr/src/tensorrt/bin/trtexec --onnx=/home/emnavi/ws_edgepoints/src/EdgePoints-MNN/model/EdgePoint.onnx --saveEngine=/home/emnavi/ws_edgepoints/src/EdgePoints-MNN/model/EdgePoint.engine
```


### 运行示例
```bash
roslaunch edgepoints match.launch
```
