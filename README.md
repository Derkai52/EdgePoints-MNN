## EdgePoints + MNN

本示例为C++复现版本，并在 Nvidia Jetson Orin NX 16GB 平台部署

### 下载
```bash
mkdir -p ws_edgepoints/src
cd ws_edgepoints/src
git clone https://github.com/Derkai52/EdgePoints-MNN.git

catkin_make
```

### 运行示例
```bash
roslaunch edgepoints match.launch
```