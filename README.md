### airsim_python

## 目录结构：


├── Readme.md  
├── airsim_avoid_APF.py 					 // 可运行于AirSim环境中的航路点跟踪+避障代码脚本文件
├── airsim_tracking_carrot.py  			 // 可运行于AirSim中的航路点跟踪代码脚本文件
├── UavAgent.py  								 // 无人机控制和状态获取相关接口
├── mymath.py  									// 代码中使用到的自定义数学运算
├── code_python  								 // python环境下算法实现和仿真相关文件（不依赖AirSim）
    ├── map_avoid 								// 避障算法仿真使用的地图图像
    ├── python_avoid_APF.py 			   // 人工势场法避障代码脚本文件
    ├── python_avoid_RRT.py  			  // RRT方法路径规划代码脚本文件
    ├── python_CarrotChasing.py  	    // 航路点跟踪代码脚本文件
    └── python_tracking_and_avoid.py  // 航路点跟踪+避障代码脚本文件
└── ObstacleDetectioon  					   // 障碍物检测接口
    ├── obstacles_detect.py
    ├── calculate_pose.py
    └── 说明.txt

###########目录结构描述
├── Readme.md                   // help
├── app                         // 应用
├── config                      // 配置
│   ├── default.json
│   ├── dev.json                // 开发环境
│   ├── experiment.json         // 实验
│   ├── index.js                // 配置控制
│   ├── local.json              // 本地
│   ├── production.json         // 生产环境
│   └── test.json               // 测试环境
├── data
├── doc                         // 文档
├── environment
├── gulpfile.js
├── locales
├── logger-service.js           // 启动日志配置
├── node_modules
├── package.json
├── app-service.js              // 启动应用配置
├── static                      // web静态资源加载
│   └── initjson
│       └── config.js         // 提供给前端的配置
├── test
├── test-service.js
└── tools
