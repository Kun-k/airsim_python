### airsim_python

## 目录结构：

```
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
```
