> [参考链接](https://github.com/GuanyaShi/CS133b-D-Lite-Simulation)

D* Lite 算法采用典型的动态规划思路实现，对比传统的A*算法，在遇到之前规划中未遇到的障碍物时，需要重新进行路径选择。此时，A\*算法需要重新更新所有节点，然后生成路径。而D\* Lite 算法避免了对所有节点的重复更新，只对动态障碍物影响区域的节点进行更新，从而提高了效率。

![Figure_3](assets/Figure_3.png)

### 1. The first Version of D* Lite



### 2. The Second Version of D* Lite

The second version of D* Lite, uses a search method derived from D* to avoid having to reorder the priority queue. The heuristics $h(s, s')$ now need to be nonnegative and forward-backward consistent. 



