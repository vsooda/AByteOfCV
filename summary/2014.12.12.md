各位好，

这封邮件简单介绍目前手头上的工作，有一些需要确定优先级，请帮忙确定。
## Overview
<!--总览， 当前所有项目状态，包括进行中， 新增等等-->
<!--**Mou**, the missing Markdown editor for *web developers*. -->
<!--[在线编辑][1]  [1]: https://www.zybuluo.com/mdeditor-->
<!-- ^引用和各种标志可以嵌套使用 -->

课题 					| 进度 					| 状态
:----------- 			| :-----------: 		| -----------:
frontalization			<br/> 正脸化。 将偏转脸的<br/>正脸调整为正脸						| 了解摄影模型，3D重建原理。测试正脸化算法  						| **新增**，进行中
face alignment			<br/>特征点配准									| 等待乔源标注数据。关注新算法，测试改进算法 						| 进行中
face detect	 			<br/>人脸检测									| 加速 											| 进行中
makeup					<br/> 化妆										| matlab demo 对照片处理效果不错。<br/>对于贴图无法获取特征点，自己实现wls滤波效率低于matlab						| 暂停
illumination normalize	<br/> 光照均衡									| 初步版本准备上线。<br/>改进版本暂停，原因同上 						| 暂停
stylization				<br/>风格化										|已有算法效果不够理想  							| 关注

## frontalization
将特征点映射到3D中，从而估计模型的姿态，并实现**正脸化**。
<!-- 本周主要学习成像模型，camera模型，2d，3d对应相关算法。-->
`从论文结果来看，效果不错。松城，智华对3d重建比较熟悉，麻烦确定一下理论是否可行。`
## face alignment
主要参考RCPR[^robust], 3000fps[^fps], SDM[^SDM], ESR[^ESR], Face detection, pose estimation, and landmark localization[^zhu2012]。 `拟采用RCPR的特征提取算法对ESR进行改进。`
## face detect
dlib开源库相对于opencv已有改善。为了实时人脸跟踪，需要更快的人脸检测算法。`拟采用fast feature pyramid方法对其进行改进`。
## 人脸特征点建库
1. 自建数据库部分：主要乔源在做，当前已经粗略标注1000张。精确标注300张。
2. 网上资源：浙大周昆公布了1万多张74个点标注结果，大致与我们相同。

从训练结果来看，自建数据库对我们自己数据监测较好，对其他数据集检测效果不好。
如果使用周昆数据库，可以使我们标注库更丰富，从而使结果更佳鲁棒。问题是：需要额外工作量。
`需要明确结论确定是否使用他们的数据集。`
## 化妆
在matlab上对于重建前的照片进行化妆，将一张图片的化妆效果传到另一张取得不错的效果。在贴图上无法定位特征点，效果不够理想。
且wls的c++实现效率无法满足移动设备要求，需要改进。
`这部分工作还很多。优先级不高的话，先放下`
## 光照均衡
简单版本已验收。
增强版本算法原理与化妆基本一致。遇到的问题同上。

**考虑不周之处，请指出。**

谢谢！

刘守达

<!-- 
###人脸检测[^LaTeX]

```
* sldkfjl [^LaTeX]
**lsdkjfl** [^LaTeX]
* sldkfj  
```
>
sdfsdfsdf
sdfsdf[^LaTeX]

	sdasdas[^LaTeX]
	sadasd
	

##特征点识别
##光照处理
##化妆
##正脸化

[在线编辑][1]

[^dd]: sdfklj 
[^LaTeX]: dsfds
[1]: https://www.zybuluo.com/mdeditor
-->
[^robust]: Burgos-Artizzu X P, Perona P, Dollár P. Robust face landmark estimation under occlusion[C]//Computer Vision (ICCV), 2013 IEEE International Conference on. IEEE, 2013: 1513-1520.
[^fps]: Ren S, Cao X, Wei Y, et al. Face Alignment at 3000 FPS via Regressing Local Binary Features[J].
[^SDM]: Xiong X, De la Torre F. Supervised descent method and its applications to face alignment[C]//Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference on. IEEE, 2013: 532-539.
[^zhu2012]: Zhu X, Ramanan D. Face detection, pose estimation, and landmark localization in the wild[C]//Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012: 2879-2886.
[^ESR]: Cao X, Wei Y, Wen F, et al. Face alignment by explicit shape regression[J]. International Journal of Computer Vision, 2014, 107(2): 177-190.