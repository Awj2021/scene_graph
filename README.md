# scene_graph
Some baselines of scene graph, and testing them using our model

## Training Environmetn: trace1. 
If you want to run this model with your own environment, please refer to the trace1(you could clone 
the conda environment and then motify.)

基本上，按照作者提供的文件组织方式，然后注意以下踩过的坑，就可以比较快的运行起来了。
## README.md中步骤问题
rename.py和dump.py的步骤得进行修改；

### 不要忘记txt2json这个步骤 (only for the "Action Genone" dataset)
tools/txt2json.py  


### 作者提供的数据处理步骤有问题
1. 应该先处理dump;（每次处理完这个步骤都要进行处理第二步）
2. 然后使用rename.py文件；

### 调试过程中的一些坑
1. 数据处理的步骤，需要注意。其中，因为作者使用的是sampled_frames, 因此这边需要注意。在dump.py的过程中，要选择如下命令：
```
python tools/dump_frames.py --frame_dir data/ag/sampled_frames --ignore_editlist --frames_store_type jpg --high_quality --sampled_frames
```
需要注意，sampled_frames的路径(也就是训练所需要的数据加载路径),在文件的lib.core.config文件中修改，
```
__C.HALF_FRAME_RELATIVE_PATH = 'sampled_frames'
```

### 其他：
在这个过程中，发现除了数据处理的**顺序**，数据加载的**路径**等问题外，其余的均没有问题，不要尝试修改代码中可能会报错的地方，只能治标不治本。