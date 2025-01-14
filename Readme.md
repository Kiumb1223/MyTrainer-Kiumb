[toc]

# Readme

> small record about my train of thought when trying to apply GNN to MOT

## 1. Brief Introduction about My Model

> Reference paper : [Recurrent graph optimal transport for learning 3D flow motion in particle tracking(Nature 2023)](https://www.nature.com/articles/s42256-023-00648-y)

![pipeline](./.assert/pipeline.bmp)

My model is an **online and graph-based** tracker. The main difference from other online and graph-based trackers is that the way of constructing graph and graph convolutions , which is innovative in the realm of MOT but also brings some limitations that still haunt on me:thinking:. So the following discussions mainly focus on the above two points.

Here comes the first discussion which is the **construction of a graph**. Specifically, the appearance features extracted by CNN Model of each object is initialized as node embedding , and the 2 dimension pixel position of each object is used to compute graph structure and edge embedding.

1. **Graph structure** : I utilize **KNN algorithm** to construct a **self-looped and directed** graph;

2. **Edge embedding** : In order to model the relative relationship between connecting nodes , I concatenate the 5 scalar below and feed them to a MLP module to encode a 16 dimension tensor as edge embedding.

$$
Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),1-DIoU(i,j))])
$$


And in order to **track each object** , there are two graphs which are built on above rules — one is based on the past trajectories and the other is based on current detections. So, the MOT problem can be converted to:star2: **graph matching problem**:star2:, which is NP complete.

Additionally, let\`s talk about my **graph convolutional operations**, which is mainly from the [paper](https://www.nature.com/articles/s42256-023-00648-y), i.e. Edgeconv. The functions goes like this, which is **minor different from the original version** : ( $[\cdot]$ means concatenation )


$$
g^{l+1}_i= \Phi(\max _{j \in \mathbb{N}_i}{f([Edge~emb~\cdot~(g_j^{l} - g_i^{l}) ])})
$$

And in order to **capture high-order discriminative features**, I rebuild the graph, which I call the `dynamic graph`, while the old one is referred to as the `static graph` to keep things clear. And I also perform the edgeconv operations on dynamic graph:


$$
g^{l+1}_i= \max _{j \in \mathbb{N}_i}{f([g_i^{l}~\cdot~(g_j^{l} - g_i^{l})])}
$$

----

The last but not least, it\`s necessary to take a glimpse of my **trajectory management strategy**. My trajectory management strategy mainly refers to **ByteTrack**. And here is my program flow chart:

<img src="./.assert/flowChart.png" alt="flowChart" style="zoom: 50%;" />

There are four states in the trajectory management strategy — **Born,Active,Sleep,Dead**. It\`s possible that `Born Trajectory` can be false positive, i.e. noise, so there are two phases in trajectory management — `Match Strategy` mainly for denoising and `Graph Matching` for matching. And the rest of the strategy is unnecessary to discuss, which is obvious in the flow chart.

----

Supplemental Details about my whole neural network, which is quite important to understand following experiments concerning **MODEL STRUCTURE**, e.g. [Sec 4.7 After more Graph Neural Network](###4.7 After more Graph Neural Network) . 

There are 5 parts of my whole model:

1. **Node Encoder**: The backbone of Node Encoder is [DenseNet121](https://arxiv.org/abs/1608.06993), initialized with pretrained weights from ImageNet, the same as `GCNNMatch`. All layers are frozen except for the last two. Additionally, I also add some extra fc layers to reduce 1024 dimensional tensor to a 32 dimensional space for the sake of addressing feature space misalignment.

   <img src="./.assert/NodeEncoder.bmp" alt="NodeEncoder" style="zoom: 33%;" />

2. **Edge Encoder**: 

   <img src="./.assert/EdgeEncoder.bmp" alt="EdgeEncoder" style="zoom: 33%;" />

3. **Static Graph Model**: When stacking more layers in GCN may lead to over-smoothing and over-squeezing, which instead makes the performance of network drop sharply. And here I simply stack three layers GCN to extract features from static graph.

   <img src="./.assert/StaticGCN.bmp" alt="StaticGCN" style="zoom: 50%;" />

   And plz pay your attention to the feature dimension and detailed parts of each static GCN layer, which are some key points of some experiments.

4. **Dynamic Graph Model**: After 1st DGCN layer , I reutilize the KNN algorithm based on **Euclidean distance** to **reconstruct** the graph in 64-dim feature space and apply DGCN layer to extract more higher-level features.

   <img src="./.assert/DynamicGCN.bmp" alt="DynamicGCN" style="zoom:50%;" />

5. **Fuse Model**:

   <img src="./.assert/fuseModel.bmp" alt="fuseModel" style="zoom:50%;" />

## 2. Experimental Settings

- Here the baseline is **GCNNMatch**，which also attempts to use GNN to solve the challenges in MOT.
- And the benchmark is **MOT17**, whose **half data for training** and **the other for validation**. 
- Apart from the half data of MOT17 for validation, there are **some private data also using for validation** (Indoor & Day  -- self dataset ) in order to test the robustness of models.

So there is some quantitative results when **GCNNMatch training on MOT17-half**  tests on **my validation set**.

|   Validation set   | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :----------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  MOT17-half(SDP)   | 48.68 | 48.43 | 49.00 | 57.39 | 46.20 | 75.72 | 56.44 | 83.56 |
| Indoor Day(YOLOv8) | 36.11 | 66.81 | 19.67 | 33.18 | 32.53 | 33.86 | 87.51 | 76.07 |

And the GCNNMatch is :warning:**extremely time-consuming**, which takes about 2 seconds to process each frame in a 30fps, 1080p video.

----

- For the purpose of  **fast training  and relatively fair comparison** , my own model also **trains on half data  of MOT17 and tests on MOT17-half**. Besides, **max epoch is set to 40, batch size is set to 16, warmup iterations are set to 500**.
- And **random seed is set to 3407** || **k is set to 2**

Here is the original quantitative results of my model without **any modification on model structure or the use of any data augmentation techniques**. 

| Validation set  | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MOT17-half(SDP) | 25.67 | 50.01 | 13.31 | 27.12 | 22.31 | 34.56 | 53.94 | 81.96 |

Obviously, my model still have a long way to go. However, what makes me proud is that my model has **a relatively fast inference speed**, which can reach up to 15 fps or so to process a 30fps, 1080p video.

----

And it is also essential to record the time consumption of training and inference in different computer power platform.

- **Training Time Analysis:**

  Here are three computing systems to train all following models of different experimental settings:

  |    Platforms    |           CPU           |      GPU       |
  | :-------------: | :---------------------: | :------------: |
  |   4090(Linux)   | Platinum 8352V (2.5GHz) | RTX4090 (8.9)  |
  |   3090(Linux)   |    i9-13900K (3GHz)     | RTX3090 (8.6)  |
  | Titan XP(Linux) |   E5-2678 v3 (2.5GHz)   | TITAN XP (6.1) |

  Additionally, there is also a minor difference in hardware architecture in the first and the second rows (Do not understand the Titan XP) : **the 4090 system uses a southbridge chip** to establish a communication link between the CPU and GPU, while the **3090 system lacks a southbridge chip**, allowing the CPU and GPU to communicate directly. As a result, **the speed on the 3090 system is faster than on the 4090 system when training.** And here is more detailed comparison:

  |    Platform     | Data <br/>Loading Time | Data <br/>Migration Time | Model Time<br>(Forward + Backward) | Total Time<br>(120 Epoch) |
  | :-------------: | :--------------------: | :----------------------: | :--------------------------------: | :-----------------------: |
  |   4090(Linux)   |        0.000 s         |          0.1 s           |               0.45 s               |        3 h 38 min         |
  |   3090(Linux)   |        0.000 s         |          0.05 s          |               0.34 s               |        2 h 15 min         |
  | Titan XP(Linux) |        0.000 s         |         0.088 s          |                2 s                 |  2 h 39 min <br>(5 GPUs)  |

  P.S. TOTAL EPOCH is set to 120 , TOTAL NUMBER of TRAIN DATASET is 2650 and BATCH SIZE is 16. Plus, whole model is trained in a single GPU, not in distributed training, except Titan XP. NUM WORKER is set to 6 in 4090 and 2 in 3090 and Titan XP.  My model is trained on 5 GPUs in Titan XP.

- **Inference Time Analysis:**

  Here is also three platforms to do evaluation (Model Inference) after training.

  |   Platforms   |           CPU           |      GPU      |
  | :-----------: | :---------------------: | :-----------: |
  | 4090 (Linux)  | Platinum 8352V (2.5GHz) | RTX4090 (8.9) |
  | 3090 (Linux)  |    i9-13900K (3GHz)     | RTX3090 (8.6) |
  | 3060 (Window) |    i5-12490F (3GHz)     | RTX3060 (8.6) |
  
  And here is the time consumption of different **resolution** video in both two platforms:
  
  |   Platforms   | 480p Video | 1080p Video | 4K video |
  | :-----------: | :--------: | :---------: | :------: |
  | 4090 (Linux)  |   25 FPS   |   13 FPS    |  9 FPS   |
  | 3090 (Linux)  |   68 FPS   |   30 FPS    |  13 FPS  |
  | 3060 (Window) |   30 FPS   |   15 FPS    |  8 FPS   |
  
  P.S. **the bigger K in KNN sets, the slower model infers.**

## 3. TO DO List

- [x] initialize the parameter of  networks plz !

- [x] attempt to use more data augmentation techniques to improve my model

  - [x] simulate low framerate
  - [x] simulate missed detections 
  - [x] simulate discontinuous trajectories

- [x] change directed graph to undirected one

- [x] change graph conv to `GraphConv-type`

- [x] add a mask based pixel distance 

  - [x] I can also statistically calculate the moving distances of different time span , framerates and resolutions

- [x] change the weight of edge 

  I can encode the some info of graph structure into the embeddings(maybe cant be so sufficient)

- [ ] Design more suitable track management ,like the lifespan of active tracks

- [x] Simplify the logic of KNN — make KNN more adaptive . It seems that variable `bt_self_loop`  is relatively useless, because self-loop only matters when there is only one object, which will lead to fatal error in my project. And this maybe save more time expense. 

- [x] do some statistical experiment about the density of the crowds in benchmarks, which is maybe helpful for the choice of best K

## 4. Experimental Records [Technique & Hyperparameters]

The quantitative results of the vanilla model  (the same results in [[Sec 1. Brief Introduction about My Model]](#1. Brief  Introduction about My Model), just repeat one more time and add some curves which can reflect something):

| Validation set  | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MOT17-half(SDP) | 25.67 | 50.01 | 13.31 | 27.12 | 22.31 | 34.56 | 53.94 | 81.96 |

<img src="./.assert/vanilaOne-index.bmp" alt="vanilaOne-index" style="zoom:25%;" />

Apart from the quantitative results of the vanilla one , let\`s focus on these two curves. The training data is 2650, which is relatively small and easily make models overfit. So I have to design more precise training strategy to avoid overfitting, where I adjust the warmup strategy and multiStep lr scheduler.

Noted that the f1 curve in evaluation phase is always above 0.9 , this f1 score is slightly different from one in MOT metric. Both of them is based on ID of tracked object, but the f1 score here is calculated in each timestamp — in other words, **the default prerequisite here is that the tracking result of each timestamp is independent**, which is quite different from one in MOT metric. Besides, **what is extremely different from real testing is that there is no missed detections in the evaluation phases , when training models.**

As we all known, the MOT problem can be viewed as a problem of **maximizing a posteriori probability** —— the tracking result of each timestamp is quite dependent on the results of previous moment. It\`s reasonable to infer that the worse tracking results from previous moment, the worse tracking results from current moment. Actually, the performance of my model is indeed like this.

### 4.1 After Data Augmentation [🎉]

> In order to avoid overfitting, it\`s overwhelmingly necessary to find out the various and valid data augmentation techniques.

![dataAugmentation ](./.assert/dataAugmentation.bmp)

There are three data augmentation techniques — **Low framerate, missed detections and discontinuous trajectories. **All of them is vividly showed in the above picture. So let\`s see the quantitative results of vanilla model after training. Oops, I changes some experimental settings. In this experiment, the total epoch is set to 120  (it maybe takes 2 hours or so in GTX3090 ), warmup iteration is set to 800 and multistep is set to 50 and 80.(Waiting to see :eyes:)

|    Conditions     |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :---------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
|    Vanilla one    |   25.67   | 50.01 |   13.31   |   27.12   |   22.31   |   34.56   | 53.94 | 81.96 |
| Data Augmentation | **35.14** | 50.43 | **24.70** | **37.29** | **30.74** | **47.37** | 56.89 | 81.94 |

<img src="./.assert/dataAug-index.bmp" alt="dataAug" style="zoom:25%;" />

:loudspeaker:  Due to special designed data augmentation techniques , like missed detections simulations, low framerate, which is beneficial for network training,  I  <strong style="color: red;">use this as the main comparison which is marked as Vanilla one<sup>*</sup>.</strong> 

### 4.2 After Undirected Graph [:tada:]

> **Graph matters in whole pipeline, which impacts on the ability of feature extraction.** Experiments in directed and undirected graphs , graphs with and without self-loops, and graph with varying K values are awaiting me.

Here is the simple illustration about the undirected graph:

![undirectedgraph](./.assert/undirectedgraph.bmp)

|       Conditions        |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :---------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
| Vanilla one<sup>*</sup> |   35.14   | 50.43 |   24.70   |   37.29   |   30.74   |   47.37   | 56.89 | 81.94 |
|    Undirected Graph     | **38.21** | 50.28 | **29.25** | **41.18** | **33.95** | **52.32** | 57.58 | 82.02 |

<img src="./.assert/UndirectedGraph-index.bmp" alt="UndirectedGraph-index" style="zoom: 25%;" />

### 4.3 After without Self-Loop [:confused:]

![](./.assert/self-loop.bmp)

Here is the simple example to show the graph with or without self loop. A special case is highlighted when only object is detected in an image (e.g., Fig3.). In such cases , a self-loop is retained for the object. This ensures the graph neural network can process the date effectively, avoiding issues that arise from an empty graph structure.

|   Conditions    |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :-------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
|  Vanilla one*   | **35.14** | 50.43 | **24.70** | **37.29** | **30.74** | **47.37** | 56.89 | 81.94 |
| *w/o* self loop |   34.82   | 50.44 |   24.23   |   36.90   |   30.43   |   46.88   | 56.83 | 81.97 |

![wo-loopGraph](./.assert/wo-loopGraph.bmp)

### 4.4 After Cosine-based Dynamic Graph [:confused:]

Considering that **the measurement based on cosine similarity in the graph match**, I change the dynamic graph based Euclidean distance to **cosine distance based** , just hoping my model can learn more discriminative features in the feature space based on cosine distance.  

|       Conditions        |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :---------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
| Vanilla one<sup>*</sup> | **35.14** | 50.43 | **24.70** | **37.29** | **30.74** | **47.37** | 56.89 | 81.94 |
|      Cosine-based       |   34.23   | 50.35 |   23.48   |   35.65   |   29.38   |   45.33   | 56.17 | 82.00 |

<img src="./.assert/cosinegraph-index.bmp" alt="cosinegraph-index" style="zoom:25%;" />

### 4.5 After Distance Mask [:tada:]

![distanceMask](./.assert/distanceMask.bmp)

What\` s distance mask here ? Actually, it is a kind of precondition which is that the movement distance of objects in adjacent frames is likely to be small, but **sometimes can be large when in low framerate video or objects moving fast**. So it\`s necessary to statistically calculate **the max movement distance** of different situations, like **different framerate, fast-moving objects or video from a moving perspective.**  And here is the statistical results in **MOT17(Train set), MOT20(Train set), DanceTrack(Train && val set)**.

![DifferentWindow](./.assert/DifferentWindow.bmp)

Noted that it seems that the **max movement distance of adjacent frames (window - 2) is more reliable**. Let\`s dive into the situations of **windowSize[2]**.

<img src="./.assert/MOT17_20.bmp" alt="MOT17" style="zoom:50%;" />

<img src="./.assert/Dancetrack.bmp" alt="Dancetrack" style="zoom:50%;" />

It seems that **the speed of object moving poses the bigger influence on the statistical experiment among all factors.**:confused: Ummm, maybe **the distance between camera and objects also matters!**

| Mask Range |         HOTA          | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-------------------------------: | :-------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   None<br>(Vanilla one<sup>*</sup>)   |   35.14   | 50.43 |   24.70   |   37.29   |   30.74   |   47.37   | 56.89 | 81.94 |
| 11 | 19.49 | 50.78 | 7.49 | 19.61 | 16.16 | 24.95 | 51.74 | 82.18 |
| 15 | 26.51 | 50.81 | 13.84 | 28.46 | 23.47 | 36.17 | 55.96 | 82.05 |
| 20 | 35.08 | 50.72 | 24.30 | 38.54 | 31.79 | 48.93 | 58.37 | 82.02 |
| 25 | 39.08 | 50.71 | 30.20 | 44.36 | 36.59 |   56.31   | 59.24 | 81.96 |
| 30 | 42.04 | 50.84 | 34.88 | 49.13 | 40.53 | 62.36 | 59.39 | 82.02 |
|             **35**             |         **44.50**         | 50.82 | 39.10 | **52.33** | **43.19** | **66.37** | 59.59 | 82.00 |
|            **50**              |         **44.40**         | 50.66 | **39.11** | **50.57** | **41.75** | **64.12** | 59.17 | 81.95 |
|             60             |    43.71    | 50.47 | 38.08 | 49.36 | 40.77 | 62.53 | 58.93 | 81.97 |
|                100                |    40.21    | 50.41 | 32.28 | 44.54 | 36.77 | 56.47 | 58.23 | 81.95 |
|                150                |    39.79    | 50.42 | 31.62 | 43.75 | 36.11 | 55.5 | 57.62 | 81.97 |
|                200​                |         37.11         | 50.33 | 27.57 | 39.74 | 32.78 | 50.45 | 57.26 | 82.00 |
|                250                | 37.75 | 50.41 | 28.49 | 40.50 | 33.41 | 51.54 | 56.92 | 81.99 |
|                300                | 36.73 | 50.27 | 27.02 | 39.20 | 32.32 | 49.81 | 56.77 | 82.00 |
|                350                | 36.66 | 50.50 | 26.79 | 38.74 | 31.96 | 49.18 | 56.69 | 81.98 |
|                400                | 35.59 | 50.34 | 25.34 | 37.54 | 30.98 | 47.63 | 56.87 | 81.99 |
|                500                | 37.09 | 50.43 | 27.48 | 39.66 | 32.72 | 50.36 | 56.55 | 81.97 |

 In order to more intuition comparison, I plot a line chart based on the above table : [without **row None (Vanilla one<sup>*</sup>)** ]

![Kfamily_metrics](./.assert/mask.png)

### 4.6 After Different K [:tada:]

According to the descriptions about construction graph in [[Sec 1. Brief Introduction about My Model]](#1. Brief Introduction about My Model), there is a hyperparameter `k` in KNN algorithm. And k is set to 2 in my vanilla model, so let`s search for best k.

Here is a picture to visually illustrates the process of graph construction with the increase of K. ( Due to the shape of each figure, the relative positions of every nodes are distorted to some extent. And **the following graph construction is based on true coordinates in original image plane**) P.S.  In  some common but special cases, like **few objects (maybe less than K value)**, my model will **adaptively adjust the K value** (make K equal to `number of object - 1` ). If K is set to a infinite number, like 999, the KNN graph will be converted into a  full connected graph.

![evolutionK](./.assert/evolutionK.bmp)

|            Different K            |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :-------------------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
|         k=2(Vanilla one*)         |   35.14   | 50.43 |   24.70   |   37.29   |   30.74   |   47.37   | 56.89 | 81.94 |
|                k=3                |   36.12   | 50.34 |   26.11   |   39.17   |   32.32   |   49.69   | 57.39 | 81.94 |
|                k=4                |   34.21   | 50.34 |   23.42   |   36.54   |   31.12   |   46.44   | 57.23 | 82.01 |
|                k=5                |   38.67   | 50.28 |   29.94   |   42.60   |   35.13   |   54.08   | 57.95 | 81.95 |
|                k=6                |   38.06   | 50.48 |   28.91   |   40.99   |   33.81   |   52.04   | 57.73 | 82.00 |
|                k=7                |   36.70   | 50.38 |   26.92   |   39.42   |   32.50   |   50.07   | 57.47 | 81.96 |
|                k=8                |   36.74   | 50.35 |   26.98   |   39.81   |   32.83   |   50.57   | 57.99 | 81.98 |
|             **k=12**              | **38.73** | 50.67 | **30.16** | **42.53** | **35.04** | **54.10** | 57.67 | 82.05 |
|               k=16                |   37.26   | 50.65 |   27.58   |   40.72   |   33.59   |   51.68   | 57.93 | 82.04 |
|               k=32                |   38.42   | 50.52 |   29.39   |   42.04   |   34.68   |   53.37   | 57.82 | 81.99 |
|               k=64                |   38.82   | 50.49 |   30.01   |   41.95   |   34.61   |   53.24   | 57.67 | 82.02 |
| k=999(INF) (Full Connected Graph) |   37.10   | 50.33 |   27.53   |   40.27   |   33.22   |   51.14   | 57.27 | 82.00 |
|        k=12 [`Undirected`]        |   37.94   | 50.53 |   28.64   |   41.38   |   34.13   |   52.54   | 57.60 | 81.97 |

|      Different K<br>+[Mask-50]       |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :----------------------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
|                 k=2                  |   43.82   | 50.71 |   38.08   |   50.08   |   41.35   |   63.47   | 59.09 | 81.97 |
|                 k=3                  |   43.67   | 50.69 |   37.89   |   50.20   |   41.43   |   63.66   | 59.11 | 82.11 |
|               **k=4**                | **44.27** | 50.59 | **38.93** | **50.73** | **41.87** | **64.33** | 59.18 | 81.99 |
|               **k=5**                |   44.19   | 50.58 |   38.00   |   50.44   |   41.65   |   63.96   | 59.15 | 82.01 |
|                 k=6                  |   43.70   | 50.46 |   38.03   |   50.06   |   41.32   |   63.47   | 58.94 | 82.00 |
|                 k=7                  |   44.08   | 50.65 |   38.55   |   50.10   |   41.36   |   63.52   | 58.91 | 81.99 |
|                 k=8                  |   42.08   | 50.43 |   35.28   |   47.74   |   39.40   |   60.58   | 58.67 | 82.00 |
|                 k=12                 |   41.85   | 50.61 |   34.80   |   47.45   |   39.16   |   60.20   | 58.52 | 82.00 |
|                 k=16                 |   42.64   | 50.67 |   36.07   |   48.26   |   39.83   |   61.21   | 58.49 | 82.00 |
|                 k=32                 |   42.77   | 50.43 |   36.44   |   48.31   |   39.88   |   64.27   | 58.68 | 82.00 |
|                 k=64                 |   42.93   | 50.32 |   36.81   |   48.49   |   40.02   |   61.50   | 58.44 | 81.99 |
| k=999(INF)<br>(Full Connected Graph) |   42.11   | 50.52 |   35.28   |   47.68   |   39.34   |   60.50   | 58.46 | 81.96 |
|       k=12<br/>[`Undirected`]        |   41.17   | 50.63 |   33.67   |   46.49   |   38.39   |   58.92   | 58.55 | 81.98 |

And here is a line chart which shows the trend of performance with increasing K : 

![Kfamily_metrics](./.assert/Kfamily_metrics.png)

----

In my opinion, the value of K plays a crucial role in enhancing the robustness of my model, particularly in scenarios where missed detections or false positives lead to a big change in graph structure.

And here are some statistical information about people count per frame in several benchmark:

| MOT17-train |   mean    |  max   |  min  |
| :---------: | :-------: | :----: | :---: |
|  MOT17-02   |   42.3    |   48   |  32   |
|  MOT17-04   | **49.86** | **56** |  45   |
|  MOT17-05   |   8.64    |   15   |   4   |
|  MOT17-09   |   11.12   |   14   |   6   |
|  MOT17-10   |   21.74   |   29   |  15   |
|  MOT17-11   |   10.58   |   19   |   7   |
|  MOT17-13   |   15.52   |   32   | **2** |
|             |   22.83   |   56   |   2   |

![MOT17_train_seq_stats](./.assert/MOT17_train_seq_stats.png)

| MOT20-train |   mean    |   max   |  min   |
| :---------: | :-------: | :-----: | :----: |
|  MOT20-01   |   52.3    |   61    |   42   |
|  MOT20-02   |   59.6    |   101   | **33** |
|  MOT20-03   |   139.7   |   205   |   84   |
|  MOT20-05   | **222.4** | **248** |  190   |
|             |   118.5   |   248   |   33   |

![MOT20_train_seq_stats](./.assert/MOT20_train_seq_stats.png)

| DanceTrack-train |   mean   |  max   |  min  |
| :--------------: | :------: | :----: | :---: |
|  dancetrack0001  |   7.0    |   7    |   6   |
|  dancetrack0002  |   7.7    |   8    |   5   |
|  dancetrack0006  |   8.8    |   9    |   4   |
|  dancetrack0008  |   7.8    |   8    |   4   |
|  dancetrack0012  |   10.8   |   12   |   4   |
|  dancetrack0015  |   8.9    |   9    |   7   |
|  dancetrack0016  |   5.8    |   6    |   4   |
|  dancetrack0020  | **34.6** | **40** |  27   |
|  dancetrack0023  |   8.8    |   9    |   7   |
|  dancetrack0024  |   5.9    |   6    |   3   |
|  dancetrack0027  |   8.5    |   12   |   5   |
|  dancetrack0029  |   6.2    |   7    |   5   |
|  dancetrack0032  |   5.6    |   6    |   2   |
|  dancetrack0033  |   7.8    |   8    |   5   |
|  dancetrack0037  |   6.9    |   7    |   5   |
|  dancetrack0039  |   4.9    |   5    |   1   |
|  dancetrack0044  |   11.1   |   13   |   7   |
|  dancetrack0045  |   13.2   |   14   |  10   |
|  dancetrack0049  |   7.9    |   8    |   6   |
|  dancetrack0051  |   9.0    |   9    |   8   |
|  dancetrack0052  |   3.7    |   4    |   2   |
|  dancetrack0053  |   4.9    |   5    |   3   |
|  dancetrack0055  |   4.8    |   5    |   3   |
|  dancetrack0057  |   4.8    |   6    |   3   |
|  dancetrack0061  |   5.0    |   5    |   5   |
|  dancetrack0062  |   5.2    |   6    |   5   |
|  dancetrack0066  |   5.0    |   5    |   5   |
|  dancetrack0068  |   5.0    |   5    |   3   |
|  dancetrack0069  |   5.9    |   6    |   4   |
|  dancetrack0072  |   4.9    |   5    |   3   |
|  dancetrack0074  |   4.4    |   5    | **1** |
|  dancetrack0075  |   6.8    |   7    |   5   |
|  dancetrack0080  |   10.0   |   16   |   6   |
|  dancetrack0082  |   20.5   |   24   |  12   |
|  dancetrack0083  |   24.9   |   25   |  22   |
|  dancetrack0086  |   15.7   |   16   |  13   |
|  dancetrack0087  |   10.7   |   11   |   9   |
|  dancetrack0096  |   26.2   |   40   |  17   |
|  dancetrack0098  |   7.0    |   8    |   4   |
|  dancetrack0099  |   10.3   |   11   |   5   |
|                  |   9.3    |   40   |   1   |

![DanceTrack-train](./.assert/DanceTrack-train.png)

| DanceTrack-val |   mean   |  max   |  min  |
| :------------: | :------: | :----: | :---: |
| dancetrack0004 |   3.8    |   4    | **1** |
| dancetrack0005 |   3.9    |   4    |   3   |
| dancetrack0007 |   7.9    |   8    |   6   |
| dancetrack0010 |   5.8    |   6    |   4   |
| dancetrack0014 |   10.5   |   12   |   3   |
| dancetrack0018 |   5.8    |   8    |   5   |
| dancetrack0019 |   6.9    |   7    |   5   |
| dancetrack0025 |   8.8    |   9    |   7   |
| dancetrack0026 |   13.2   |   18   |   7   |
| dancetrack0030 |   6.0    |   6    |   5   |
| dancetrack0034 |   12.1   |   14   |   7   |
| dancetrack0035 |   7.5    |   8    |   5   |
| dancetrack0041 |   16.7   |   22   |   8   |
| dancetrack0043 |   9.2    |   12   |   7   |
| dancetrack0047 |   7.8    |   8    |   5   |
| dancetrack0058 |   6.9    |   7    |   6   |
| dancetrack0063 |   7.6    |   8    |   2   |
| dancetrack0065 |   5.0    |   5    |   4   |
| dancetrack0073 |   12.0   |   14   |   8   |
| dancetrack0077 |   9.0    |   9    |   9   |
| dancetrack0079 |   12.1   |   18   |   7   |
| dancetrack0081 |   19.1   |   20   |  15   |
| dancetrack0090 |   13.1   |   14   |   9   |
| dancetrack0094 | **19.5** | **23** |  14   |
| dancetrack0097 |   3.9    |   4    |   2   |
|                |   9.4    |   23   |   1   |

![DanceTrack-val](./.assert/DanceTrack-val.png)

### 4.7 After more Edge Embeddings

The reason why I wanna change the weight of edge is **the week connection between similar objects or closely positioned objects.** In other words, for similar objects in close positions, **the current model has week differentiation capability (i.e. insufficient feature discriminability).** More details in the following picture.

![whyEdge](./.assert/whyEdge.bmp)

How to alleviate or even solve this problem? Some trial methods are waiting for me to practice.

#### 4.7.1 several Variants of Edge Embedding [:tada:]

Curious about the influences of edge embedding , I design several variants of Edge embedding  and do some experiments. And here are some mathematical formulation of my ideas. [The edge embedding of `Vanilla model`  mainly refers [SUSHI(CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Cetintas_Unifying_Short_and_Long-Term_Tracking_With_Graph_Hierarchies_CVPR_2023_paper.pdf) , which is a offline and graph-based tracker. And it\` another motivation to encourage me to solve the puzzle ]

1. **4-dim** Edge embedding (normalized by the **length and width of the original image**):
   $$
   Edge~emb := f([\frac{x_j - x_i}{w},\frac{y_j-y_i}{h},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i})])
   $$

2. **4-dim** Edge embedding (normalized by the **length and width of the bounding box of source nodes** ,i.e. neighbor nodes):
   $$
   Edge~emb := f([\frac{x_j - x_i}{w_j},\frac{y_j-y_i}{h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i})])
   $$

3. **4-dim** Edge embedding (normalized by the **length and width of the bounding box of target nodes**):
   $$
   Edge~emb := f([\frac{x_j - x_i}{w_i},\frac{y_j-y_i}{h_i},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i})])
   $$

4. **4-dim** Edge embedding (normalized by the **mean width and height of nodes**):
   $$
   Edge~emb := f([\frac{2(x_j - x_i)}{w_i+w_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i})])
   $$

5. **4-dim** Edge embedding (normalized by **length and width of smallest enclosing bbox**):
   $$
   Edge~emb := f([\frac{x_j - x_i}{w_{convex}},\frac{y_j-y_i}{h_{convex}},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i})])
   $$

6. **4-dim** Edge embedding(normalized by **maximum shape of nodes**):
   $$
   Edge~emb := f([\frac{x_j - x_i}{\max{(w_i,w_j)}},\frac{y_j-y_i}{\max{(h_i,h_j)}},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i})])
   $$

7. **4-dim** Edge embedding (normalized by the **mean height of nodes**):
   $$
   Edge~emb := f([\frac{2(x_j - x_i)}{h_i+h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i})])
   $$

8. **4-dim** Edge embedding (normalized by the **mean width of nodes**):
   $$
   Edge~emb := f([\frac{2(x_j - x_i)}{w_i+w_j},\frac{2(y_j-y_i)}{w_i+w_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i})])
   $$

9. **5-dim** Edge embedding (add **IOU distance** as another dimension to supply more information):
   $$
   Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),1-IoU(i,j)])
   $$

10. **5-dim** Edge embedding (add  **IOU ** rather than **IOU distance** ):
$$
Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),IoU(i,j)])
$$

11. **5-dim** Edge embedding (add  **GIOU distance ** as another dimension to supply more information):
    $$
    Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),1-GIoU(i,j)])
    $$

12. **5-dim** Edge embedding (add  **GIOU ** rather than **GIOU distance** ):
    $$
    Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),GIoU(i,j)])
    $$

13. **5-dim** Edge embedding (add **DIOU distance** as another dimension to supply more information): [Actually, the same as `Vanilla model`]
    $$
    Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),1-DIoU(i,j)])
    $$

14. **5-dim** Edge embedding (add **DIOU** rather than **DIOU distance**):
    $$
    Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),DIoU(i,j)])
    $$

15. **5-dim** Edge embedding (add **CIOU distance** as another dimension to supply more information): 
    $$
    Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),1-CIoU(i,j)])
    $$

16. **5-dim** Edge embedding (add **CIOU** rather than CIOU distance):
    $$
    Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),CIoU(i,j)])
    $$

17. **5-dim** Edge embedding (add **EIOU distance** as another dimension to supply more information): 
    $$
    Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),1-EIoU(i,j)])
    $$

18. **5-dim** Edge embedding (add **EIOU** rather than EIOU distance):
    $$
    Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),EIoU(i,j)])
    $$

19. **6-dim** Edge embedding (add **Cosine distance** as another dimension to supply more information):
    $$
    Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),1-DIoU(i,j),cosineDist(i,j))])
    $$

20. **8-dim** Edge embedding (Motivated by **IOU Family -- GIOU , DIOU , CIOU , EIOU, SIOU**):
    $$
    Edge~emb := &f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},1-GIoU(i,j),centerPointDist(i,j),\\&log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),\frac{(w_j-w_i)^2}{w_{convex}^2},\frac{(h_j-h_i)^2}{h_{convex}^2})])
    $$

    * The reason why I calculate the edge embedding is that **this 4 original items reveals more information about directions** rather than more precise quantitative information. After making a simple research and code implementation on IOU Family, I design this 8-dim edge embedding. And the only thing puzzles me that relativity of each item — **the 4 original items with some directional info is relative by mean height of paired bounding box ,while the other is relative by smallest enclosing bounding box ( the same as EIOU )**. And waiting to see :eyes:

21. **8-dim** Edge embedding (The variation of No. 16: Since the calculation of GIoU takes the smallest enclosing bounding box into account, I have standardized **the relativity of all elements based on this smallest enclosing bounding box**)
    $$
    Edge~emb := &f([\frac{x_j - x_i}{w_{convex}},\frac{y_j-y_i}{h_{convex}},1-GIoU(i,j),centerPointDist(i,j),\\&log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),\frac{(w_j-w_i)^2}{w_{convex}^2},\frac{(h_j-h_i)^2}{h_{convex}^2})])
    $$
    
22. **8-dim** Edge embedding (The variation of No. 17: While the first four items need to have their relativity standardized based on the smallest enclosing bounding box, the other four items should instead have their relativity **standardized based on the maximum of the nodes** rather than the smallest enclosing bounding box):
    $$
    Edge~emb := &f([\frac{x_j - x_i}{w_{convex}},\frac{y_j-y_i}{h_{convex}},1- GIoU(i,j),centerPointDist(i,j),\\&log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),\frac{(w_j-w_i)^2}{\max^2{(w_i,w_j)}},\frac{(h_j-h_i)^2}{\max ^2{(h_i,h_j)}})])
    $$

In order to better  organize and manage these experiments, it is necessary to rename these experiments:

| Experimental Index |  Experimental Name  |
| :----------------: | :-----------------: |
|         1          |      ImgNorm4       |
|         2          |      SrcNorm4       |
|         3          |      TgtNorm4       |
|         4          |    MeanSizeNorm4    |
|         5          |   MeanHeightNorm4   |
|         6          |   MeanWidthNorm4    |
|         7          |     ConvexNorm4     |
|         8          |      MaxNorm4       |
|         9          |        IOUd5        |
|         10         |        IOU5         |
|         11         |       GIOUd5        |
|         12         |        GIOU5        |
|         13         |       DIOUd5        |
|         14         |        DIOU5        |
|         15         |       CIOUd5        |
|         16         |        CIOU5        |
|         17         |       EIOUd5        |
|         18         |        EIOU5        |
|         19         |     DIOUd-Cos6      |
|         20         | IouFamily8-vanilla  |
|         21         |  IouFamily8-convex  |
|         22         | IouFamily8-separate |

And here are the summary results.

|          Experiment<br>+[Mask-50]           | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-----------------------------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|                  ImgNorm4                   | 44.52 | 50.46 | 39.48 | 50.73 | 41.88 | 64.32 | 58.60 | 82.00 |
| <strong style='color:red'>SrcNorm4</strong> | 45.31 | 50.51 | 40.84 | 52.36 | 43.23 | 66.38 | 59.19 | 81.97 |
| <strong style='color:red'>TgtNorm4</strong> | 45.70 | 50.53 | 41.51 | 53.10 | 43.85 | 67.30 | 59.27 | 81.99 |
|                MeanSizeNorm4                | 44.93 | 50.53 | 40.15 | 51.97 | 42.90 | 65.89 | 59.32 | 82.01 |
|               MeanHeightNorm4​​               | 45.13 | 50.39 | 40.57 | 52.67 | 43.48 | 66.79 | 59.13 | 82.02 |
|               MeanWidthNorm4                | 45.05 | 50.69 | 40.24 | 52.16 | 43.06 | 66.14 | 59.32 | 82.03 |
|                 ConvexNorm4                 | 45.11 | 50.58 | 40.43 | 51.70 | 42.69 | 65.52 | 59.16 | 82.01 |
| <strong style='color:red'>MaxNorm4</strong> | 45.45 | 50.21 | 41.08 | 52.63 | 43.45 | 66.74 | 59.35 | 82.01 |
|                    IOUd5                    | 44.56 | 50.70 | 39.36 | 51.33 | 42.39 | 65.06 | 59.23 | 81.98 |
|                    IOU5                     | 44.64 | 50.58 | 39.59 | 51.58 | 42.58 | 65.39 | 59.13 | 81.96 |
| <strong style='color:green'>GIOUd5</strong> | 45.51 | 50.52 | 41.18 | 52.54 | 43.39 | 66.60 | 59.20 | 82.00 |
|                    GIOU5                    | 44.79 | 50.73 | 39.74 | 50.58 | 41.77 | 64.12 | 59.32 | 82.01 |
|        DIOUd5 <br> (`Vanilla model`)        | 44.40 | 50.66 | 39.11 | 50.57 | 41.75 | 64.12 | 59.17 | 81.95 |
|                    DIOU5                    | 44.98 | 50.60 | 40.17 | 51.96 | 42.90 | 65.88 | 59.17 | 81.99 |
| <strong style='color:green'>CIOUd5</strong> | 45.43 | 50.75 | 40.86 | 52.75 | 43.55 | 66.89 | 59.37 | 81.96 |
|                    CIOU5                    | 45.30 | 50.60 | 40.75 | 51.94 | 42.89 | 65.83 | 59.18 | 81.97 |
|                   EIOUd5​​                    | 44.94 | 50.64 | 40.08 | 51.35 | 42.41 | 65.09 | 59.33 | 81.98 |
|                    EIOU5                    | 43.47 | 50.73 | 37.44 | 49.81 | 41.14 | 63.13 | 59.24 | 81.98 |
|                 DIOUd-Cos6                  |       |       |       |       |       |       |       |       |
|             IouFamily8-vanilla              |       |       |       |       |       |       |       |       |
|              IouFamily8-convex              |       |       |       |       |       |       |       |       |
|             IouFamily8-separate             |       |       |       |       |       |       |       |       |

<a id='weirdPhenomenon'></a>The results are quite beyond my expectation. And In my intuition, the more information edge embedding contains , the better model will be. However, the fact goes away my intuition. But WHAT can I learn from all those results??  Noticed that some directional information maybe impacts more on model, so I plan do some extra experiments —— the sequence of each item in IoUFamily8 and another type of directional information of width and height.

1. **IouFamily8-vanilla-seq**：
   $$
   Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),\\1-GIoU(i,j),centerPointDist(i,j),\frac{(w_j-w_i)^2}{w_{convex}^2},\frac{(h_j-h_i)^2}{h_{convex}^2})])
   $$

2. **IouFamily8-convex-seq**：
   $$
   Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),\\1-GIoU(i,j),centerPointDist(i,j),\frac{(w_j-w_i)^2}{w_{convex}^2},\frac{(h_j-h_i)^2}{h_{convex}^2})])
   $$

3. **IouFamily8-separate-seq**：
   $$
   Edge~emb := f([\frac{x_j - x_i}{w_{convex}},\frac{y_j-y_i}{h_{convex}},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),\\1- GIoU(i,j),centerPointDist(i,j),\frac{(w_j-w_i)^2}{\max^2{(w_i,w_j)}},\frac{(h_j-h_i)^2}{\max ^2{(h_i,h_j)}})])
   $$

4. **ConvexNorm4-v2**：
   $$
   Edge~emb := f([\frac{x_j - x_i}{w_{convex}},\frac{y_j-y_i}{h_{convex}},\frac{w_j - w_i}{w_{convex}},\frac{h_j-h_i}{h_{convex}}])
   $$

5. **MaxNorm4-v2**：
   $$
   Edge~emb := f([\frac{x_j - x_i}{\max{(w_i,w_j)}},\frac{y_j-y_i}{\max{(h_i,h_j)}},\frac{w_j - w_i}{\max{(w_i,w_j)}},\frac{h_j-h_i}{\max{(h_i,h_j)}}])
   $$

6. **GIOUd5-v2**：
   $$
   Edge~emb := f([\frac{x_j - x_i}{\max{(w_i,w_j)}},\frac{y_j-y_i}{\max{(h_i,h_j)}},\\\frac{w_j - w_i}{\max{(w_i,w_j)}},\frac{h_j-h_i}{\max{(h_i,h_j)}},1-GIoU(i,j)])
   $$

7. **GIOU5-v2**：
   $$
   Edge~emb := f([\frac{x_j - x_i}{\max{(w_i,w_j)}},\frac{y_j-y_i}{\max{(h_i,h_j)}}\\ \frac{w_j - w_i}{\max{(w_i,w_j)}},\frac{h_j-h_i}{\max{(h_i,h_j)}},GIoU(i,j)])
   $$

8. **DIOU5-v2**：
   $$
   Edge~emb := f([\frac{x_j - x_i}{\max{(w_i,w_j)}},\frac{y_j-y_i}{\max{(h_i,h_j)}},\\ \frac{w_j - w_i}{\max{(w_i,w_j)}},\frac{h_j-h_i}{\max{(h_i,h_j)}},DIoU(i,j)])
   $$

9. **CIOU5-v2**：
   $$
   Edge~emb := f([\frac{x_j - x_i}{\max{(w_i,w_j)}},\frac{y_j-y_i}{\max{(h_i,h_j)}},\\ \frac{w_j - w_i}{\max{(w_i,w_j)}},\frac{h_j-h_i}{\max{(h_i,h_j)}},CIoU(i,j)])
   $$

10. **IouFamily6-convex** :
    $$
    Edge~emb := f([\frac{x_j - x_i}{w_{convex}},\frac{y_j-y_i}{h_{convex}},\frac{w_j - w_i}{w_{convex}},\frac{h_j-h_i}{h_{convex}},\\1- GIoU(i,j),centerPointDist(i,j)])
    $$
    

|       Experiment        | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :---------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ----- |
| IouFamily8-vanilla-seq  |       |       |       |       |       |       |       |       |
|  IouFamily8-convex-seq  | 28.79 | 51.00 | 16.47 | 27.70 | 23.15 | 34.47 | 52.71 | 81.89 |
| IouFamily8-separate-seq |       |       |       |       |       |       |       |       |
|     ConvexNorm4-v2      | 44.33 | 50.64 | 39.00 | 51.00 | 42.11 | 64.65 | 59.24 | 81.97 |
|       MaxNorm4-v2       | 44.58 | 50.62 | 39.47 | 51.08 | 42.17 | 64.75 | 59.23 | 82.00 |
|  GIOUd5-v2:red_circle:  |       |       |       |       |       |       |       |       |
|  GIOU5-v2:red_circle:   |       |       |       |       |       |       |       |       |
|  DIOU5-v2:red_circle:   |       |       |       |       |       |       |       |       |
|  CIOU5-v2:red_circle:   |       |       |       |       |       |       |       |       |
|    IouFamily6-convex​​    |       |       |       |       |       |       |       |       |

#### 4.7.2 Attention Mechanism [:eyes:]

<a id='Attention'></a>

### 4.8 After more Graph Neural Network [:tada:]

Motivated by the influence of self loop, I wanna explore more about variants graph convolution operations. Although all of these variants are based on `MessagePassing` paradigm, there is no any paper about graph-based trackers to figure out the best model in MOT. And I will attempt to change the structure of `vanilla model`  and do experiments to find the best, with my limited knowledge of GNN and MOT. Here are several variants of `vanilla model` that I wanna try:

1. Deepen message function of my graph neural network in `Vanilla model`.(marked as `deepMsg`)
   $$
   ^{s}g^{l+1}_i=& \Phi\{\max _{j \in \mathbb{N}_i}{f([~Edge~emb ~\cdot~(^{s}g_j^{l} - ^{s}g_i^{l})~ ]\}})\\
   ^{d}g^{l+1}_i=& \max _{j \in \mathbb{N}_i}{f([^{d}g_i^{l}~\cdot~(^{d}g_j^{l} - ^{d}g_i^{l})~]})\\
   $$

   - The reason why I wanna deepen my message function of vanilla model is that there is only one full connected layer $f(\cdot )$ to fuse the feature $Edge~emb$ and $g_j -g_i$, which has not enough ability to align the feature space, thus harmful to my model. So I wanna deepen my message function.:pray:

   ![deepMsg](./.assert/deepMsg.bmp)

2. Change aggregation `max` to `mean`: (marked as `meanAggr`)
   $$
   ^{s}g^{l+1}_i=&  \Phi (\frac{1}{N_i}\sum _{j \in \mathbb{N}_i}{f([Edge~emb\cdot~(^s g_j^{l} - ^s g_i^{l}) ])})\\
   ^{d}g^{l+1}_i=& \frac{1}{N_i}\sum _{j \in \mathbb{N}_i}{f([^d g_i^{l}~\cdot~(^d g_j^{l} - ^d g_i^{l})])}
   $$
   ![meanAggr](./.assert/meanAggr.bmp)

3. Reimplement graph convolutions: (which is similar to Graphconv  || marked as `Graphconv`)
   $$
   ^{s}g^{l+1}_i=& \Phi \{~f_1(^{s}g_i^l) +\max _{j \in \mathbb{N}_i}{f([Edge~emb\cdot~(^s g_j^{l} - ^s g_i^{l}) ])}~\}\\
   ^{d}g^{l+1}_i=& \max _{j \in \mathbb{N}_i}{f([^d g_i^{l}~\cdot~(^d g_j^{l} - ^d g_i^{l})])}
   $$

   - Let\`s consider an extremely special case that there is only one object detected. In such cases, the graph contains only one nodes and only one edge , i.e. its self loop. The node features extracted by `NodeEncoder` are normal, while edge features generated by `EdgeEncoder` become a extremely sparse tensor. After my model calculating, the output node features also become a extremely sparse tensor, whose essential features, like appearance features, are lost in the calculation of static graph convolution network. So inspired by the idea of `Graphconv` and residual connection , I propose this formulation which can keep some inherent features rather than differences. And hoping it can improve the robustness of my model :pray: 

4. Add edge embedding in dynamic graph: ( Because my model will construct the dynamic graph in higher feature space, it\`s of necessity to **recalculate the edge embedding** ,recorded as $Edge~emb^*$ || marked as `DoubleEdgeEmb`) 
   $$
   ^{s}g^{l+1}_i=& \Phi \{~f_1(^{s}g_i^l) +\max _{j \in \mathbb{N}_i}{f([Edge~emb~\cdot ~(^s g_j^{l} - ^s g_i^{l}) ])}~\}\\
   ^{d}g^{l+1}_i=& \max _{j \in \mathbb{N}_i}{f([^{d}g_i^l~ \cdot ~Edge~emb^* ~\cdot ~(^d g_j^{l} - ^d g_i^{l})])}
   $$

5. Extra concatenate of node\`s own features: (marked as `selfConcat`)]
   $$
   ^{s}g^{l+1}_i=& \Phi \{~\max _{j \in \mathbb{N}_i}{f([^s g_i^{l}~\cdot~Edge~emb~\cdot ~(^s g_j^{l} - ^s g_i^{l}) ]~\}} \\
   ^{d}g^{l+1}_i=& \max _{j \in \mathbb{N}_i}{f([^d g_i^{l}~\cdot~Edge~emb^* \cdot ~(^d g_j^{l} - ^d g_i^{l})~ ])}
   $$

|          Model Variants          | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :------------------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|     Vanilla one<sup>*</sup>      | 35.14 | 50.43 | 24.70 | 37.29 | 30.74 | 47.37 | 56.89 | 81.94 |
|             deepMsg              | 33.54 | 50.37 | 22.54 | 34.25 | 28.24 | 43.51 | 56.27 | 81.98 |
|             meanAggr             | 26.69 | 51.01 | 14.17 | 25.15 | 21.01 | 31.31 | 51.81 | 81.87 |
|   Graphconv <br>*w/* self-loop   |       |       |       |       |       |       |       |       |
|  Graphconv <br/>*w/o* self-loop  |       |       |       |       |       |       |       |       |
|  DouleEdgeEmb<br>*w/* self-loop  | 30.88 | 51.38 | 18.77 | 29.97 | 25.06 | 37.27 | 56.10 | 81.86 |
| DouleEdgeEmb<br/>*w/o* self-loop | 37.43 | 50.50 | 27.94 | 40.16 | 33.12 | 51.00 | 58.11 | 81.99 |
|            selfConcat            | 30.31 | 51.24 | 18.13 | 30.29 | 25.33 | 37.68 | 56.09 | 81.88 |

Noted that the method of adding up to enhance one\`s own features is not quite effective, i.e. `GraphConv` and `DoubleEdgeEmb`. Instead, the method of concatenation seems to perform better, i.e. `selfConcat`. So let me dig further out the potentiality of concatenation.

1. Just concatenate of node\`s own feature in the static graph convolution network: (marked as `StatiSelfConcat`)
   $$
   ^{s}g^{l+1}_i=& \Phi \{~\max _{j \in \mathbb{N}_i}{f([^s g_i^{l}~\cdot~Edge~emb~\cdot ~(^s g_j^{l} - ^s g_i^{l}) ]~\}} \\
   ^{d}g^{l+1}_i=& \max _{j \in \mathbb{N}_i}{f([^d g_i^{l}~\cdot(^d g_j^{l} - ^d g_i^{l})~ ])}
   $$

2. Swap two graph convolution methods: (marked as `SwapConv`)
   $$
   ^{s}g^{l+1}_i=& \Phi\{\max _{j \in \mathbb{N}_i}{f([^{s}g_i^{l}~\cdot~(^{s}g_j^{l} - ^{s}g_i^{l})~]\}})\\
   ^{d}g^{l+1}_i=& \max _{j \in \mathbb{N}_i}{f([~Edge~emb ~\cdot~(^{d}g_j^{l} - ^{d}g_i^{l})~ ]})\\
   $$

|            Model Variants            | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :----------------------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  StaticSelfConcat<br>*w/* self-loop  | 30.60 | 51.18 | 18.49 | 30.84 | 25.79 | 38.36 | 55.80 | 81.87 |
| StaticSelfConcat<br/>*w/o* self-loop |       |       |       |       |       |       |       |       |
|      SwapConv<br>*w/* self-loop      | 38.78 | 50.61 | 29.88 | 43.16 | 35.60 | 54.80 | 58.41 | 82.01 |
|     SwapConv<br/>*w/o* self-loop     |       |       |       |       |       |       |       |       |

----

Noticed that the big improvement by `deepMsg`, I think the misalignment of edge embedding feature space and node embedding feature space is quite harmful to my model. And this problem becomes more serious with increasing stacking GNN layers.  :confused: In my opinion, this problem also leads to the weird and counterintuitive, at least for me , phenomenon in [Sec 4.6.1 several Variants of Edge Embedding](#weirdPhenomenon). More specifically, the more information the original edge embedding contains , the more complicated the original edge embedding feature space is. And it is quite impossible for fewer fc layers to align the edge feature space and node feature space. And it is necessary to redo some experiments concerning variants of edge embedding by utilizing some fc layers to transform edge embedding into a suitable feature space. And there are two measures I wanna take:

1. **Attention Mechanism**(more details in  [Sec 4.6.2](#Attention))： Especially [CEN-DGCNN](https://www.nature.com/articles/s41598-023-44224-1)-like attention mechanism, which updates the edge embedding with increasing of GCNN.
2. **Simple Strategy**： I just add some extra fc layers to transform edge embedding into a suitable feature space at the beginning of each GNN.

Now let\`s focus on **Simple Strategy** in this Section. Considering the size of edge embedding, this can further be divided into three approaches: the first is to always maintain consistency in the feature dimensions, i.e. 16 dimensions , the second is to allow the dimension of edge embedding to increase progressively as GCNN layers are stacked, in parallel with the increasing dimensions of node features., while the last one is to utilize raw edge attribute , like the formulations in [Sec 4.6.1 ](#weirdPhenomenon), without any fc layers to  elevate dimension to 16-dim. Named separately as `FixedEdgeDim` , `ProgressiveEdgeDim` and `RawEdgeAttr` .

![simpleStrategy](./.assert/simpleStrategy.bmp)

P.S. the following experiments will not apply `deepMsg`, for fair comparison.

| Vanilla Model<br>[ConvexNorm4-v2] + | HOTA | DetA | AssA | IDF1 | IDR  | IDP  | MOTA | MOTP |
| :---------------------------------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|            FixedEdgeDim             |      |      |      |      |      |      |      |      |
|         ProgressiveEdgeDim          |      |      |      |      |      |      |      |      |
|             RawEdgeAttr             |      |      |      |      |      |      |      |      |

| Vanilla Model<br>[MaxNorm4-v2] + | HOTA | DetA | AssA | IDF1 | IDR  | IDP  | MOTA | MOTP |
| :------------------------------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|           FixedEdgeDim           |      |      |      |      |      |      |      |      |
|        ProgressiveEdgeDim        |      |      |      |      |      |      |      |      |
|           RawEdgeAttr            |      |      |      |      |      |      |      |      |

| Vanilla Model[DIoUd5] + | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :---------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|      FixedEdgeDim       |       |       |       |       |       |       |       |       |
|   ProgressiveEdgeDim    |       |       |       |       |       |       |       |       |
|       RawEdgeAttr       | 34.53 | 50.28 | 23.89 | 36.85 | 30.37 | 46.85 | 56.12 | 81.98 |

| Vanilla Model[GIoUd5] + | HOTA | DetA | AssA | IDF1 | IDR  | IDP  | MOTA | MOTP |
| :---------------------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|      FixedEdgeDim       |      |      |      |      |      |      |      |      |
|   ProgressiveEdgeDim    |      |      |      |      |      |      |      |      |
|       RawEdgeAttr       |      |      |      |      |      |      |      |      |

| Vanilla Model<br>[IoUFamily8-convex-seq] + | HOTA | DetA | AssA | IDF1 | IDR  | IDP  | MOTA | MOTP |
| :----------------------------------------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|                FixedEdgeDim                |      |      |      |      |      |      |      |      |
|             ProgressiveEdgeDim             |      |      |      |      |      |      |      |      |
|                RawEdgeAttr                 |      |      |      |      |      |      |      |      |

What surprises me is that there is a slight improvement in most experiments. After in-depth thinking, there is indeed an issue of feature space misalignment here, one space represents the  edge embedding space , while the other corresponds to the node embedding space. More specifically, One is position feature space,while the other is appearance feature space.

<a id='sec4.7'></a>And in the view of multi-modal fusion or multi-view fusion, what I should focus more on is the **fusion** rather than alignment. It also explains well why the model `deepMsg` performs better — due to the added a fully connected layer, which fuses the appearance features and position features better. 

### 4.9 After Bigger Model

Inspired by the conclusion and conjecture in [Sec 4.7](#sec4.7), I wanna conduct some experiments concerning the depth of model.

#### 4.9.1 About Message Layer

![deepmsg2](./.assert/deepmsg2.bmp)

|      Model Variant      | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :---------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla one<sup>*</sup> | 35.14 | 50.43 | 24.70 | 37.29 | 30.74 | 47.37 | 56.89 | 81.94 |
|        DeepMsg-1        | 33.54 | 50.37 | 22.54 | 34.25 | 28.24 | 43.51 | 56.27 | 81.98 |
|        DeepMsg-2        |       |       |       |       |       |       |       |       |

<img src="./.assert/smoothFamily.bmp" alt="smoothFamily" style="zoom:50%;" />

| Model Variant  | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Vanilla one*  | 35.14 | 50.43 | 24.70 | 37.29 | 30.74 | 47.37 | 56.89 | 81.94 |
| Static-smooth  | 34.45 | 50.44 | 23.74 | 35.81 | 29.54 | 45.46 | 56.48 | 81.98 |
| Dynamic-smooth | 34.69 | 50.43 | 24.06 | 36.28 | 29.90 | 46.10 | 56.52 | 81.99 |

#### 4.9.2 About Update Layer

![upd-2](./.assert/upd-2.bmp)

| Model Variant | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-----------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla one*  | 35.14 | 50.43 | 24.70 | 37.29 | 30.74 | 47.37 | 56.89 | 81.94 |
|   DeepUpd-2   |       |       |       |       |       |       |       |       |

#### 4.9.3 About Fuse Model

![deepFuse](./.assert/deepFuse.bmp)

| Model Variant | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-----------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla one*  | 35.14 | 50.43 | 24.70 | 37.29 | 30.74 | 47.37 | 56.89 | 81.94 |
|  DeepFuse-1   | 34.16 | 50.35 | 23.37 | 35.70 | 29.44 | 45.35 | 56.40 | 81.99 |
|  DeepFuse-2   | 33.94 | 50.24 | 23.13 | 35.45 | 29.21 | 45.08 | 56.28 | 81.97 |
|  DeepFuse-3   |       |       |       |       |       |       |       |       |

#### 4.9.4 Summary



| Model Variant | Mode Size (MB) | Params (M) | FLOPS (G) |
| :-----------: | :------------: | :--------: | :-------: |
| Vanilla one*  |     38.26      |    9.85    |   29.98   |
| Bigger Model  |                |            |           |

![flops-para(vanilla)](./.assert/flops-para(vanilla).bmp)

### 4.10 After Superior Appearance Feature Extractor [:eyes:]



### 4.11 After Larger Dataset [:tada:]

One of the core and useful experience or intuition in the era of deep learning is that `if you have a large big dataset and you train a very big neural network, then success is guaranted` (quoted from [llya`s speech at NeurlPS conference in 2024](https://www.bilibili.com/video/BV1cSBGYkE9w/?spm_id_from=333.337.search-card.all.click&vd_source=812705912b7abe259d54d8593a97a8b3)) , like CLIP model trained in 400 million dataset.

So I also attempt to add more benchmark, like MOT20 and DanceTrack, to enlarge the training dataset. Finally, the number of data is reached to **48856**, which is **20x times than before**. And the time expense is also huge , which maybe cost **3 days** to complete the whole training, **3x times than before.**

|       Conditions        | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :---------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla one<sup>*</sup> | 23.31 | 49.68 | 11.03 | 22.87 | 18.76 | 29.28 | 51.74 | 81.89 |
|     Larger Dataset      |       |       |       |       |       |       |       |       |

## 5. Experimental Records [Track Management]

In [[Sec 1. Brief Introduction about My Model]](#1. Brief Introduction about my model), I have a brief introduction about the trajectory management in the whole pipeline, but it\`s necessary to supplement more information in order to understand what I\`ll do in this section.

![statetransition](./.assert/statetransition.bmp)

To sum up, here is the state transition network of my track management. And there are four states of my single trajectory. And it\` extremely important to understand the whole switching process in my track management. 

The detailed significance of these four states of each trajectory:

1. `BORN state`\:  only those high-confidence detections (confidence >= 0.7) filtered by two-phase matching can be initialized as `BORN state` trajectory in the whole pipeline of track management. 
   - `BORN` ->`BORN`: Considering the case of detector false alarms, namely mistakenly identifying non-target objects as targets (FP, False Positive), there is possibility that `BORN` state is a kind of noise. Therefore, here sets the second matching phase, i.e. IOU Matching, to filter out the noise. And I set a variable `cnt_to_Active` ( default: 1 ) , which counts the continuous times of `BORN` state trajectories are successfully matched , to control the ability of filtration.
   - `BORN`->`ACTIVE`: I set a relatively strict rule that only those trajectories , whose state is switched from `BORN` state to `Active` for the first time , can be assigned with unique id.
   - `BRON`->`Dead`: the meaning of this switching process is that those `BORN`-state trajectories are noises, which should be filtered out.
2. `ACTIVE state`: the `Active`-state trajectories 
3. `SLEEP state`:
4. `DEAD state`: 

### 5.1 plz live longer [:eyes:]

### 5.2 plz look better [:eyes:] 

## 6. Experimental Records [Code Optimization]

What I always aspire for in my coding project is beautiful , efficient and simple programing logic. Needless to say that my whole project will run in an embeded system, which have lower computing capability and demands more efficient code. So I decide to record something about code optimization in the process of project optimization.

### 6.1 Remove Variable bt_self_loop

First of all, let me explain why I set this variable at the beginning of coding. It\`s possible that there is one object detected or tracked, where the number of objects is less than K value in KNN algorithm. And the graph-based project encounters fatal error, due to inability to construct a graph. Considering that the mathematical formulation of my graph convolution operations, I find that self loop of each node should have no impact on my model\`s performance, because subtracting oneself equals zero which is filter by the aggregation `max`. So why not add self loop for each node ，which can fix this fatal bug without affecting the performance of the model.  And here goes the original code about KNN algorithm, meanwhile is also the point waiting to optimization further.

```python
def knn(x: torch.tensor, k: int, bt_cosine: bool=False,
        bt_self_loop: bool=False,bt_directed: bool=True) -> torch.Tensor:
    """
    Calculate K nearest neighbors, supporting Euclidean distance and cosine distance.
    
    Args:
        x (Tensor): Input point set, shape of (n, d), each row represents a d-dimensional feature vector.
        k (int): Number of neighbors.
        bt_cosine (bool): Whether to use cosine distance.
        bt_self_loop (bool): Whether to include self-loop (i.e., whether to consider itself as its own neighbor).
        bt_directed (bool): return the directed graph or the undirected one. 

    Returns:
        edge_index (tensor): the edge index of the graph, shape of (2, n * k).
    """
    
    num_node = x.shape[0]

    if num_node <= k :
        # raise ValueError("The number of points is less than k, please set k smaller than the number of points.")
        logger.warning(f"SPECIAL SITUATIONS: The number of points is less than k, set k to {x.shape[0] -1}")
        k = num_node - 1
    
    if k > 0:
        if bt_cosine:   # cosine distance
            x_normalized = F.normalize(x, p=2, dim=1)
            cosine_similarity_matrix = torch.mm(x_normalized, x_normalized.T)
            dist_matrix  = 1 - cosine_similarity_matrix  
        else:           # Euclidean distance
            assert len(x.shape) == 2  
            dist_matrix = torch.cdist(x, x) 
            
        dist_matrix.fill_diagonal_(float('inf'))  
    
        _, indices1 = torch.topk(dist_matrix, k, largest=False, dim=1)
        indices2 = torch.arange(0, num_node, device=x.device).repeat_interleave(k)
    else:
        indices1 = torch.tensor([],device=x.device)
        indices2 = torch.tensor([],device=x.device)
    
    if bt_self_loop:
        indices_self = torch.arange(0,num_node,device=x.device)
        if bt_directed:
            return torch.stack([  # flow: from source node to target node 
                torch.cat([indices1.flatten(),indices_self],dim=-1),
                torch.cat([indices2,indices_self],dim=-1),
            ]).to(x.device).to(torch.long)
        else:
            return torch.stack([  # flow: from source node to target node 
                torch.cat([indices1.flatten(),indices_self,indices2],dim=-1),
                torch.cat([indices2,indices_self,indices1.flatten()],dim=-1),
            ]).to(x.device).to(torch.long)
    else:
        if bt_directed:
            return torch.stack([indices1.flatten(),indices2]).to(x.device).to(torch.long)  # flow: from source node to target node 
        else:
            return torch.stack([  # flow: from source node to target node 
                torch.cat([indices1.flatten(),indices2],dim=-1),
                torch.cat([indices2,indices1.flatten()],dim=-1),
            ]).to(x.device).to(torch.long)
```

If variable `bt_self_loop` set True, the above code will add self loop for every objects  without consideration about whether the only one object detected or tracked . So why not add self loop only when there is one object? **This method can simply further the graph structure and save more time expanse and space expanse.** :point_right: [Modified version of KNN](./models/graphtoolkit.py)​ :point_left:

----

Oops, I neglect the influence of self loop. And after removing loop of each nodes , and the experimental results plunge. More details in [Sec 4.6 After Weight of edge ](###4.6 After Weight of edge). But what benefits from misfortune allows me to understand more about my model.
