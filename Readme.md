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
g^{l+1}_i:= \Phi(\max _{j \in \mathbb{N}_i}{f([Edge~emb~\cdot~(g_j^{l} - g_i^{l}) ])})
$$

And in order to **capture high-order discriminative features**, I rebuild the graph, which I call the `dynamic graph`, while the old one is referred to as the `static graph` to keep things clear. And I also perform the edgeconv operations on dynamic graph:


$$
g^{l+1}_i:= \max _{j \in \mathbb{N}_i}{f([g_i^{l}~\cdot~(g_j^{l} - g_i^{l})])}
$$

----

The last but not least, it\`s necessary to take a glimpse of my **trajectory management strategy**. My trajectory management strategy mainly refers to **ByteTrack**. And here is my program flow chart:

<img src="./.assert/flowChart.png" alt="flowChart" style="zoom: 50%;" />

There are four states in the trajectory management strategy — **Born,Active,Sleep,Dead**. It\`s possible that `Born Trajectory` can be false positive, i.e. noise, so there are two phases in trajectory management — `Match Strategy` mainly for denoising and `Graph Matching` for matching. And the rest of the strategy is unnecessary to discuss, which is obvious in the flow chart.

----

Supplemental Details about my whole neural network, which is quite important to understand following experiment concerning [Sec 4.7 After more Graph Neural Network](###4.7 After more Graph Neural Network). 

There are 5 parts of my whole model:

1. **Node Encoder**: The backbone of Node Encoder is [DenseNet121](https://arxiv.org/abs/1608.06993), initialized with pretrained weights from ImageNet, the same as `GCNNMatch`. All layers are frozen except for the last two. Additionally, I also add some extra fc layers to reduce 1024 dimensional tensor to a 32 dimensional space for the sake of addressing feature space misalignment.

   <img src="./.assert/NodeEncoder.bmp" alt="NodeEncoder" style="zoom:50%;" />

2. **Edge Encoder**: 

   <img src="./.assert/EdgeEncoder.bmp" alt="EdgeEncoder" style="zoom:50%;" />

3. **Static Graph Model**: When stacking more layers in GCN may lead to over-smoothing and over-squeezing, which instead makes the performance of network drop sharply. And here I simply stack three layers GCN to extract features from static graph.

   

4. **Dynamic Graph Model**:

   

5. **Fuse Model**:

   

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
- And **random seed is set to 3407 ** || **k is set to 2 **

Here is the original quantitative results of my model without **any modification on model structure or the use of any data augmentation techniques**. 

| Validation set  | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MOT17-half(SDP) | 24.25 | 49.21 | 12.03 | 24.24 | 19.79 | 31.27 | 51.34 | 81.96 |

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

- [ ] change the weight of edge 

  I can encode the some info of graph structure into the embeddings(maybe cant be so sufficient)

- [ ] Design more suitable track management ,like the lifespan of active tracks

- [x] Simplify the logic of KNN — make KNN more adaptive . It seems that variable `bt_self_loop`  is relatively useless, because self-loop only matters when there is only one object, which will lead to fatal error in my project. And this maybe save more time expense. 

- [ ] do some statistical experiment about the density of the crowds in benchmarks, which is maybe helpful for the choice of best K

## 4. Experimental Records [Technique & Hyperparameters]

The quantitative results of the vanilla model  (the same results in [[Sec 1. Brief Introduction about My Model]](#1. Brief  Introduction about My Model), just repeat one more time and add some curves which can reflect something):

| Validation set  | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MOT17-half(SDP) | 24.25 | 49.21 | 12.03 | 24.24 | 19.79 | 31.27 | 51.34 | 81.96 |

<img src="./.assert/vanilaOne-index.bmp" alt="vanilaOne-index" style="zoom:25%;" />

Apart from the quantitative results of the vanilla one , let\`s focus on these two curves. The training data is 2650, which is relatively small and easily make models overfit. So I have to design more precise training strategy to avoid overfitting, where I adjust the warmup strategy and multiStep lr scheduler.

Noted that the f1 curve in evaluation phase is always above 0.9 , this f1 score is slightly different from one in MOT metric. Both of them is based on ID of tracked object, but the f1 score here is calculated in each timestamp — in other words, **the default prerequisite here is that the tracking result of each timestamp is independent**, which is quite different from one in MOT metric. Besides, **what is extremely different from real testing is that there is no missed detections in the evaluation phases , when training models.**

As we all known, the MOT problem can be viewed as a problem of **maximizing a posteriori probability** —— the tracking result of each timestamp is quite dependent on the results of previous moment. It\`s reasonable to infer that the worse tracking results from previous moment, the worse tracking results from current moment. Actually, the performance of my model is indeed like this.

### 4.1 After Data Augmentation [🎉]

> In order to avoid overfitting, it\`s overwhelmingly necessary to find out the various and valid data augmentation techniques.

![dataAugmentation ](./.assert/dataAugmentation.bmp)

There are three data augmentation techniques — **Low framerate, missed detections and discontinuous trajectories. **All of them is vividly showed in the above picture. So let\`s see the quantitative results of vanilla model after training. Oops, I changes some experimental settings. In this experiment, the total epoch is set to 120  (it maybe takes 2 hours or so in GTX3090 ), warmup iteration is set to 800 and multistep is set to 50 and 80.(Waiting to see :eyes:)

|    Conditions     |   HOTA    |   DetA    |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :---------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
|    Vanilla one    | **24.25** | **49.21** | **12.03** | **24.24** | **19.79** | **31.27** | 51.34 | 81.96 |
| Data Augmentation |   23.31   |   49.68   |   11.03   |   22.87   |   18.76   |   29.28   | 51.74 | 81.89 |

<img src="./.assert/dataAug-index.bmp" alt="dataAug" style="zoom:25%;" />

:loudspeaker:  Due to special designed data augmentation techniques , like missed detections simulations, low framerate, which is beneficial for network training,  I  <strong style="color: red;">use this as the main comparison which marks as Vanilla one<sup>*</sup>.</strong> 

### 4.2 After Undirected Graph [:tada:]

> **Graph matters in whole pipeline, which impacts on the ability of feature extraction.** Experiments in directed and undirected graphs , graphs with and without self-loops, and graph with varying K values are awaiting me.

Here is the simple illustration about the undirected graph:

![undirectedgraph](./.assert/undirectedgraph.bmp)

|       Conditions        |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :---------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
| Vanilla one<sup>*</sup> |   23.31   | 49.68 |   11.03   |   22.87   |   18.76   |   29.28   | 51.74 | 81.89 |
|    Undirected Graph     | **30.91** | 50.32 | **19.12** | **33.24** | **27.40** | **42.23** | 56.61 | 81.93 |

<img src="./.assert/UndirectedGraph-index.bmp" alt="UndirectedGraph-index" style="zoom: 25%;" />



### 4.3 After without Self-Loop [:tada:]

![](./.assert/self-loop.bmp)

Here is the simple example to show the graph with or without self loop. A special case is highlighted when only object is detected in an image (e.g., Fig3.). In such cases , a self-loop is retained for the object. This ensures the graph neural network can process the date effectively, avoiding issues that arise from an empty graph structure.

|   Conditions    |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :-------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
|  Vanilla one*   |   23.31   | 49.68 |   11.03   |   22.87   |   18.76   |   29.28   | 51.74 | 81.89 |
| *w/o* self loop | **28.39** | 50.06 | **16.21** | **29.87** | **24.61** | **38.00** | 55.42 | 81.96 |



### 4.4 After Distance Mask [:tada:]

![distanceMask](./.assert/distanceMask.bmp)

What\` s distance mask here ? Actually, it is a kind of precondition which is that the movement distance of objects in adjacent frames is likely to be small, but **sometimes can be large when in low framerate video or objects moving fast**. So it\`s necessary to statistically calculate **the max movement distance** of different situations, like **different framerate, fast-moving objects or video from a moving perspective.**  And here is the statistical results in **MOT17(Train set), MOT20(Train set), DanceTrack(Train && val set)**.

![DifferentWindow](./.assert/DifferentWindow.bmp)

Noted that it seems that the **max movement distance of adjacent frames (window - 2) is more reliable**. Let\`s dive into the situations of **windowSize[2]**.

<img src="./.assert/MOT17_20.bmp" alt="MOT17" style="zoom:50%;" />

<img src="./.assert/Dancetrack.bmp" alt="Dancetrack" style="zoom:50%;" />

It seems that **the speed of object moving poses the bigger influence on the statistical experiment among all factors.**:confused: Ummm, maybe **the distance between camera and objects also matters!**

| Mask Range |         HOTA          | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-------------------------------: | :-------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   None(Vanilla one<sup>*</sup>)   |   23.31   | 49.68 |   11.03   |   22.87   |   18.76   |   29.28   | 51.74 | 81.89 |
|                **35**             |         43.33         | 50.84 | 37.11 | 51.01 | 42.11 | 64.70 | 59.46 | 81.99 |
|                **50**              |         **45.19**         | 50.51 | **40.61** | **52.12** | **43.04** | **66.07** | 59.05 | 81.98 |
|                **60**             |    43.59    | 50.33 | 37.93 | 49.91 | 41.20 | 63.27 | 58.73 | 81.95 |
|                100                |    39.72    | 50.34 | 31.53 | 43.74 | 36.11 | 55.45 | 57.71 | 81.92 |
|                150                |    37.91    | 50.40 | 28.73 | 40.82 | 33.69 | 51.77 | 57.03 | 81.97 |
|                200                |         36.47         | 50.33 | 26.61 | 39.49 | 32.58 | 50.10 | 56.69 | 81.95 |
|                250                |         35.09         | 50.26 | 24.69 | 37.80 | 31.18 | 47.99 | 56.26 | 81.98 |
|                300                |    34.69    | 50.42 | 24.03 | 36.72 | 30.29 | 46.62 | 56.20 | 81.97 |
|                350                | 29.33 | 49.58 | 17.47 | 30.08 | 24.72 | 38.42 | 53.06 | 81.91 |
|                400                |   31.91   | 50.26 | 20.40 | 33.99 | 28.02 | 43.18 | 55.78 | 81.96 |
|                450                |    32.70    | 50.33 | 21.38 | 35.12 | 28.96 | 44.63 | 55.77 | 81.97 |
|                500                | 32.58 | 50.27 | 21.26 | 35.16 | 28.99 | 44.66 | 55.90 | 81.96 |

 In order to more intuition comparison, I plot a line chart based on the above table : [without **row None (Vanilla one<sup>*</sup>)** ]

![Kfamily_metrics](./.assert/mask.png)

### 4.5 After Different K [:tada:]

According to the descriptions about construction graph in [[Sec 1. Brief Introduction about My Model]](#1. Brief Introduction about My Model), there is a hyperparameter `k` in KNN algorithm. And k is set to 2 in my vanilla model, so let`s search for best k.

Here is a picture to visually illustrates the process of graph construction with the increase of K. ( Due to the shape of each figure, the relative positions of every nodes are distorted to some extent. And **the following graph construction is based on true coordinates in original image plane**) P.S.  In  some common but special cases, like **few objects (maybe less than K value)**, my model will **adaptively adjust the K value** (make K equal to `number of object - 1` ). If K is set to a infinite number, like 999, the KNN graph will be converted into a  full connected graph.

![evolutionK](./.assert/evolutionK.bmp)

|             Different K              |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :----------------------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
|     k=2(Vanilla one<sup>*</sup>)     |   23.31   | 49.68 |   11.03   |   22.87   |   18.76   |   29.28   | 51.74 | 81.89 |
|                 k=3                  |   24.58   | 49.92 |   12.20   |   25.82   |   21.23   |   32.94   | 52.99 | 81.98 |
|                 k=4                  |   27.89   | 50.02 |   15.66   |   29.33   |   24.13   |   37.38   | 54.52 | 81.98 |
|                 k=5                  |   29.58   | 50.14 |   17.56   |   30.84   |   25.40   |   39.26   | 55.35 | 81.95 |
|                 k=6                  |   30.40   | 50.20 |   18.54   |   32.79   |   27.01   |   41.70   | 55.47 | 81.98 |
|                 k=7                  |   32.60   | 50.25 |   21.27   |   34.86   |   28.75   |   44.26   | 56.53 | 81.99 |
|               **k=8**                |   35.09   | 50.05 |   24.75   |   39.01   |   32.14   |   49.61   | 56.89 | 82.00 |
|                 k=12                 |   33.85   | 50.18 |   22.96   |   37.12   |   30.60   |   47.19   | 56.72 | 82.00 |
|                 k=16                 |   32.37   | 50.40 |   20.90   |   35.49   |   29.24   |   45.12   | 57.11 | 82.03 |
|               **k=32**               | **36.11** | 50.24 | **26.09** | **40.01** | **32.97** | **50.88** | 56.66 | 82.02 |
|                 k=64                 |   33.65   | 50.30 |   22.65   |   36.37   |   29.96   |   46.26   | 56.46 | 82.04 |
| k=999(INF)<br>(Full Connected Graph) |   35.02   | 50.51 |   24.43   |   38.64   |   31.85   |   49.13   | 56.85 | 82.06 |

And here is a line chart which shows the trend of performance with increasing K : 

![Kfamily_metrics](./.assert/Kfamily_metrics.png)

----

In my opinion, the value of K plays a crucial role in enhancing the robustness of my model, particularly in scenarios where missed detections or false positives lead to a big change in graph structure.

### 4.6 After more Edge Embeddings

The reason why I wanna change the weight of edge is **the week connection between similar objects or closely positioned objects.** In other words, for similar objects in close positions, **the current model has week differentiation capability (i.e. insufficient feature discriminability).** More details in the following picture.

![whyEdge](./.assert/whyEdge.bmp)

How to alleviate or even solve this problem? Some trial methods are waiting for me to practice.

#### 4.6.1 several Variants of Edge Embedding [:tada:]

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

|          Experiment           |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :---------------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
|           ImgNorm4            |   25.45   | 49.98 |   13.07   |   26.09   |   21.46   |   33.25   | 53.03 | 81.99 |
|           SrcNorm4            |   27.37   | 50.13 |   15.07   |   28.49   |   23.47   |   36.24   | 54.16 | 81.95 |
|           TgtNorm4            |   27.39   | 50.09 |   15.10   |   28.37   |   23.37   |   36.09   | 54.81 | 81.95 |
|         MeanSizeNorm4         |   27.52   | 50.12 |   15.22   |   28.19   |   23.22   |   35.86   | 55.01 | 81.97 |
|        MeanHeightNorm4        |   27.07   | 50.22 |   14.71   |   27.67   |   22.80   |   35.19   | 54.79 | 81.97 |
|        MeanWidthNorm4         |   23.10   | 49.63 |   10.85   |   23.32   |   19.13   |   29.85   | 51.70 | 81.93 |
|        **ConvexNorm4**        |   27.87   | 49.84 |   15.67   |   30.92   |   25.44   |   39.41   | 53.53 | 81.98 |
|         **MaxNorm4**          |   28.29   | 49.66 |   16.21   | **31.26** | **25.70** | **39.88** | 53.06 | 81.97 |
|             IOUd5             |   27.95   | 50.33 |   15.64   |   29.46   |   24.28   |   37.44   | 55.58 | 81.96 |
|             IOU5              |   26.70   | 50.22 |   14.30   |   28.32   |   23.33   |   36.03   | 55.11 | 81.97 |
|          **GIOUd5**           | **30.11** | 50.22 | **18.20** | **31.36** | **25.85** | **39.85** | 55.24 | 81.99 |
|             GIOU5             |   28.18   | 50.12 |   15.97   |   29.32   |   24.15   |   37.30   | 55.12 | 81.95 |
| DIOUd5 <br> (`Vanilla model`) |   23.31   | 49.68 |   11.03   |   22.87   |   18.76   |   29.28   | 51.74 | 81.89 |
|             DIOU5             |   29.15   | 50.12 |   17.09   |   30.44   |   25.06   |   38.75   | 55.13 | 81.97 |
|            CIOUd5             |   28.09   | 50.24 |   15.83   |   29.41   |   24.24   |   37.40   | 55.40 | 81.98 |
|             CIOU5             |   28.51   | 50.14 |   16.34   |   30.20   |   24.89   |   38.40   | 55.30 | 81.97 |
|            EIOUd5             |   28.11   | 50.28 |   15.83   |   29.46   |   24.27   |   37.47   | 55.23 | 81.99 |
|             EIOU5             |   27.72   | 50.25 |   15.41   |   29.28   |   24.13   |   37.23   | 55.27 | 81.95 |
|          DIOUd-Cos6           |   29.54   | 50.14 |   17.51   |   31.09   |   25.63   |   39.51   | 55.31 | 81.96 |
|      IouFamily8-vanilla       |   27.54   | 50.09 |   15.27   |   29.77   |   24.52   |   37.86   | 55.25 | 81.95 |
|       IouFamily8-convex       |   28.58   | 49.81 |   16.50   |   30.65   |   25.21   |   39.09   | 54.35 | 81.99 |
|      IouFamily8-separate      |   28.39   | 49.69 |   16.29   | **31.20** | **25.67** | **39.75** | 53.75 | 81.96 |

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

|            Experiment             |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :-------------------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | ----- |
|      IouFamily8-vanilla-seq       |   28.04   | 50.24 |   15.75   |   29.57   |   24.36   |   37.64   | 55.17 | 81.98 |
|       IouFamily8-convex-seq       | **32.89** | 49.64 |   21.94   | **36.89** | **30.37** | **46.99** | 55.37 | 81.98 |
|      IouFamily8-separate-seq      |   29.32   | 49.74 |   17.39   |   31.69   |   26.08   |   40.39   | 54.06 | 81.96 |
| **ConvexNorm4-v2**<br>[`deepMsg`] |   42.18   | 50.24 |   35.55   |   48.97   |   40.36   |   62.24   | 58.53 | 82.01 |
|  **MaxNorm4-v2**<br>[`deepMsg`]   | **42.68** | 49.96 | **36.60** | **49.09** | **40.48** | **62.38** | 58.72 | 81.96 |
|          ConvexNorm4-v2           |   29.56   | 49.64 |   17.71   |   32.76   |   26.94   |   41.78   | 53.89 | 81.95 |
|          **MaxNorm4-v2**          | **31.77** | 49.63 | **20.48** | **34.85** | **28.66** | **44.46** | 53.77 | 82.00 |
|             GIOUd5-v2             |   28.30   | 49.84 |   16.18   |   30.87   |   25.37   |   39.37   | 53.11 | 82.00 |
|             GIOU5-v2              |   28.72   | 49.67 |   16.71   |   31.00   |   25.49   |   39.56   | 53.12 | 82.01 |
|             DIOU5-v2              |   28.06   | 49.62 |   15.96   |   30.17   |   24.82   |   38.47   | 52.62 | 82.00 |
|             CIOU5-v2              |   25.69   | 49.37 |   13.50   |   25.62   |   20.98   |   32.90   | 51.49 | 81.94 |

#### 4.5.2 Attention Mechanism [:eyes:]

<a id='Attention'></a>

### 4.7 After more Graph Neural Network [:tada:]

Motivated by the influence of self loop, I wanna explore more about variants graph convolution operations. Although all of these variants are based on `MessagePassing` paradigm, there is no any paper about graph-based trackers to figure out the best model in MOT. And I will attempt to change the structure of `vanilla model`  and do experiments to find the best, with my limited knowledge of GNN and MOT. Here are several variants of `vanilla model` that I wanna try:

1. Deepen message function of my graph neural network in `Vanilla model`.(marked as `deepMsg`)
   $$
   ^{s}g^{l+1}_i:= \Phi\{\max _{j \in \mathbb{N}_i}{f([~Edge~emb ~\cdot~(^{s}g_j^{l} - ^{s}g_i^{l})~ ]\}})\\
   ^{s}g^{l+1}_i:= \max _{j \in \mathbb{N}_i}{f([^{s}g_i^{l}~\cdot~(^{s}g_j^{l} - ^{s}g_i^{l})~]})\\
   $$

   - The reason why I wanna deepen my message function of vanilla model is that there is only one full connected layer $f(\cdot )$ to fuse the feature $Edge~emb$ and $g_j -g_i$, which has not enough ability to align the feature space, thus harmful to my model. So I wanna deepen my message function.:pray:

2. Change aggregation `max` to `mean`: (marked as `meanAggr`)
   $$
   ^{s}g^{l+1}_i:=  \Phi (\frac{1}{N_i}\sum _{j \in \mathbb{N}_i}{f([Edge~emb\cdot~(^s g_j^{l} - ^s g_i^{l}) ])})\\
   ^{d}g^{l+1}_i:= \frac{1}{N_i}\sum _{j \in \mathbb{N}_i}{f([^d g_i^{l}~\cdot~(^d g_j^{l} - ^d g_i^{l})])}
   $$

3. Reimplement graph convolutions: (which is similar to Graphconv  || marked as `Graphconv`)
   $$
   ^{s}g^{l+1}_i:= \Phi \{~f_1(^{s}g_i^l) +\max _{j \in \mathbb{N}_i}{f([Edge~emb\cdot~(^s g_j^{l} - ^s g_i^{l}) ])}~\}\\
   ^{d}g^{l+1}_i:= \max _{j \in \mathbb{N}_i}{f([^d g_i^{l}~\cdot~(^d g_j^{l} - ^d g_i^{l})])}
   $$

   - Let\`s consider an extremely special case that there is only one object detected. In such cases, the graph contains only one nodes and only one edge , i.e. its self loop. The node features extracted by `NodeEncoder` are normal, while edge features generated by `EdgeEncoder` become a extremely sparse tensor. After my model calculating, the output node features also become a extremely sparse tensor, whose essential features, like appearance features, are lost in the calculation of static graph convolution network. So inspired by the idea of `Graphconv` and residual connection , I propose this formulation which can keep some inherent features rather than differences. And hoping it can improve the robustness of my model :pray: 

4. Add edge embedding in dynamic graph: ( Because my model will construct the dynamic graph in higher feature space, it\`s of necessity to **recalculate the edge embedding** ,recorded as $Edge~emb^*$ || marked as `DoubleEdgeEmb`) 
   $$
   ^{s}g^{l+1}_i:=& \Phi \{~f_1(^{s}g_i^l) +\max _{j \in \mathbb{N}_i}{f([Edge~emb~\cdot ~(^s g_j^{l} - ^s g_i^{l}) ])}~\}\\
   ^{d}g^{l+1}_i:=& \max _{j \in \mathbb{N}_i}{f([^{d}g_i^l~ \cdot ~Edge~emb^* ~\cdot ~(^d g_j^{l} - ^d g_i^{l})])}
   $$

5. Extra concatenate of node\`s own features: (marked as `selfConcat`)]
   $$
   ^{s}g^{l+1}_i:=& \Phi \{~\max _{j \in \mathbb{N}_i}{f([^s g_i^{l}~\cdot~Edge~emb~\cdot ~(^s g_j^{l} - ^s g_i^{l}) ]~\}} \\
   ^{d}g^{l+1}_i:=& \max _{j \in \mathbb{N}_i}{f([^d g_i^{l}~\cdot~Edge~emb^* \cdot ~(^d g_j^{l} - ^d g_i^{l})~ ])}
   $$

|          Model Variants          |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :------------------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
|     Vanilla one<sup>*</sup>      |   23.31   | 49.68 |   11.03   |   22.87   |   18.76   |   29.28   | 51.74 | 81.89 |
|           **deepMsg**            | **28.11** | 50.12 | **15.86** | **29.37** | **24.20** | **37.35** | 55.46 | 82.00 |
|             meanAggr             |   23.73   | 49.62 |   11.45   |   23.37   |   19.18   |   29.91   | 52.07 | 81.93 |
|   Graphconv <br>*w/* self-loop   |   24.98   | 50.13 |   12.53   |   24.32   |   20.01   |   30.99   | 53.50 | 81.98 |
|  Graphconv <br/>*w/o* self-loop  |   22.93   | 49.51 |   10.70   |   23.10   |   18.95   |   29.60   | 51.40 | 81.94 |
|  DouleEdgeEmb<br>*w/* self-loop  |   26.51   | 50.16 |   14.12   |   26.95   |   22.19   |   34.32   | 54.26 | 81.98 |
| DouleEdgeEmb<br/>*w/o* self-loop |   24.59   | 49.65 |   12.27   |   24.92   |   20.47   |   31.86   | 52.55 | 81.94 |
|          **selfConcat**          | **28.64** | 50.05 | **16.55** | **29.99** | **24.70** | **38.15** | 54.89 | 81.92 |

----

Noticed that the big improvement by `deepMsg`, I think the misalignment of edge embedding feature space and node embedding feature space is quite harmful to my model. And this problem becomes more serious with increasing stacking GNN layers.  :confused: In my opinion, this problem also leads to the weird and counterintuitive, at least for me , phenomenon in [Sec 4.6.1 several Variants of Edge Embedding](#weirdPhenomenon). More specifically, the more information the original edge embedding contains , the more complicated the original edge embedding feature space is. And it is quite impossible for fewer fc layers to align the edge feature space and node feature space. And it is necessary to redo some experiments concerning variants of edge embedding by utilizing some fc layers to transform edge embedding into a suitable feature space. And there are two measures I wanna take:

1. **Attention Mechanism**(more details in  [Sec 4.6.2](#Attention))： Especially [CEN-DGCNN](https://www.nature.com/articles/s41598-023-44224-1)-like attention mechanism, which updates the edge embedding with increasing of GCNN.
2. **Simple Strategy**： I just add some extra fc layers to transform edge embedding into a suitable feature space at the beginning of each GNN.

Now let\`s focus on **Simple Strategy** in this Section. Considering the size of edge embedding, this can further be divided into three approaches: the first is to always maintain consistency in the feature dimensions, i.e. 16 dimensions , the second is to allow the dimension of edge embedding to increase progressively as GCNN layers are stacked, in parallel with the increasing dimensions of node features., while the last one is to utilize raw edge attribute , like the formulations in [Sec 4.6.1 ](#weirdPhenomenon), without any fc layers to  elevate dimension to 16-dim. Named separately as `FixedEdgeDim` , `ProgressiveEdgeDim` and `RawEdgeAttr` .

P.S. the following experiments will not apply `deepMsg`, for fair comparison.



### 4.8 After Cosine-based Dynamic Graph [:tada:]

Considering that **the measurement based on cosine similarity in the graph match**, I change the dynamic graph based Euclidean distance to **cosine distance based** , just hoping my model can learn more discriminative features in the feature space based on cosine distance. 

|       Conditions        |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  | MOTP  |
| :---------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :---: |
| Vanilla one<sup>*</sup> |   23.31   | 49.68 |   11.03   |   22.87   |   18.76   |   29.28   | 51.74 | 81.89 |
|      Cosine-based       | **27.19** | 50.33 | **14.81** | **28.39** | **23.40** | **36.08** | 55.34 | 81.93 |

<img src="./.assert/cosinegraph-index.bmp" alt="cosinegraph-index" style="zoom:25%;" />

### 4.9 After Superior Appearance Feature Extractor [:eyes:]



### 4.10 After Larger Dataset [:tada:]

One of the core and useful experience or intuition in the era of deep learning is that `if you have a large big dataset and you train a very big neural network, then success is guaranted` (quoted from [llya`s speech at NeurlPS conference in 2024](https://www.bilibili.com/video/BV1cSBGYkE9w/?spm_id_from=333.337.search-card.all.click&vd_source=812705912b7abe259d54d8593a97a8b3)) , like CLIP model trained in 400 million dataset.

So I also attempt to add more benchmark, like MOT20 and DanceTrack, to enlarge the training dataset. Finally, the number of data is reached to **48856**, which is **20x times than before**. And the time expense is also huge , which maybe cost **3 days** to complete the whole training, **3x times than before.**

P.S. the model structure of Larger Dataset is slightly different Vanilla one<sup>*</sup>, where I **add another dim into edge embedding , replace edgeconv into graphconv and change dynamic graph based on Euclidean distance to Cosine distance based**. All of modifications **poses negative influences** on model performance according to the experimental results in this documentation. **Therefore, the impact of data volume on model performance is quite evident.**

|       Conditions        |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    | MOTA  |   MOTP    |
| :---------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :---: | :-------: |
| Vanilla one<sup>*</sup> |   23.31   | 49.68 |   11.03   |   22.87   |   18.76   |   29.28   | 51.74 |   81.89   |
|     Larger Dataset      | **43.19** | 39.26 | **47.78** | **45.18** | **33.01** | **71.60** | 41.55 | **88.47** |

## 5. Experimental Records [Track Management]

In [[Sec 1. Brief Introduction about My Model]](#1. Brief Introduction about my model), I have a brief introduction about the trajectory management in the whole pipeline, but it\`s necessary to supplement more information in order to understand what I\`ll do in this section.

![statetransition](./.assert/statetransition.bmp)

To sum up, here is the state transition network of my track management. And there are four states of my single trajectory. And it\` extremely important to understand the whole switching process in my track management. 

The detailed significance of these four states of each trajectory:

1. `BORN state`\:  only those high-confidence detections (confidence >= 0.7) filtered by two-phase matching can be initialized as `BORN state` trajectory in the whole pipeline of track management. And 
2. `ACTIVE state`:
3. `SLEEP state`:
4. `DEAD state`: 

### 5.1 plz live longer [:eyes:]

### 5.2 more robust Appearance Fuse Strategies [:eyes:]

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
