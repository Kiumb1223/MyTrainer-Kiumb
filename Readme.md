[toc]

# Readme

> make a small record about my train of thought when trying to apply GNN to MOT

## 1. Brief Introduction about My Model

> Reference paper : [Recurrent graph optimal transport for learning 3D flow motion in particle tracking(Nature 2023)](https://www.nature.com/articles/s42256-023-00648-y)

![pipeline](./.assert/pipeline.bmp)

My model is an **online and graph-based** tracker. The main difference from other online and graph-based trackers is that the way of constructing graph and graph convolutions , which is innovative in the realm of MOT but also brings some limitations that still puzzle me:thinking:. So the following discussions mainly focus on the above two points.

Here comes the first discussion which is the **construction of a graph**. Specifically, the appearance features extracted by CNN Model of each object is initialized as node embedding , and the 2 dimension pixel position of each object is used to compute graph structure and edge embedding.

1. **Graph structure** : I utilize **KNN algorithm** to construct a **self-looped and directed** graph;

2. **Edge embedding** : In order to model the relative relationship between connecting nodes , I concatenate the 6 elements below and feed them to a MLP module to encode a 16 dimension features as edge embedding.

$$
Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),diouDist(i,j))])
$$


And in order to **track each object** , there are two graphs which are built on above rules — one is based on the past trajectories and the other is based on current detections. So, the MOT problem can be converted to:star2: **graph matching problem**:star2:, which is NP complete.

Additionally, let\`s talk about my **graph convolutional operations**, which is mainly from the [paper](https://www.nature.com/articles/s42256-023-00648-y), i.e. Edgeconv. The functions goes like this, which is **minor different from the original version** : ( $[\cdot]$ means concatenation )


$$
g^{l+1}_i:= \max _{j \in \mathbb{N}_i}{f([g_i^{l}~\cdot~(g_j^{l} - g_i^{l})~\cdot~ Edge~emb])}
$$

And in order to **capture high-order discriminative features**, I rebuild the graph, which I call the `dynamic graph`, while the old one is referred to as the `static graph` to keep things clear. And I also perform the edgeconv operations on dynamic graph:


$$
g^{l+1}_i:= \max _{j \in \mathbb{N}_i}{f([g_i^{l}~\cdot~(g_j^{l} - g_i^{l})])}
$$

----

The last but not least, it\`s necessary to take a glimpse of my **trajectory management strategy**. My trajectory management strategy mainly refers to **ByteTrack**. And here is my program flow chart:

<img src="./.assert/flowChart.png" alt="flowChart" style="zoom: 67%;" />

There are four states in the trajectory management strategy — **Born,Active,Sleep,Dead**. It\`s possible that `Born Trajectory` can be false positive, i.e. noise, so there are two phases in trajectory management — `Match Strategy` for denoising and `Graph Matching` for matching. And the rest of the strategy is unnecessary to discuss, which is obvious in the flow chart.

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

- For the purpose of  **fast training (36min or so) and relatively fair comparison** , my own model also **trains on half data  of MOT and tests on MOT17-half**. Besides, **max epoch is set to 40, batch size is set to 16, warmup iterations are set to 500**.
- And **random seed is set to 3407 ** || **k is set to 2 **

Here is the original quantitative results of my model without **any modification on model structure or the use of any data augmentation techniques**. 

| Validation set  | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MOT17-half(SDP) | 23.96 | 45.44 | 12.66 | 23.14 | 18.08 | 32.14 | 41.54 | 83.79 |

Obviously, my model still have a long way to go. However, what makes me proud is that my model has **a relatively fast inference speed**, which can reach up to 15 fps or so to process a 30fps, 1080p video.

----



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

## 4. Experimental Record [Technique & Hyperparameters]

The quantitative results of the vanilla model  (the same results in [[Sec 1]](#1. Brief  Introduction about My Model), just repeat one more time and add some curves which can reflect something):

| Validation set  | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :-------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MOT17-half(SDP) | 23.96 | 45.44 | 12.66 | 23.14 | 18.08 | 32.14 | 41.54 | 83.79 |

![vanilaOne-index](./.assert/vanilaOne-index.bmp)

Apart from the quantitative results of the vanilla one , let\`s focus on these two curves. The training data is 2650, which is relatively small and easily make models overfit. So I have to design more precise training strategy to avoid overfitting, where I adjust the warmup strategy and multiStep lr scheduler.

Noted that the f1 curve in evaluation phase surges to 0.9 or so and then slowly grows up, this f1 score is slightly different from one in MOT metric. Both of them is based on ID of tracked object, but the f1 score here is calculated in each timestamp — in other words, **the default prerequisite here is that the tracking result of each timestamp is independent**, which is quite different from one in MOT metric.

As we all known, the MOT problem can be viewed as a problem of **maximizing a posteriori probability** —— the tracking result of each timestamp is quite dependent on the results of previous moment. It\`s reasonable to infer that the worse tracking results from previous moment, the worse tracking results from current moment. Actually, the performance of my model is indeed like this.

### 4.1 After data Augmentation [🎉]

> In order to avoid overfitting, it\`s overwhelmingly necessary to find out the various and valid data augmentation techniques.

![dataAugmentation ](./.assert/dataAugmentation.bmp)

There are three data augmentation techniques — Low framerate, missed detections and discontinuous trajectories. All of them is vividly showed in the above picture. So let\`s see the quantitative results of vanilla model after training. Oops, I changes some experimental settings. In this experiment, the total epoch is set to 120  (it maybe takes 2 hours or so in GTX3090 ), warmup iteration is set to 800 and multistep is set to 50 and 80.(Waiting to see :eyes:)

|       Conditions        |   HOTA    |   DetA    | AssA  |   IDF1    |    IDR    |  IDP  |   MOTA    |   MOTP    |
| :---------------------: | :-------: | :-------: | :---: | :-------: | :-------: | :---: | :-------: | :-------: |
| Vanilla one<sup>*</sup> |   23.96   |   45.44   | 12.66 |   23.14   |   18.08   | 32.14 |   41.54   |   83.79   |
|    Data Augmentation    | **25.29** | **51.08** | 12.57 | **25.02** | **20.49** | 32.13 | **50.01** | **83.86** |

![dataAug](./.assert/dataAug-index.bmp)

:loudspeaker: Obviously,**my model is not overfitting in the whole training phase due to three data augmentation techniques. And I wanna  **<strong style="color: red;">use this as the main comparison which marks as Vanilla one<sup>*</sup>.</strong> 

### 4.2 After Graphconv [:sob:]

And I wanna try another way to implement graph convolution operations: (which is similar to Graph conv)
$$
g^{l+1}_i:= f_1(g_i^l) + \max _{j \in \mathbb{N}_i}{f_2([g_i^{l}~\cdot~(g_j^{l} - g_i^{l})])}
$$
Oops,there is a big drop compared with previous results, which is beyond my expectations.:sob:

|       Conditions        |   HOTA    |   DetA    |   AssA    |   IDF1    |    IDR    |    IDP    |   MOTA    | MOTP  |
| :---------------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :---: |
| Vanilla one<sup>*</sup> | **25.29** |   51.08   | **12.57** | **25.02** | **20.49** | **32.13** | **50.01** | 83.86 |
|        Graphconv        |   20.83   | **51.12** |   8.52    |   18.45   |   15.09   |   23.73   |   42.72   | 83.86 |

![dataAug](./.assert/graphconv-index.bmp)

### 4.3 After Undirected graph [:sob:]

Here is the simple illustration about the undirected graph:

![undirectedgraph](./.assert/undirectedgraph.bmp)

Unfortunately, it doesn`t seem to have a significant improvement, but rather a slight decrease. :sob:

|       Conditions        |   HOTA    |   DetA    |   AssA    |   IDF1    |    IDR    |    IDP    |   MOTA    |   MOTP    |
| :---------------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Vanilla one<sup>*</sup> | **25.29** | **51.08** | **12.57** | **25.02** | **20.49** | **32.13** | **50.01** | **83.86** |
|    Undirected Graph     |   23.88   |   51.02   |   11.22   |   23.13   |   18.93   |   29.72   |    50     |   83.80   |

![UndirectedGraph-index](./.assert/UndirectedGraph-index.bmp)

### 4.4 After Distance mask [:eyes:]

![distanceMask](./.assert/distanceMask.bmp)

What\` s distance mask here ? Actually, it is a kind of precondition which is that the movement distance of objects in adjacent frames is likely to be small, but **sometimes can be large when in low framerate video or objects moving fast**. So it\`s necessary to statistically calculate **the max movement distance** of different situations, like **different framerate, fast-moving objects or video from a moving perspective.**  And here is the statistical results in **MOT17(Train set), MOT20(Train set), DanceTrack(Train && val set)**.

![DifferentWindow](./.assert/DifferentWindow.bmp)

Noted that it seems that the **max movement distance of adjacent frames (window - 2) is more reliable**. Let\`s dive into the situations of **windowSize[2]**.

![MOT17](./.assert/MOT17_20.bmp)

![Dancetrack](./.assert/Dancetrack.bmp)

It seems that **the speed of object moving poses the bigger influence on the statistical experiment among all factors.**:confused: Oops, maybe **the distance between camera and objects also matters!**



### 4.5 After Different K [:tada:]

According to the descriptions about construction graph in [[Sec 1]](#1. Brief Introduction about My Model), there is a hyperparameter `k` in knn algorithm. And k is set to 2 in my vanilla model, so let`s search for best k.

|         Different K          | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :--------------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| k=2(Vanilla one<sup>*</sup>) | 25.29 | 51.08 | 12.57 | 25.02 | 20.49 | 32.13 | 50.01 | 83.86 |
|             k=3              | 27.29 | 51.14 | 14.61 | 27.84 | 22.83 | 35.67 | 52.56 | 83.82 |
|             k=4              | 28.87 | 51.33 | 16.30 | 30.06 | 24.67 | 38.46 | 53.53 | 83.85 |
|             k=5              |       |       |       |       |       |       |       |       |
|             k=6              |       |       |       |       |       |       |       |       |
|             k=7              |       |       |       |       |       |       |       |       |
|             k=8              |       |       |       |       |       |       |       |       |
|          k=5(6-dim)          | 27.75 | 50.37 | 15.32 | 27.78 | 22.60 | 36.02 | 51.06 | 83.84 |
|          k=6(6-dim)          | 31.47 | 50.82 | 19.54 | 32.69 | 26.71 | 42.12 | 54.23 | 83.80 |



### 4.6 After Weight of edge 

The reason why I wanna change the weight of edge is **the week connection between similar objects or closely positioned objects.** In other words, for similar objects in close positions, **the current model has week differentiation capability (i.e. insufficient feature discriminability).** More details in the following picture.

![whyEdge](./.assert/whyEdge.bmp)

How to alleviate or even solve this problem? Some trial methods are waiting for me to practice.

#### 4.6.1 Add Cosine Distance in edge embedding [:confused:]

I all **cosine distance of connected nodes** to edge embedding of static graph:

$$
Edge~emb := f([\frac{2(x_j - x_i)}{h_i + h_j},\frac{2(y_j-y_i)}{h_i+h_j},log(\frac{w_j}{w_i}),log(\frac{h_j}{h_i}),diouDist(i,j),cosineDist(i,j))])
$$

And here are the quantitative results blow：（Compared with the results of dataAugmentation, it\`s slightly smaller :confused:）

|       Conditions        |   HOTA    | DetA  |   AssA    |   IDF1    |    IDR    |    IDP    |   MOTA    |   MOTP    |
| :---------------------: | :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Vanilla one<sup>*</sup> |   25.29   | 51.08 |   12.57   | **25.02** | **20.49** | **32.13** | **50.01** | **83.86** |
|     6-dim Edge emb      | **25.41** | 51.08 | **12.70** |   23.81   |   19.50   |   30.67   |   48.83   |   83.84   |

![AddCosineDist-index](./.assert/AddCosineDist-index.bmp)

#### 4.6.2 Attention Mechanism [:eyes:]



### 4.7 After Enlarge Dataset [:tada:]

|       Conditions        | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :---------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla one<sup>*</sup> | 25.29 | 51.08 | 12.57 | 25.02 | 20.49 | 32.13 | 50.01 | 83.86 |
|     Larger Dataset      | 43.19 | 39.26 | 47.78 | 45.18 | 33.01 | 71.60 | 41.55 | 88.47 |

### 4.8 After Cosine-based Dynamic Graph [:sob:]

|       Conditions        | HOTA  | DetA  | AssA  | IDF1  |  IDR  |  IDP  | MOTA  | MOTP  |
| :---------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla one<sup>*</sup> | 25.29 | 51.08 | 12.57 | 25.02 | 20.49 | 32.13 | 50.01 | 83.86 |
|      Cosine-based       | 22.27 | 51.00 | 9.78  | 20.41 | 16.68 | 26.29 | 45.91 | 83.83 |



### 4.9 After Superior Appearance Feature Extractor [:eyes:]

### 4.10 After the same GCN opts in sGraph and dGraph [:eyes:]

Suddenly, a idea comes up to my brain — why I **do different GCN operations in static graph and dynamic graph separately**, **and why not do the same opts in both graphs??**

So here comes the formulation of static graph and dynamic graph. Because my model will construct the dynamic graph in higher feature space, it\`s of necessity to **recalculate the edge embedding**  (marks as $Edge~emb^*$) :
$$
^{s}g^{l+1}_i:=& \max _{j \in \mathbb{N}_i}{f([^s g_i^{l}~\cdot~(^s g_j^{l} - ^s g_i^{l})~\cdot ~Edge~emb])}\\
^{d}g^{l+1}_i:=& \max _{j \in \mathbb{N}_i}{f([^d g_i^{l}~\cdot~(^d g_j^{l} - ^d g_i^{l})~\cdot ~Edge~emb^*])}
$$
And waiting to see :eyes:



## 5. Experimental Record [Track Management]

In [[Sec 1]](#1. Brief Introduction about my model), I have a brief introduction about the trajectory management in the whole pipeline, but it\`s necessary to supplement more information in order to understand what I\`ll do in this section.

![statetransition](./.assert/statetransition.bmp)

To sum up, here is the state transition network of my track management. And there are four states of my single trajectory. And it\` extremely important to understand the whole switching process in my track management. 

The detailed significance of these four states of each trajectory:

1. BORN state:
2. ACTIVE state:
3. SLEEP state:
4. DEAD state: 

### 5.1 plz live longer [:eyes:]

### 5.2 more robust appearance [:eyes:]
