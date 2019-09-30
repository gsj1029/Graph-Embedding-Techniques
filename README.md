# Graph-Embedding-Techniques
It provides some interesting graph embedding techniques based on task-free or task-specific intuitions.

### Table of Contents  

1. [Pure Network Embedding](#1-pure-network-embedding)
    * 1.1. [Node Proximity Relationship](#11-node-proximity-relationship)
    * 1.2. [Structural Identity](#12-structural-identity)
2. [Attributed Network Embedding](#2-attributed-network-embedding)
    * 2.1. [Attribute Vectors](#21-attribute-vectors)
    * 2.2. [Text Content](#22-text-content)
3. [Graph Neural Networks](#3-graph-neural-networks)
    * 3.1. [Node Classification](#31-node-classification)
    * 3.2. [Graph Classification](#32-graph-classification)
    * 3.3. [Link Prediction](#33-link-prediction)
    * 3.4. [Community Detection](#34-community-detection)
    * 3.5. [Rare Category Characterization](#35-rare-category-characterization)
4. [Graph Kernels](#4-graph-kernels)


### 1. Pure Network Embedding

#### 1.1. Node Proximity Relationship

- **DeepWalk: Online Learning of Social Representations (KDD'14).** [[Paper]](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf) [[Python Code]](https://github.com/phanein/deepwalk)

- **LINE: Large-scale Information Network Embedding (WWW'15).** [[Paper]](http://www.www2015.it/documents/proceedings/proceedings/p1067.pdf) [[C++ Code]](https://github.com/tangjianpku/LINE)

- **node2vec: Scalable Feature Learning for Networks (KDD'16).** [[Paper]](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf) [[Project]](https://snap.stanford.edu/node2vec/#code)[[Python Code]](https://github.com/aditya-grover/node2vec)

- **Watch Your Step: Learning Node Embeddings via Graph Attention (NIPS'18).** [[Paper]](https://arxiv.org/pdf/1710.09599.pdf) [[Python Code]](https://github.com/benedekrozemberczki/AttentionWalk)

- **Deep Graph Infomax (ICLR'19).** [[Paper]](https://arxiv.org/pdf/1809.10341.pdf) [[OpenReview]](https://openreview.net/forum?id=rklz9iAcKQ) [[Code]](https://github.com/PetarV-/DGI)

#### 1.2. Structural Identity

- **struc2vec: Learning Node Representations from Structural Identity (KDD'17).** [[Paper]](https://arxiv.org/pdf/1704.03165.pdf) [[Python Code]](https://github.com/leoribeiro/struc2vec)

- **Learning Structural Node Embeddings via Diffusion Wavelets (KDD'18).** [[Paper]](https://cs.stanford.edu/people/jure/pubs/graphwave-kdd18.pdf) [[Project]](http://snap.stanford.edu/graphwave/) [[Python Code]](https://github.com/snap-stanford/graphwave)

### 2. Attributed Network Embedding

#### 2.1 Attribute Vectors

- **Label Informed Attributed Network Embedding (WSDM'17).** [[Paper]](http://www.public.asu.edu/~jundongl/paper/WSDM17_LANE.pdf) [[MATLAB Code]](https://github.com/xhuang31/LANE)

- **Accelerated Attributed Network Embedding (SDM'17).** [[Paper]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974973.71) [[Python Code]](https://github.com/xhuang31/AANE_Python) [[MATLAB Code]](https://github.com/xhuang31/AANE_MATLAB)

- **Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking (ICLR'18).** [[Paper]](https://arxiv.org/pdf/1707.03815.pdf)[[OpenReview]](https://openreview.net/forum?id=r1ZdKJ-0W) [[Python Code]](https://github.com/abojchevski/graph2gauss)


#### 2.2. Text Content

- **Network Representation Learning with Rich Text Information (IJCAI'15).** [[Paper]](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) [[MATLAB Code]](https://github.com/albertyang33/TADW)

- **CANE: Context-Aware Network Embedding for Relation Modeling (ACL'17).** [[Paper]](http://nlp.csai.tsinghua.edu.cn/~tcc/publications/acl2017_cane.pdf) [[Python Code]](https://github.com/thunlp/CANE)

- **Diffusion Maps for Textual Network Embedding (NIPS'18).** [[Paper]](http://papers.nips.cc/paper/7986-diffusion-maps-for-textual-network-embedding) [[Python Code]](https://github.com/dylanz0426/DMTE)


### 3. Graph Neural Networks 

#### 3.1. Node Classification

- **Diffusion-Convolutional Neural Networks (NIPS'16).** [[Paper]](https://arxiv.org/pdf/1511.02136.pdf) [[Code]](https://github.com/jcatw/dcnn)

- **Geometric Deep Learning: Going beyond Euclidean data (SPM'17).** [[Paper]](https://ieeexplore.ieee.org/abstract/document/7974879) [[Project]](http://geometricdeeplearning.com)

- **Inductive Representation Learning on Large Graphs (NIPS'17).** [[Paper]](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf) [[Project]](http://snap.stanford.edu/graphsage/) [[Code]](https://github.com/williamleif/GraphSAGE)

- **Semi-Supervised Classification with Graph Convolutional Networks (ICLR'17).** [[Paper]](https://arxiv.org/pdf/1609.02907.pdf)[[OpenReview]](https://openreview.net/forum?id=SJU4ayYgl) [[Code]](https://github.com/tkipf/gcn)

- **Neural Message Passing for Quantum Chemistry (ICML'17).** [[Paper]](https://arxiv.org/pdf/1704.01212.pdf) [[Code]](https://github.com/priba/nmp_qc)

- **Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning (AAAI'18).** [[Paper]](https://arxiv.org/pdf/1801.07606.pdf) [[Code]](https://github.com/YuCheng12345/gcn_Co-Training_Self-Training)

- **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling (ICLR'18).** [[Paper]](https://arxiv.org/pdf/1801.10247.pdf)[[OpenReview]](https://openreview.net/forum?id=rytstxWAW) [[Code]](https://github.com/matenure/FastGCN)

- **Stochastic Training of Graph Convolutional Networks with Variance Reduction (ICML'18).** [[Paper]](http://proceedings.mlr.press/v80/chen18p.html) [[Code]](https://github.com/thu-ml/stochastic_gcn)

- **Graph Attention Networks (ICLR'18).** [[Paper]](https://arxiv.org/pdf/1710.10903.pdf)[[OpenReview]](https://openreview.net/forum?id=rJXMpikCZ) [[Code]](https://github.com/PetarV-/GAT)

- **Relational Inductive Biases, Deep Learning, and Graph Networks (arXiv'18).** [[Paper]](https://arxiv.org/pdf/1806.01261.pdf) [[Code]](https://github.com/deepmind/graph_nets)

#### 3.2. Graph Classification

- **Learning Convolutional Neural Networks for Graphs (ICML'16).** [[Paper]](http://proceedings.mlr.press/v48/niepert16.pdf) [[Code]](https://github.com/tvayer/PSCN)

- **Deriving Neural Architectures from Sequence and Graph Kernels (ICML'17).** [[Paper]](https://arxiv.org/pdf/1705.09037.pdf) [[Code]](https://github.com/taolei87/icml17_knn)

- **An End-to-End Deep Learning Architecture for Graph Classification (AAAI'18).** [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17146/16755) [[Code]](https://github.com/muhanzhang/pytorch_DGCNN)

- **Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks (AAAI'19).** [[Paper]](https://arxiv.org/pdf/1810.02244.pdf) [[Code]](https://github.com/chrsmrrs/k-gnn)

- **How Powerful are Graph Neural Networks? (ICLR'19).** [[Paper]](https://arxiv.org/pdf/1810.00826.pdf)[[OpenReview]](https://openreview.net/forum?id=ryGs6iA5Km)[[Code]](https://github.com/weihua916/powerful-gnns)

- **Capsule Graph Neural Network (ICLR'19).** [[Paper]](https://openreview.net/pdf?id=Byl8BnRcYm)[[OpenReview]](https://openreview.net/forum?id=Byl8BnRcYm)[[Code]](https://github.com/XinyiZ001/CapsGNN)

#### 3.3. Link Prediction

- **Link Prediction Based on Graph Neural Networks (NIPS'18).** [[Paper]](http://papers.nips.cc/paper/7763-link-prediction-based-on-graph-neural-networks) [[Code]](https://github.com/muhanzhang/SEAL)

#### 3.4. Community Detection

- **Supervised Community Detection with Line Graph Neural Networks (ICLR'19).** [[Paper]](https://arxiv.org/pdf/1705.08415.pdf) [[Code]](https://github.com/afansi/multiscalegnn)

#### 3.5. Rare Category Characterization

- **SPARC: Self-Paced Network Representation for Few-Shot Rare Category Characterization (KDD'18).** [[Paper]](http://www.public.asu.edu/~dzhou23/papers/KDD2018_SPARC.pdf) [[Code]](https://sites.google.com/view/dawei-zhou/publications)

- **RSDNE: Exploring Relaxed Similarity and Dissimilarity from Completely-imbalanced Labels for Network Embedding (AAAI'18).**
[[Paper]](https://pdfs.semanticscholar.org/ae42/7d5ca009bf1af53af01249e19d67385744ff.pdf) [[Code]](https://github.com/zhengwang100/RSDNE)

### 4. Graph Kernels

- **Matching Node Embeddings for Graph Similarity (AAAI'17).** [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14494/14426) [[Code]](http://www.lix.polytechnique.fr/Labo/Ioannis.Nikolentzos/)

- **RetGK: Graph Kernels based on Return Probabilities of Random Walks (NIPS'18).** [[Paper]](https://papers.nips.cc/paper/7652-retgk-graph-kernels-based-on-return-probabilities-of-random-walks) [[Code]](https://sites.wustl.edu/zhenzhang/publication/)

- **Anonymous Walk Embeddings (ICML'18).** [[Paper]](http://proceedings.mlr.press/v80/ivanov18a/ivanov18a.pdf) [[Code]](https://github.com/nd7141/AWE)

- **Scalable Global Alignment Graph Kernel Using Random Features: From Node Embedding to Graph Embedding (KDD'19).** [[Paper]](http://mason.gmu.edu/~lzhao9/materials/papers/rt1709o-wuAemb.pdf) [[Code]](https://sites.google.com/a/email.wm.edu/teddy-lfwu/publications?authuser=0)
