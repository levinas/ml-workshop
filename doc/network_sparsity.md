



| Network Model  | Year | Depth | Parameters         | Dataset     | FLOPs | Training Size  |     Compression (1/X) |
| -------        | ---- | ----: | -----------------: | ----------: | ----: | -------------: |      ---------------: |
| LeNet-5        | 1998 |     5 | 431 K              | MNIST       |       | 47 MB          |                   109 |
| AlexNet        | 2012 |     8 | 61 M               | ILSVRC-2012 |       | 722 GB         |                  1480 |
| VGGNet-16      | 2014 |    16 | 138 M              | ILSVRC-2012 | 15 G  | 722 GB         |                   654 |
| GoogLeNet      | 2014 |    22 | 5 M                | ILSVRC-2012 |       | 722 GB         |                 18050 |
| ResNet-152     | 2015 |   152 | ~24 M              | ILSVRC-2012 | 11 G  | 722 GB         |                  3760 |
| ResNet-1202    | 2015 |  1202 | 19 M               | CIFAR-10    |       | 614 MB         |      4 <sup>(1)</sup> |
| VGG-16 Pruned  | 2015 |    16 | 10 M               | ILSVRC-2012 |       | 722 GB         | 19,240 <sup>(2)</sup> |
| XNOR-Net(Alex) | 2016 |     8 | (binary) 61 M      | ILSVRC-2012 |       | 722 GB         | 95,720 <sup>(3)</sup> |
| COTS           | 2013 |     3 | 11 B               | 10M-Youtube |       | 4.8 TB         |                    55 |
| LBANN-400K     | 2016 |     3 | 79 B               | ILSVRC-2012 |       | 722 GB         |                  1.15 |
| LBANN-100K     | 2016 |     3 | ~20 B              | YFCC100M    |       | ~50 TB         |                  ~300 |
| Kaggle-gggg    | 2013 |     3 | ~50 M              | Merck-QSAR  |       | ~1 GB          |                   ~20 |


Notes:  
(1) Accuracy of the 1202-layer ResNet is not as good as a 110-layer version, possibly due to overfitting.  
(2) A three-step pruning method reduces AlexNet's connections by 13X with no loss of accuracy.  
(3) This binary version of the AlexNet saves memory by 64X, causing 3% accuracy loss (Top-5: 80%->77%).  
