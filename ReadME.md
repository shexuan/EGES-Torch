### Pytorch实现的EGES
实现了skip-gram方式的EGES，loss部分采用了负采样。

模型使用参考训练模型的notebook，使用`forward_input`函数得到'warm' item的Embedding，使用`forward_cold`函数得到'cold' item 的Embedding，
这里直接使用的负采样item的对数似然作为loss，略加更改可改为类似于Tensorflow sampledsoftmax的loss，待后续补上。

注1：demo中数据预处理部分为简便直接将user的所有item视为一个sentence，省去了deepwalk部分，因为网上已有很多实现，后续有时间的话会补上。

注2：在例子中所有Embedding table均使用index 0作为padding，因模型中支持多值side info，如电影的side类型——动作/喜剧，
    这种多值在输入时需要padding到统一长度，因此将所有字段进行了padding，可视自己情况进行更改，模型中提供了是否进行padding参数，
    也即item index是否从0开始。

参考文献：

Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba.

