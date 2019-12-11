# mgs-score-regressor

This is a master thesis thema.


下载最新文件
要到脸部识别
做一次脸部文件，下载，上传
同一标签的老鼠全放test
把cuda问题解决掉（可能是kfold)


## Data split:

I have 3 experiments: KXN, IN, K. Among them, KXN has the largest amount of data. So we train on this subset at first:

- Setting the size of train/validation/test dataset as 8:1:1
- For test dataset, we include not only all 900+ images with mgs score, but also balanced data such as 1000+ 'pain' images / 1100 'no pain' images.
- the validation dataset is also balanced by StratifiedKFold.


## Parameters

Using ResNet152, batch_size 128, lr=0.001, 50 epochs I get following result:
- Adam: val_acc= 93.73%
- Radam: val_acc= 93.31%, TPR/TNR=0.938,0.945 on test data

#### But I notice that, the model could be better if I train it more than 50 epochs next time. depends on the val acc history diagram. The validation accuracy has still not converged.


-时间表
－总览
－内容ｃｈａｐｔｅｒ解释原因
- related work section and why it's important
- experiments why we do it
- 加入topic in my email


<pre><code>

</code></pre>
