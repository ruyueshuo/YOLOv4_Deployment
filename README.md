# 利用 DeepStream5.0 部署自己的 Yolo v4 模型

这个项目将帮助你在darknet框架下创建自己的数据集检测模型（以反光工作服检测为例），并利用英伟达公司的DeepStream在边缘设备(Nvidia Xavier NX 开发板)上进行部署。

在开始这个项目之前，请先确保您对以下内容有所了解。

* Yolov4: https://arxiv.org/abs/2004.10934

* DeepStream: https://developer.nvidia.com/deepstream-sdk

* Darknet: http://pjreddie.com/darknet/

## 0.配置环境
此项目使用的环境主要包含两部分：

* 训练：Ubuntu 18.04.5, CUDA Version: 10.1, Darknet A版, Nvidia 2080 ti 显卡.
* 部署：Nvidia Xavier NX 开发板, JetPack 4.4, TensorRT 7.1, Deepstream 5.0.

其中，Darknet 的配置参考[这里](https://github.com/AlexeyAB/darknet/blob/master/README.md), JetPack 的配置参考[这里](https://developer.nvidia.com/embedded/jetpack),

另外，还使用了项目[pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4),用于将训练好的模型转为ONNX格式。

## 1.数据准备

这个项目使用的反光衣检测数据集来自[这里](https://github.com/gengyanlei/reflective-clothes-detect.git)，
共有反光衣检测数据1083张。

* 进入[百度云盘下载链接](https://pan.baidu.com/s/1_Ei9bYmUpa-8q-hXZk1u8w) 提取码->(dooh).
* 将数据文件夹中‘labels/'下的文件复制到‘JPEGImages/’中，即保证两者在同一个文件夹下。
* 修改配置文件
    ```sh
    cd darkent
    ```
    * 在‘cfg/'路径下创建配置文件‘yolov4-reflective.cfg'，当然可以直接从'yolov4-custom.cfg'直接复制过来。
    * 修改参数，classes = 2， filters=(classes + 5)x3 = 21，注意：每个[yolo]层里的classes及其前一个[convolutional]层里的filters参数都需要相应修改，注意是每一个！
    ```ini
        [convolutional]
        filters=21
        
        [yolo]
        classes=2
    ```
    * 其他参数如batch，learning_rate等可以根据需要修改。
* 准备数据文件，包含四个文件‘reflective.data','reflective.names','train-reflective.txt','valid-reflective.txt'.
    * 进行训练集验证集数据划分，将划分好的文件 'train_reflective.txt' 和 'valid_reflective.txt' 放到 'data/' 文件夹下。
        ```ini
            python3 split_datasets(<imagePath> <savePath>)
        ```
      参数说明：
      
      imagePath即下载数据中‘JPEGImages’文件的路径
      
      savePath即生成文件的保存路径，可以直接设置为'/path where darknet installed/data/'
    * 创建‘reflective.data'：
        ```ini
            classes = 2
            train = data/train-reflective.txt 
            valid = data/valid-reflective.txt
            test = data/test-reflective.txt
            names = data/reflective.names
            backup = reflective/
        ```
    * 创建'reflective.names'：
        ```ini
            other
            reflective
        ```
* 详细过程也可参考darknet的[README](https://github.com/AlexeyAB/darknet/blob/master/README.md)。

## 2.模型训练

* 执行训练代码，会在主目录下生成‘chart.png'，可以查看训练过程的loss和map变化，权重文件保存在‘reflective/’文件夹下：
    ```sh
    cd darknet/
    mkdir reflective
    # yolov4:
    sudo ./darknet detector train ./data/reflective.data  ./cfg/yolov4-reflective.cfg ./pretrained/yolov4.conv.137 -map -gpus 0,1 -dont_show
    # yolov4-tiny:
    sudo ./darknet detector train ./data/reflective.data  ./cfg/yolov4-tiny-reflective.cfg ./pretrained/yolov4-tiny.conv.29 -map -gpus 0,1 -dont_show  
    ```
* 让我们来看看训练效果如何（yolov4-tiny模型）：
    ```sh
    # yolov4-tiny:
    python3 darknet_video.py --input data/test.mp4 --out_filename data/test_result.mp4 --weights reflective/yolov4-tiny-reflective_best.weights  --data_file data/reflective.data  --config_file cfg/yolov4-tiny-reflective.cfg --dont_show     
    ```
* 也可以直接采用Pytorch版本进行训练[pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4.git).

## 3.模型转换

* 将训练好的权重文件转换为ONNX格式，需要注意的是，pytorch-YOLOv4要求pytorch<=1.4.0：
    ```sh
    cd pytorch-YOLOv4/
    # python3 demo_darknet2onnx.py <cfgFile> <weightFile> <imageFile> <batchSize>
    ```
    ```ini
    参数说明：  
    <cfgFile> - 配置文件
    <weightFile> - darknet 训练好的模型
    <imageFile> - 测试图片
    <batchSize> - 推断时的 batch size，若为正整数(batch_size=1)则静态 batch size，若为负数或0(batch_size=-1)则为动态 batch size.
    ```
     ```sh
    # 动态
    python3 demo_darknet2onnx.py cfg/yolov4-tiny-reflective.cfg yolov4-tiny-reflective_best.weights 0a24f21c-b0b5-49e6-8794-c2f099bbd712.jpg  -1
    # 静态
    python3 demo_darknet2onnx.py cfg/yolov4-tiny-reflective.cfg yolov4-tiny-reflective_best.weights 0a24f21c-b0b5-49e6-8794-c2f099bbd712.jpg 1
    ```
* 如果设置(batch_size=1),你会得到一个名为‘yolov4_1_3_416_416_static.onnx’文件。  

## 4.部署

* 下面来到 Nvidia Xavier NX 开发板。
* 首先下载 Nvidia 官方出品的 yolov4_deepstream 项目：
    ```sh
    cd /opt/nvidia/deepstream/deepstream-5.0/sources/
    sudo git clone https://github.com/NVIDIA-AI-IOT/yolov4_deepstream.git
    ```
* 将模型的ONNX文件复制到yolov4_deepstream 项目路径下。
* 生成模型的 TensorRT Engine。

    如果是静态模型：
    ```sh
    trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
    # 例如：
    trtexec --onnx=yolov4_1_3_416_416_static.onnx  --explicitBatch  --workspace=2048 --saveEngine=yolov4_1_3_416_416_static.engine --fp16
    ```
    如果是动态模型：
    ```sh
    trtexec --onnx=<onnx_file> \
    --minShapes=input:<shape_of_min_batch> --optShapes=input:<shape_of_opt_batch> --maxShapes=input:<shape_of_max_batch> \
    --workspace=<size_in_megabytes> --saveEngine=<engine_file> --fp16  
    # 例如:
    trtexec --onnx=yolov4_-1_3_416_416_dynamic.onnx \
    --minShapes=input:1x3x416x416 --optShapes=input:4x3x416x416 --maxShapes=input:8x3x416x416 \
    --workspace=2048 --saveEngine=yolov4_-1_3_416_416_dynamic.engine --fp16
    ```
  有可能会报错找不到‘trtexec’，这时需要使用完整路径，使用find指令查找一下文件路径：
     ```sh
    sudo find / -name trtexec
    # 我的显示为
    # /usr/src/tensorrt/bin/trtexec
    /usr/src/tensorrt/bin/trtexec --onnx=yolov4_1_3_416_416_static.onnx  --explicitBatch  --workspace=2048 --saveEngine=yolov4_1_3_416_416_static.engine --fp16
    ```
  这部分会比较耗时，耐心等待即可。完成后会有以下输出，其中的‘mean: 4.52398 ms’即为平均推断时间，还是很快的。
    ```ini
    [10/20/2020-19:56:29] [I] Host Latency
    [10/20/2020-19:56:29] [I] min: 4.34442 ms (end to end 4.35388 ms)
    [10/20/2020-19:56:29] [I] max: 7.12915 ms (end to end 7.14398 ms)
    [10/20/2020-19:56:29] [I] mean: 4.52398 ms (end to end 4.53851 ms)
    [10/20/2020-19:56:29] [I] median: 4.49951 ms (end to end 4.51025 ms)
    [10/20/2020-19:56:29] [I] percentile: 5.46606 ms at 99% (end to end 6.10291 ms at 99%)
    [10/20/2020-19:56:29] [I] throughput: 0 qps
    [10/20/2020-19:56:29] [I] walltime: 3.00956 s
    [10/20/2020-19:56:29] [I] Enqueue Time
    [10/20/2020-19:56:29] [I] min: 2.3894 ms
    [10/20/2020-19:56:29] [I] max: 4.89722 ms
    [10/20/2020-19:56:29] [I] median: 2.64331 ms
    [10/20/2020-19:56:29] [I] GPU Compute
    [10/20/2020-19:56:29] [I] min: 4.2439 ms
    [10/20/2020-19:56:29] [I] max: 7.02356 ms
    [10/20/2020-19:56:29] [I] mean: 4.42492 ms
    [10/20/2020-19:56:29] [I] median: 4.40088 ms
    [10/20/2020-19:56:29] [I] percentile: 5.36792 ms at 99%
    [10/20/2020-19:56:29] [I] total compute time: 2.93372 s
    &&&& PASSED TensorRT.trtexec # /usr/src/tensorrt/bin/trtexec --onnx=yolov4_1_3_416_416_static.onnx --explicitBatch=1 --workspace=2048 --saveEngine=yolov4_1_3_416_416_static.engine --fp16
    ```
* 将 TensorRT Engine 部署至 DeepStream：
    ```sh
    cd yolov4_deepstream/deepstream_yolov4/nvdsinfer_custom_impl_Yolo/
    ```
    * 修改nvdsparsebbox_Yolo.cpp’文件中NUM_CLASSES_YOLO参数为要检测的类别数，此项目为2。
    ```sh
    static const int NUM_CLASSES_YOLO = 2;
    ```
    * 编译‘nvdsparsebbox_Yolo.cpp’文件，将得到一个名为‘libnvdsinfer_custom_impl_Yolo.so’的动态链接库。
    ```sh
    export CUDA_VER=10.2  //CUDA版本号
    make
    ```
* 将之前生成的‘yolov4_1_3_416_416_static.engine’文件复制到‘/opt/nvidia/deepstream/deepstream-5.0/sources/yolov4_deepstream‘
* 编辑 config_infer_primary_yoloV4.txt 和 deepstream_app_config_yoloV4.txt

    config_infer_primary_yoloV4.txt 中一些需要修改或关注的参数：
    
    ```ini
    [property]
    gpu-id=0  
    net-scale-factor=0.0039215697906911373
    #0=RGB, 1=BGR
    model-color-format=0
    # 需要修改！模型 engine 文件路径
    model-engine-file=yolov4_1_3_418_418_static.engine  
    # 需要修改！显示的类别名称， 内容同 reflective.names 文件
    labelfile-path=labels.txt  
    # 需要修改！等于视频流的输入个数
    batch-size=1  
    ## 0=FP32, 1=INT8, 2=FP16 模型推理精度
    network-mode=2
    # 需要修改！类别个数，等于labelfile中类别个数
    num-detected-classes=2  
    gie-unique-id=1
    network-type=0
    is-classifier=0
    ## 0=Group Rectangles, 1=DBSCAN, 2=NMS, 3= DBSCAN+NMS Hybrid, 4 = None(No clustering)
    cluster-mode=2
    maintain-aspect-ratio=1
    parse-bbox-func-name=NvDsInferParseCustomYoloV4
    custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
    #scaling-filter=0
    #scaling-compute-hw=0
    
    [class-attrs-all]
    # iou 阈值
    nms-iou-threshold=0.6  
    pre-cluster-threshold=0.4
    ```
  
    deepstream_app_config_yoloV4.txt 中一些需要修改或关注的参数：
    ```ini
    [tiled-display]
    enable=0
    rows=1
    columns=1
    width=418
    height=418
    
    [source0]
    # 需要修改！一个source就是一个视频源，增加视频源可以在后面添加[source1][source2]..
    enable=1
    #Type - 1=CameraV4L2 2=URI 3=MultiURI 
    type=3
    # 需要修改！视频地址
    uri=file:/opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_yolov4/test.mp4
    num-sources=1
    
    [sink0]
    enable=1
    #Type - 1=FakeSink 2=EglSink 3=File
    type=3
    sync=0
    source-id=0
    gpu-id=0
    nvbuf-memory-type=0
    #1=mp4 2=mkv
    container=1
    #1=h264 2=h265
    codec=1
    # 需要修改！视频推断后的保存路径
    output-file=yolov4.mp4
    
    [streammux]
    gpu-id=0
    ##Boolean property to inform muxer that sources are live
    live-source=0
    batch-size=1
    ##time out in usec, to wait after the first buffer is available
    ##to push the batch even if the complete batch is not formed
    batched-push-timeout=40000
    ## Set muxer output width and height
    width=418
    height=418

    
    [primary-gie]
    enable=1
    gpu-id=0
    # 以下三项的设置与config_infer_primary_yoloV4.txt文件保持一致
    # 需要修改！模型engine文件路径
    model-engine-file=yolov4_1_3_418_418_static.engine  
    # 需要修改！label文件路径
    labelfile-path=labels.txt  
    # 需要修改！
    batch-size=1
    ```

## 5.运行测试

* 运行下面的命令，将会生成结果文件‘yolov4.mp4’。
    ```sh
    deepstream-app -c deepstream_app_config_yoloV4.txt
    ```
    App run successful.大功告成！
    Fps大概172左右。
    
## 6.结果

* Deepstream部署结果原本Darknet结果对比看[这里](https://www.bilibili.com/video/BV18K4y1Z7LZ)。



