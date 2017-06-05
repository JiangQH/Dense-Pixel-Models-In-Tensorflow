# Dense-Pixel-Models-In-Tensorflow  

This is a repo contains tensorflow implementations of current state-of-art semantic segmentation nets, and expand them to dense pixel tasks including surface normal and depth estimation from monocular image.  

Some of them are the latest Cvpr2017 receiving papers and some of them have their influence in the dense pixel tasks.  

Mainly it contains following nets   
- [ ]  [FCN](https://arxiv.org/abs/1411.4038) 
- [x] [Segnet](https://arxiv.org/abs/1511.00561)     
- [x] [Enet](https://arxiv.org/abs/1606.02147)    
- [ ] [DUC](https://arxiv.org/abs/1702.08502)    
- [ ] [DeepLab](https://arxiv.org/abs/1606.00915)    

Further More I will try to deploy the google MobileNets([MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)) to some of the above nets to see whether it can bring speed up without lossing much accuracy.  

## Packages  

Currently only part of them are completed, as    
 * [common](./common) this folder contains  some common tool for all nets used to do the training job.    
 * [enet](./enet) contains the full implementation of the [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147) in tensorflow.  
 *  [segnet](./segnet) contains the full implementation of the [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561) in tensorflow. 
 

##  Note & Basic Usage

Note that this repo uses [PyYam](http://pyyaml.org/)  config file to config the project. Make sure that it has been installed if you want to use the config file.   

### Training    

```
python train.py -c ./conf.yml
```

### Deploying   

```
python deploy.py -i[input_dir] -o[output_dir] -m[model_file]
```

For more detail, refer to the code. 

##  License  

Note this project is released for personal and research use only. For a commercial license please contact the original paper authors.   


