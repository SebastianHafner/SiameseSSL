# spacenet7

Ideas to implement:
* Normalize input for resnet using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
* Thaw pretrained weights at the beginning of the training for resnet
* LR schedule (even ADAM is only optimal under a lr regime of ~√n)
* Replace transpose convs unet https://distill.pub/2016/deconv-checkerboard/
* Lovász-softmax loss
* Maybe some more data augmentation?
