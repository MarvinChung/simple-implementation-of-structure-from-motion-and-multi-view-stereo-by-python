# simple implementation of structure from motion and multi view stereo by python

structure from motion sparse point cloud to dense point cloud 

dataset is available at:
http://vision.middlebury.edu/mview/data/

## Usage:
```
python main.py -img_p dinoRing/ -par_p dinoRing/dinoR_par.txt -t png -scale 10
```

| command | Function | 
| -------- | -------- |
| -img_p     | path to images |
| -par_p     | path to intrinsic and extrinsic parameters |
| -t     | type of the image sequences |
| -scale     | scale for the visualization |
| --debug | debug mode |
| --nonSequence | input is not image seqeunce (not yet support)|

## SFM
Sparse point cloud of structure from motion

#### without bundle adjustment
![](https://i.imgur.com/5pNYI4y.png)

#### with bundle adjustment
![](https://i.imgur.com/friMHey.png)

### MVS
After 100000 iteration

![](https://i.imgur.com/HXZyli5.png)




