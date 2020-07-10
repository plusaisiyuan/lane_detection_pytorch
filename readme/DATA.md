# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation and training, you will need to setup dataset.


### CULane
- Download the images and annotation files (88880 for training set, 9675 for validation set, and 34680 for test set) from 
    [Google Drive](https://drive.google.com/open?id=1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu)
    or
    [Baidu Cloud](https://pan.baidu.com/s/1KUtzC24cH20n6BtU5D0oyw).
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${LaneNet_ROOT}
  |-- data
  `-- |-- CULane
      `-- |-- annotations
          |   |-- instances_train2017.json
          |   |-- instances_val2017.json
          |   |-- person_keypoints_train2017.json
          |   |-- person_keypoints_val2017.json
          |   |-- image_info_test-dev2017.json
          |---|-- train_gt.txt
          |---|-- val_gt.txt
          `---|-- test_img.txt
  ~~~
  

### L4E
- Download the images and annotation files (88880 for training set, 9675 for validation set, and 34680 for test set) from 
    [Google Drive](https://drive.google.com/open?id=1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu)
    or
    [Baidu Cloud](https://pan.baidu.com/s/1KUtzC24cH20n6BtU5D0oyw).
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${LaneNet_ROOT}
  |-- data
  `-- |-- L4E
      `-- |-- annotations
          |---|-- train_gt.txt
          |---|-- val_gt.txt
          `---|-- test_img.txt
  ~~~


