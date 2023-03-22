# A Lightweight Graph Transformer Network for Human Mesh Reconstruction from 2D Human Pose
The project is an official implementation of our paper [A Lightweight Graph Transformer Network for Human Mesh Reconstruction from 2D Human Pose](https://arxiv.org/pdf/2111.12696).

## Installation
Check [INSTALL.md](docs/INSTALL.md) for installation instructions.

## Quick demo
We provide demo codes to run end-to-end inference on the test images. Please check [DEMO.md](docs/DEMO.md) for details.

## Datasets Download
Please download the required datasets following [DOWNLOAD.md](docs/DOWNLOAD.md). 

## Experiment

The `experiment` directory should contain following folders. 
```
${ROOT}  
|-- experiment  
|   |-- pam_h36m
|   |   |-- best.pth.tar
|   |-- gtrs_h36m
|   |   |-- final.pth.tar
|   |-- pam_3dpw
|   |   |-- best.pth.tar
|   |-- pam_3dpw
|   |   |-- final.pth.tar
```

### Pretrained model weights
The pretrained model weights can be download from [here](https://drive.google.com/file/d/1fIzs5zaEcqzjOmggYXlD8lcO4JAJPb4o/view?usp=sharing) to a corresponding directory.

### Testing

You can choose the config file in `${ROOT}/asset/yaml/` to evaluate the corresponding experiment by running: 

```
python main/test.py --gpu 0,1, --cfg ./asset/yaml/gtrs_{input joint set}_test_{dataset name}.yml
```

For example, if you want to test the results on Human3.6M dataset, you can run:

```
python main/test.py --gpu 0,1, --cfg ./asset/yaml/gtrs_human36J_test_human36.yml
```

### Training
We provide the pretrain models of PAM module. To train GTRS, you can run: 

```
python main/train.py --gpu 0,1, --cfg ./asset/yaml/gtrs_{input joint set}_train_{dataset name}.yml
```

For example, if you want to train on Human3.6M dataset, you can run:

```
python main/train.py --gpu 0,1, --cfg ./asset/yaml/gtrs_human36J_train_human36.yml
```

Also if you prefer training from the scratch, you should pre-train PAM module first by running:

```
python main/train.py --gpu 0,1, --cfg ./asset/yaml/pam_{input joint set}_train_{dataset name}.yml
```

## Citations
If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{zheng2022lightweight,
  title={A lightweight graph transformer network for human mesh reconstruction from 2d human pose},
  author={Zheng, Ce and Mendieta, Matias and Wang, Pu and Lu, Aidong and Chen, Chen},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={5496--5507},
  year={2022}
}
```


## License

Our research code is released under the MIT license. See [LICENSE](LICENSE) for details. 



## Acknowledgments

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[hongsukchoi/Pose2Mesh_RELEASE](https://github.com/hongsukchoi/Pose2Mesh_RELEASE) 

[mks0601/I2L-MeshNet_RELEASE](https://github.com/mks0601/I2L-MeshNet_RELEASE) 

[yangsenius/TransPose](https://github.com/yangsenius/TransPose) 



