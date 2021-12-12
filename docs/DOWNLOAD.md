We use exectly same datasets as [`pose2mesh`](https://github.com/hongsukchoi/Pose2Mesh_RELEASE). Please following the instructions to perpare datasets and files (all download links are provided in their repository).


### Data

The `data` directory structure should follow the below hierarchy.
```
${ROOT}  
|-- data  
|   |-- Human36M  
|   |   |-- images  
|   |   |-- annotations   
|   |   |-- J_regressor_h36m_correct.npy
|   |   |-- absnet_output_on_testset.json 
|   |-- MuCo  
|   |   |-- data  
|   |   |   |-- augmented_set  
|   |   |   |-- unaugmented_set  
|   |   |   |-- MuCo-3DHP.json
|   |   |   |-- smpl_param.json
|   |-- COCO  
|   |   |-- images  
|   |   |   |-- train2017  
|   |   |   |-- val2017  
|   |   |-- annotations  
|   |   |-- J_regressor_coco.npy
|   |   |-- hrnet_output_on_valset.json
|   |-- PW3D 
|   |   |-- data
|   |   |   |-- 3DPW_latest_train.json
|   |   |   |-- 3DPW_latest_validation.json
|   |   |   |-- darkpose_3dpw_testset_output.json
|   |   |   |-- darkpose_3dpw_validationset_output.json
|   |   |-- imageFiles
|   |-- AMASS
|   |   |-- data
|   |   |   |-- cmu
|   |-- SURREAL
|   |   |-- data
|   |   |   |-- train.json
|   |   |   |-- val.json
|   |   |   |-- hrnet_output_on_testset.json
|   |   |   |-- simple_output_on_testset.json
|   |   |-- images
|   |   |   |-- train
|   |   |   |-- test
|   |   |   |-- val
```

### Pytorch SMPL

Please following instructions in [`pose2mesh`](https://github.com/hongsukchoi/Pose2Mesh_RELEASE) for SMPL files.
