## Demo
Please follow the produce in ([`Pose2Mesh`](https://github.com/hongsukchoi/Pose2Mesh_RELEASE)) for the visualization. 
The demo ckp (slightly different from the provided ckp which is the old version, but if you re-train our model, you should get this demo ckp) is provided ([`here`](https://drive.google.com/file/d/1PdOymnqadk2Rzjgf2He660mH_uHYlluv/view?usp=share_link).)

Replace the demo ckp, then run 
`python demo/run.py --gpu 0 --input_pose demo/h36m_joint_input.npy --joint_set human36` if you want to run on gpu 0.










## Additionally
We provide demo codes for end-to-end inference here. But sometimes it may have the scaling issue. 

Please note here we implement this demo based on another repository ([`I2LMeshNet`](https://github.com/mks0601/I2L-MeshNet_RELEASE)) to achieve end-to-end inference. 

I prepared all files needed, just unzip after downloading from ([`here`](https://drive.google.com/file/d/14lEwP8UyNn0b2Zh_CSslAwLhhgKGAZ7Z/view?usp=sharing).)

We integrate ([`TransPose`](https://github.com/yangsenius/TransPose)) as 2D pose detector with our proposed GTRS for human mesh recovery. 

All you need to do is prepare an [Anaconda](https://www.anaconda.com/) virtual environment (we prepare a requirements.txt), then

* Prepare `input.jpg` at `demo` folder.

* Go to `demo` folder and **edit** `bbox` (following I2LMeshNet, you need manually enter the bbx info for each image) in `demo/demo.py`, line 84.

* run `python demo.py --gpu 0 --stage param --test_epoch 93` if you want to run on gpu 0.

* You can see `output_mesh_param.jpg`, `rendered_mesh_param.jpg`, `rendered_mesh_param_b.jpg`, and `output_mesh_param.obj`.


