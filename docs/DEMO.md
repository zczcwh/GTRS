## Demo
We provide demo codes for end-to-end inference here. 

Please note here we implement this demo based on another repository ([`I2LMeshNet`](https://github.com/mks0601/I2L-MeshNet_RELEASE)) to achieve end-to-end inference. 

I prepared all files needed, just unzip after downloading from ([`here`](https://drive.google.com/file/d/14lEwP8UyNn0b2Zh_CSslAwLhhgKGAZ7Z/view?usp=sharing).)

We integrate ([`TransPose`](https://github.com/yangsenius/TransPose)) as 2D pose detector with our proposed GTRS for human mesh recovery. 

All you need to do is prepare an [Anaconda](https://www.anaconda.com/) virtual environment (we prepare a requirements.txt), then

* Prepare `input.jpg` at `demo` folder.

* Go to `demo` folder and **edit** `bbox` (following I2LMeshNet, you need manually enter the bbx info for each image) in `demo/demo.py`, line 84.

* run `python demo.py --gpu 0 --stage param --test_epoch 93` if you want to run on gpu 0.

* You can see `output_mesh_param.jpg`, `rendered_mesh_param.jpg`, `rendered_mesh_param_b.jpg`, and `output_mesh_param.obj`.


