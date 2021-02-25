## Installing rllab envs

### Prerequisites
install modules:
```
conda install --file rllab_req.txt
pip install -r rllab_req_pip.txt
```
### Add git repos
```
$ git submodule add -f https://github.com/rll/rllab.git softlearning/environments/rllab
$ cd softlearning/environments/rllab/
$ git submodule add -f https://github.com/jachiam/cpo sandbox/cpo
```

### Install mujoco for rllab: rllab requires mujoco 1.31.

Download mujoco131:
```
$ wget https://www.roboti.us/download/mjpro131_linux.zip
$ ./scripts/setup_mujoco.sh
	<Enter mjpro131_linx.zip for mujoco path>
	<Enter path to mujoco license file>
```

Because rllab is installed as a submodule, imports are a bit chaotic. For these to work please add
```
$ echo "">__init__.py
```

### Fix a strange error
Inside the rllab repo navigate to
```
$ cd rllab/rllab/envs/
$ vim proxy_env.py
```

and comment out the line
```
# Serializable.quick_init(self, locals())
```

### File
Inside the CMPBO repo you will need the following files:
- Rllabadapter
- static functions
- edit softlearning/environments/utils.py to acknowledge rllabadapter
- a config file
