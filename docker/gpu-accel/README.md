# Docker image for NVIDIA GPU runs.

This folder implements a docker image that runs stanc3 with the required dependencies
to enable GPU runs through numpyro on the resulting files. The default command is a 
jupyter notebook for which an example is provided to ilustrate how it works.

To build this image execute in your command line:
```
./build <tag name>
```

Running the image depends on how your container platform support GPUs. Assuming no
special support exist you could run the image in a machine with CUDA and NVIDIA drivers
with the following recipe - here assuming we have 4 GPUs:

```
mkdir -p drivers/bin
for i in \
 nvidia-cuda-mps-control \
 nvidia-cuda-mps-server \
 nvidia-debugdump \
 nvidia-persistenced \
 nvidia-smi \
 ; do cp -f $(which $i) drivers/bin ; done

mkdir -p drivers/lib64
ln -s ../lib64 drivers/lib

cp -f \
  /lib64/libnvidia* \
  /lib64/libcuda* \
  drivers/lib64 

docker run \
  -it --rm \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidia1:/dev/nvidia1 \
  --device /dev/nvidia2:/dev/nvidia2 \
  --device /dev/nvidia3:/dev/nvidia3 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  -v $(pwd)/drivers:/usr/local/nvidia:ro \
  -p 0.0.0.0:8888:8888 \
  <selected tag>
```

This will spin a jupyter notebook that is exposed at port 8888.
