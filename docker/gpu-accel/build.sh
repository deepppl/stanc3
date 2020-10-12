#!/bin/bash

tag="$1"

if [ -z "$tag" ] ; then
  echo "Run as './build.sh <image tag>'"
  exit 1
fi

# We are using the devel images for both development and final images as the nvidia tools
# are need for JIT. If we don't plan to JIT images, we can use the runtime base image
# to keep images sizes down.

if [ $(uname -m) = 'ppc64le' ] ; then
  cat context/Dockerfile | \
  sed 's;##DevImage##;nvidia/cuda-ppc64le:10.2-cudnn7-devel-ubuntu18.04;g' | \
  sed 's;##BaseImage##;nvidia/cuda-ppc64le:10.2-cudnn7-devel-ubuntu18.04;g' \
  > context/.Dockerfile 
else
  cat context/Dockerfile | \
  sed 's;##DevImage##;nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04;g' | \
  sed 's;##BaseImage##;nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04;g' \
  > context/.Dockerfile 
fi

docker build -t "$tag" -f context/.Dockerfile context
