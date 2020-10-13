FROM ##DevImage## AS dev
LABEL maintainer "Samuel Antao <samuel.antao@ibm.com>"

#
# install GCC
#
RUN set -eux ; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    gcc-8 \
    g++-8 \
    gfortran-8 \
    ; \
  rm -rf /var/lib/apt/lists/* ; \
  update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-8 \
    --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-8 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-8
 
#    
# install OpenMPI
#
RUN set -eux ; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    openmpi-bin \
    ; \
  rm -rf /var/lib/apt/lists/*

ENV OMPI_CC gcc
ENV OMPI_CXX g++
ENV OMPI_F77 gfortran
ENV OMPI_FC gfortran


#    
# Point to CUDA libs
#
ENV CUDADIR=/usr/local/cuda 
  
#    
# install java
#
RUN set -eux ; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
     openjdk-8-jdk-headless \
    ; \
  rm -rf /var/lib/apt/lists/*
  
#    
# install extra utilities
#
RUN set -eux ; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    curl \
    less \
    nano \
    cmake \
    git \
    libjpeg9-dev \
    zip \
    unzip \
    ; \
  rm -rf /var/lib/apt/lists/*
   
#
# Path to install all packages including anaconda.
#
ENV InstallPath /opt/user

RUN set -eux ; \
  groupadd -r user --gid=1000  ; \
  useradd -m -r -g user --uid=1000 --home-dir=$InstallPath --shell=/bin/bash user

USER user:user
WORKDIR $InstallPath 

#
# Install miniconda
#
RUN set -eux ; \
  cd $InstallPath ; \
  MinicondaInstallFile=Miniconda3-py37_4.8.3-Linux-$(uname -m).sh ; \
  curl -LO https://repo.continuum.io/miniconda/$MinicondaInstallFile ; \
  if [ $(uname -m) = ppc64le ] ; then \
    echo "bcd33ea9240e2720ec004af43194c3fe6d39581e4a957a26621e00c232ca5ca1  $MinicondaInstallFile" | sha256sum -c - ; \
  else \
    echo "bb2e3cedd2e78a8bb6872ab3ab5b1266a90f8c7004a22d8dc2ea5effeb6a439a  $MinicondaInstallFile" | sha256sum -c - ; \
  fi ; \
  bash $MinicondaInstallFile -f -b -p $InstallPath/miniconda3 ; \
  rm -rf $MinicondaInstallFile
  
#
# Install antlr4 and test
#
RUN set -eux ; \
  mkdir $InstallPath/antlr4 ; \
  cd $InstallPath/antlr4 ; \
  curl -LO https://www.antlr.org/download/antlr-4.7.2-complete.jar ; \
  echo "6852386d7975eff29171dae002cc223251510d35f291ae277948f381a7b380b4  antlr-4.7.2-complete.jar" | sha256sum -c - ; \
  echo "#!/bin/bash" >> antlr4 ; \
  echo "export CLASSPATH=\".:$(pwd)/antlr-4.7.2-complete.jar:\$CLASSPATH\"" >> antlr4 ; \
  echo "java -Xmx8192M -cp \"$(pwd)/antlr-4.7.2-complete.jar:\$CLASSPATH\" org.antlr.v4.Tool \$@" >> antlr4 ; \
  chmod +x antlr4
  
ENV PATH $InstallPath/antlr4:$PATH

RUN set -eux ; \
  antlr4
  
#
# Install MAGMA
#
RUN set -eux ; \
  cd $InstallPath ; \
  . miniconda3/bin/activate ; \
  conda create -n magma python=3.7 ; \
  conda activate magma ; \
  conda install openblas

ENV OPENBLASDIR $InstallPath/miniconda3/envs/magma
  
RUN set -eux ; \
  cd $InstallPath ; \
  . miniconda3/bin/activate ; \
  conda activate magma ; \
  mkdir -p $InstallPath/magma/install ; \
  cd $InstallPath/magma/install ; \
  git clone https://bitbucket.org/icl/magma  ; \
  cd magma ; \
  git checkout -b v2.5.3 v2.5.3 ; \
  sed 's/#GPU_TARGET.*/GPU_TARGET ?= Kepler Pascal Volta/g' \
    make.inc-examples/make.inc.openblas > make.inc ; \
  make lib -j ; \
  make sparse-lib -j ; \
  make install prefix=$InstallPath/magma ; \
  cd ; rm -rf $InstallPath/magma/install
    
ENV MAGMA_HOME $InstallPath/magma

#
# Install fxdiv
#
RUN set -eux ; \
  mkdir -p $InstallPath/fxdiv/install/obj ; \
  cd $InstallPath/fxdiv/install ; \
  git clone https://github.com/Maratyszcza/FXdiv ; \
  cd obj ; \
  cmake -DCMAKE_INSTALL_PREFIX=$InstallPath/fxdiv ../FXdiv ; \
  make -j ; \
  make -j install ; \
  rm -rf $InstallPath/fxdiv/install

ENV C_INCLUDE_PATH $InstallPath/fxdiv/include
ENV CPLUS_INCLUDE_PATH $InstallPath/fxdiv/include
ENV LIBRARY_PATH $InstallPath/fxdiv/lib64

#
# Install pytorch
#
COPY pytorch-v1.5.0.patch $InstallPath
RUN set -eux ; \
  cd $InstallPath ; \
  . miniconda3/bin/activate ; \
  conda create -n pytorch python=3.7
  
RUN set -eux ; \
  cd $InstallPath ; \
  . miniconda3/bin/activate ; \
  conda activate pytorch ; \
  git clone --single-branch -b v1.6.0 --recursive https://github.com/pytorch/pytorch pytorch-v1.6.0 ; \
  cd pytorch*/ ; \
#  git apply < $InstallPath/pytorch-v1.5.0.patch ; \
  git submodule sync ; \
  git submodule update --init --recursive ; \
  conda install numpy ninja pyyaml setuptools cffi openblas ; \
  if [ $(uname -m) = x86_64 ] ; then \
    conda install mkl mkl-include ; \
  fi ; \
  export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} ; \
  export TORCH_CUDA_ARCH_LIST="3.7;6.0;7.0" ; \
  python setup.py install ; \
  cd ; \
  rm -rf $InstallPath/pytorch-v1.5.0.patch
  
#
# Install pytorch-vision and test
#
RUN set -eux ; \
  cd $InstallPath ; \
  . miniconda3/bin/activate ; \
  conda activate pytorch ; \
  git clone --single-branch -b v0.7.0 --recursive https://github.com/pytorch/vision pytorch-vision-v0.7.0 ; \
  cd pytorch-vision*/ ; \
  export FORCE_CUDA=1 ; \
  export TORCH_CUDA_ARCH_LIST="3.7;6.0;7.0" ; \
  python setup.py install ; \
  cd ; \
  true
  
#
# Install deepppl dependencies
#
RUN set -eux ; \
  cd $InstallPath ; \
  . miniconda3/bin/activate ; \
  conda create --name deepppl --clone pytorch
  
# Install bazel needed to install JAX
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; \
  mkdir -p $InstallPath/bazel/install ; \
  cd $InstallPath/bazel/install ; \
  curl -LO https://github.com/bazelbuild/bazel/releases/download/3.5.0/bazel-3.5.0-dist.zip ; \
  echo "334429059cf82e222ca8a9d9dbbd26f8e1eb308613463c2b8655dd4201b127ec  bazel-3.5.0-dist.zip" | sha256sum -c - ; \
  unzip bazel-3.5.0-dist.zip ; \
  env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh ; \
  cd ; \
  mv $InstallPath/bazel/install/output/bazel $InstallPath/bazel ; \
  rm -rf $InstallPath/bazel/install

ENV PATH=$InstallPath/bazel:$PATH

# Install required python packages for JAX
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; \
  conda install scipy cython six

# Install JAX
COPY jax-v0.1.51.patch $InstallPath
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; \
  git clone --single-branch -b jaxlib-v0.1.51 https://github.com/google/jax ; \
  cd jax ; \
  git apply < $InstallPath/jax-v0.1.51.patch ; \
  python build/build.py \
    --enable_march_native false \
    --enable_mkl_dnn false \
    --cudnn_path /usr \
    --enable_cuda true \
    --cuda_compute_capabilities 3.5,3.7,6.0,7.0 ; \
  pip install -e build ; \
  pip install -e . ; \
  cd $InstallPath ; \
  true 
  
# Already installed dependencies:
#   numpy (pytorch dependency)
#   pytorch (from source)
#   pytorch-vision (from source)
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; pip3 install \
    'antlr4-python3-runtime==4.7.2'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; conda install \
    'astor==0.7.1'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; pip install \
    'astpretty==1.2.1'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; pip install \
    'observations==0.1.4'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; pip install \
    'pyro-ppl==1.3.1'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; pip install \
    'numpyro==0.3.0'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; conda install \
    'pytest>=3.6.1'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; conda install \
    'requests>=2.20.0'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; pip install \
    'ipdb>=0.11'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; conda install \
    'matplotlib>=2.2.2'
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; conda install \
    'pandas>=0.25.1'
  
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; \
  git clone https://github.com/deepppl/deepppl ; \
  cd deepppl/deepppl ; \
  make -j ; \
  pip install ..
  
# Remove pytorch environment as it was cloned to deepppl one.
RUN set -eux ; \
  cd $InstallPath ; \
  . miniconda3/bin/activate ; \
  conda env remove -y -n pytorch  
  
##########################################################################################
FROM ##BaseImage##
LABEL maintainer "Samuel Antao <samuel.antao@ibm.com>"

#
# install system dependencies
#
RUN set -eux ; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    gcc-8 \
    g++-8 \
    gfortran-8 \
    openmpi-bin \
    openjdk-8-jre-headless \
    curl \
    less \
    nano \
    libjpeg9 \
    zip \
    unzip \
    git \
    make \
    curl \
    patch \
    m4 \
    xz-utils \
    libcap-dev \
    ; \
  rm -rf /var/lib/apt/lists/* ; \
  update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-8 \
    --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-8 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-8
 

ENV OMPI_CC gcc
ENV OMPI_CXX g++
ENV OMPI_F77 gfortran
ENV OMPI_FC gfortran

#    
# Point to CUDA libs
#
ENV CUDADIR=/usr/local/cuda 
  
#
# Path to install all packages including anaconda.
#
ENV InstallPath /opt/user

RUN set -eux ; \
  groupadd -r user --gid=1000  ; \
  useradd -m -r -g user --uid=1000 --home-dir=$InstallPath --shell=/bin/bash user

USER user:user
WORKDIR $InstallPath 

# Copy the relevant bits - pytorch is installed in miniconda so not need to copy it over 
# separately.
COPY --chown=user:user --from=dev $InstallPath/antlr4 $InstallPath/antlr4

COPY --chown=user:user --from=dev $InstallPath/deepppl $InstallPath/deepppl

COPY --chown=user:user --from=dev $InstallPath/fxdiv $InstallPath/fxdiv

RUN set -eux ; \
  mkdir $InstallPath/jax
COPY --chown=user:user --from=dev $InstallPath/jax/build $InstallPath/jax/build
COPY --chown=user:user --from=dev $InstallPath/jax/jaxlib $InstallPath/jax/jaxlib
COPY --chown=user:user --from=dev $InstallPath/jax/jax $InstallPath/jax/jax
COPY --chown=user:user --from=dev $InstallPath/jax/jax.egg-info $InstallPath/jax/jax.egg-info

COPY --chown=user:user --from=dev $InstallPath/magma $InstallPath/magma

COPY --chown=user:user --from=dev $InstallPath/miniconda3 $InstallPath/miniconda3


# Install notebook in miniconda
RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; \
  conda install jupyter 

# Install OCaml dependencies.
RUN set -eux ; \
  mkdir -p $InstallPath/ocaml/install ; \
  cd $InstallPath/ocaml/install ; \
  git clone --single-branch -b 4.07.1 https://github.com/ocaml/ocaml  ; \
  cd ocaml ; \
  ./configure --prefix $InstallPath/ocaml ; \
  make world.opt -j ; \
  make install -j
ENV PATH $InstallPath/ocaml/bin:$PATH

RUN set -eux ; \
  mkdir -p $InstallPath/opam/install ; \
  cd $InstallPath/opam/install ; \
  git clone --single-branch -b 2.0 https://github.com/ocaml/opam  ; \
  cd opam ; \
  ./configure --prefix=$InstallPath/opam ; \
  make lib-ext ; \
  make; \
  make install
ENV PATH $InstallPath/opam/bin:$PATH

RUN set -eux ; \
  opam init -y --disable-sandboxing; \
  eval $(opam config env)

RUN set -eux ; \
  eval $(opam config env) ; \
#  opam install -y core_kernel fmt menhir menhirLib ppx_deriving ppx_jane re yojson ; \
  opam install -y dune.1.8.2 core_kernel.v0.11.1 menhir.20181113 ppx_deriving.4.2.1 fmt.0.8.6 yojson

RUN set -eux ; \
  eval $(opam config env) ; \
  cd  $InstallPath ; \
  git clone --single-branch -b deep https://github.com/mandel/stanc3 ; \
  cd stanc3 ; \
  dune subst ; \
  dune build @install --profile static 
ENV PATH $InstallPath/stanc3/_build/install/default/bin:$PATH  
RUN stanc --help

# Add the relevant folders in stanc3 to the conda environment. This provide the runtime modules
# used by the generated numpyro files.
RUN set -eux ; \
  echo "/opt/user/stanc3" > $InstallPath/miniconda3/envs/deepppl/lib/python3.7/site-packages/zzz-stanc3.pth

RUN set -eux ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; \
  pip install torchdiffeq 

# Copy example notebook
COPY --chown=user:user notebook $InstallPath/notebook

EXPOSE 8888  

WORKDIR $InstallPath/notebook
CMD set -eux ; \
  eval $(opam config env) ; \
  cd $InstallPath ; . miniconda3/bin/activate ; conda activate deepppl ; \
  nvidia-smi ; \
  jupyter-notebook --ip=0.0.0.0 --port=8888
  
  
# RUN set -eux ; \
#   . /venvs/flow/bin/activate; \
#   python -c "import sys; [print(x) for x in sys.path if '/venvs/' in x]" \
#     > /opt/conda/lib/python3.7/site-packages/zzz-foc-mvp2.pth ; \
#   deactivate

