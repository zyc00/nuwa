# If you do not need pixsfm (most users do not need this)
# skip this file and modify the final lines of requirements.txt

# ceres-solver-2.1.0
conda deactivate
sudo apt-get install -y git cmake build-essential libgoogle-glog-dev \
    libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
mkdir ceres && cd ceres || exit
wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
tar zxf ceres-solver-2.1.0.tar.gz
mkdir ceres-bin && cd ceres-bin || exit
cmake ../ceres-solver-2.1.0 && make -j32
#make test
sudo make install


# colmap @fcd6493ead138c314f28b2cb07e27ab0bc891b9d
conda deactivate
sudo apt-get install -y gcc-10 g++-10 ninja-build libboost-program-options-dev \
    libboost-filesystem-dev libboost-graph-dev libboost-system-dev \
    libflann-dev libfreeimage-dev libmetis-dev libgtest-dev \
    libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev
git clone https://github.com/colmap/colmap.git
cd colmap && git checkout fcd6493
mkdir build && cd build || exit
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native
ninja && sudo ninja install


# pip dependencies
pip install git+https://github.com/colmap/pycolmap.git@v0.4.0
pip install git+https://github.com/cvg/pyceres@v1.0
pip install git+https://github.com/cvg/Hierarchical-Localization.git@988dd3a68b4d7def44c2870c9c3fd2069b037f64
pip install git+https://github.com/cvg/pixel-perfect-sfm.git@40f7c1339328b2a0c7cf71f76623fb848e0c0357
