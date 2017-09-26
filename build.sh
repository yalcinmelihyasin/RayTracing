if [ -d "build" ]; then
  rm -rf build
fi

#export CUDA
#export CUDA_PATH=$CUDA_PATH

mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" ../
