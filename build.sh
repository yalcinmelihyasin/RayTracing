if [ -d "build" ]; then
  rm -rf build
fi

mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" ../
