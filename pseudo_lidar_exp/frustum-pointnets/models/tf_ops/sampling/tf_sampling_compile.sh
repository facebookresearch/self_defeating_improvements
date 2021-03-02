##/bin/bash
##/usr/local/cuda-8.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#/private/home/ruihan/local/cuda-10.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#
## TF1.2
##g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#
## TF1.4
##/private/home/ruihan/anaconda3/envs/fpn/bin/python
#TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#
#echo $TF_LIB
#
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /private/home/ruihan/anaconda3/envs/fpn/lib/python3.7/site-packages/tensorflow/include -I /private/home/ruihan/local/cuda-10.0/include -I /private/home/ruihan/anaconda3/envs/fpn/lib/python3.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /private/home/ruihan/local/cuda-10.0/lib64/ -L/private/home/ruihan/anaconda3/envs/fpn/lib/python3.7/site-packages/tensorflow -ltensorflow_framework -O2 -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework #-D_GLIBCXX_USE_CXX11_ABI=0


#/bin/bash
CUDA_ROOT=/private/home/ruihan/local/cuda-10.0
TF_ROOT=/private/home/ruihan/anaconda3/envs/fpn2/lib/python3.8/site-packages/tensorflow
$CUDA_ROOT/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#TF 1.8
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0