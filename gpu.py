import tensorflow as tf
from tensorflow.python.client import device_lib

print("🔹 TensorFlow version:", tf.__version__)

# Check basic GPU availability
gpus = tf.config.list_physical_devices('GPU')
print("🔹 Available GPUs:", gpus)

print("\n🔹 Full device list:")
for d in device_lib.list_local_devices():
    print(d)

# Optional: Check CUDA and cuDNN build versions
from tensorflow.python.platform import build_info as tf_build_info
print("\n🔹 CUDA version:", tf_build_info.build_info.get('cuda_version'))
print("🔹 cuDNN version:", tf_build_info.build_info.get('cudnn_version'))
