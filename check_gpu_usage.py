import tensorflow as tf
from tensorflow.keras import backend as K

print('TensorFlow version:')
print(tf.__version__)

print("First GPU's name:", tf.test.gpu_device_name())

print('Check available GPUs:')
tf.config.experimental.list_physical_devices(device_type='GPU')
