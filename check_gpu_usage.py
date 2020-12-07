import tensorflow as tf
from tensorflow.keras import backend as K

print('TensorFlow version:')
print(tf.__version__)

print('-----------------------')

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('Could not find any GPU device')
print('Found GPU at: {}'.format(device_name))

print('-----------------------')

print('List physical devices:')
print(tf.config.experimental.list_physical_devices(device_type='GPU'))
