import tensorflow as tf
from sklearn.metrics import mean_squared_error

from datasets.Turbofan_Engine_Degradation.turbofan_preprocessing import turbofan_loading_and_preprocessing

print(tf.__version__)
print("First GPU's name: ", tf.test.gpu_device_name())
tf.config.experimental.list_logical_devices(device_type='GPU')

x_train, x_test, y_train, y_test = turbofan_loading_and_preprocessing()

model = tf.keras.Sequential()

model.add(tf.keras.layers.InputLayer(input_shape=len(x_train.keys())))

model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(1, activation='linear'))

adam = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=adam, loss='mse', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1024, validation_data=(x_test, y_test),
          verbose=1)

y_pred = model.predict(x_test)

loss = mean_squared_error(y_test, y_pred)

print('Loss: ', loss)
