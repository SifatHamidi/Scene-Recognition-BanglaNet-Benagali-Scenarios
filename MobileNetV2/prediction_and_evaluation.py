# -*- coding: utf-8 -*-
"""Prediction and Evaluation

The image labels with their corresponding index number-

0.   'Jackfruit'
1.   'Mango field'
2.   'Pohela Boishakh'
3.   'Rice field'
4.   'Rickshaw'
5.   'River Boat'
6.   'Traffic Jam'
7.   'Village House'
8.   'churi'
9.   'flood'
10.  'fuchka'
11.  'mosque' 
12.  'nakshi pitha'
"""

model.load_weights('/content/drive/My Drive/mobilenet.hdf5')
print('model weights loaded')

test_gen = datagen.flow_from_directory(data_root/'Test',
                                       target_size = (Dim,Dim),
                                       batch_size = 10,
                                       shuffle = False)

#test_gen.reset()
pred=model.predict_generator(test_gen,
                             steps = test_steps_per_epoch ,
                             verbose = 1)

predicted_class_indices = np.argmax(pred, axis = 1)
print(predicted_class_indices[89])

predicted_class_indices = np.argmax(pred, axis = 1)
print(predicted_class_indices[89])

import cv2
test = cv2.imread('/content/drive/MyDrive/Test/test/0204.jpg')
test1 = cv2.resize(test, (256, 256))
plt.imshow(test1)
plt.show()

val_loss,val_acc = model.evaluate(val_gen)
print('Validation loss', val_loss)
print('Validation accuracy', val_acc)

def plot_history(histories, key = 'accuracy'):
  plt.figure(figsize = (15,5))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label = name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color = val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])

# plot history
plot_history([('model', fit_history)])
