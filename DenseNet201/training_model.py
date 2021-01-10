# -*- coding: utf-8 -*-
"""Training Model


"""

model = Sequential()
densenet = DenseNet201(include_top=False, 
               weights='imagenet' , 
               input_shape = (Dim, Dim, 3), 
               pooling='avg')
model.add(densenet)
model.add(Dense(64, kernel_regularizer = l2(0.001), activation = 'sigmoid'))
model.add(Dropout(0.25)) #hyperparameter changed, randomley select some layer and exclude them in the next step, we changed this value in order to remove uncessary layer and overfitting problem 
model.add(Dense(64, kernel_regularizer = l2(0.001),activation = 'sigmoid'))
model.add(Dropout(0.25)) #Now we will make a dropout layer to prevent overfitting, which functions by randomly eliminating some of the connections between the layers (0.2 means it drops 20% of the existing connections):
model.add(Dense(Num_class, activation = 'softmax')) #softmax is used here because the last layer decide which class the image belong to,as it is a multi class dataset, softmax is the idea choice for multi class classification 


print('finished loading model.')

model.compile(optimizer = Adam(lr=3e-4),loss ='categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

cb_early_stopper = EarlyStopping(monitor = 'val_loss',patience = 10)

cb_checkpointer=ModelCheckpoint(filepath='/content/drive/My Drive/BanglaNet/densenet201.hdf5',
                               monitor = 'val_loss',
                               save_best_only = True,
                               mode = 'auto')

reducelr=ReduceLROnPlateau(monitor = 'val_loss', 
                           factor = 0.2, 
                           patience = 5, 
                           min_lr = 5e-4)
csv_logger = CSVLogger('densenet201.csv', append=True, separator=',')

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

fit_history=model.fit(train_gen,steps_per_epoch = train_steps_per_epoch,
                      epochs = 50, validation_data=val_gen,
                      validation_steps = val_steps_per_epoch,
                      callbacks  =[cb_checkpointer,cb_early_stopper,reducelr, tensorboard_callback, csv_logger])

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs
