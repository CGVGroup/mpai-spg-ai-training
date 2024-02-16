import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

features=5
cars=4
sequence_length=20
feat_pred=3

##Model definitions

def MLP_Model(units):
    return tf.keras.models.Sequential([layers.InputLayer(input_shape=(sequence_length,cars*features)),
                                       layers.Flatten(),
                                       layers.Dense(units,activation="linear"),
                                       layers.Dense(feat_pred*cars,activation="linear")])

def LSTM_Model(droupout_rate, 
               cells=3,
               units=32,
               normalization=False, 
               mlp_cells=0,
               mlp_units=32
              ):

    inputs=tf.keras.Input(shape=(sequence_length,cars*features))
    x=(inputs)
    if(normalization):
        x=layers.BatchNormalization() (x)
    for index in range(cells):
        x=layers.LSTM(units,dropout=droupout_rate,return_sequences=True)(x)
        x= layers.BatchNormalization()(x)
    x=layers.LSTM(units,dropout=droupout_rate)(x)
    x=layers.Flatten()(x)
    for i in range(mlp_cells):
        x=layers.Dense(mlp_units) (x)
    outputs=layers.Dense(feat_pred*cars,activation="linear")(x)
            
    return tf.keras.models.Model(inputs=inputs,outputs=outputs,name=f"LSTM_NN{cells}")

#We tried using a CNN & LSTM model but  
def CNN_LSTM_Model(droupout_rate, units=32,filters=32,kernel_size=3,pool_size=1,normalization=False):
    
    inputs=tf.keras.Input(shape=(sequence_length,cars*features))
    x=(inputs)
    if(normalization):
        x=layers.BatchNormalization() (x)
    #CNN
    x=layers.Conv1D(filters,kernel_size,padding="same",activation="linear")(x)
    x=layers.MaxPool1D(pool_size=pool_size)(x)
    #LSTM
    x=layers.LSTM(units,dropout=droupout_rate)(x)

    outputs=layers.Dense(feat_pred*cars,activation="linear")(x)
            
    return tf.keras.models.Model(inputs=inputs,outputs=outputs,name=f"CNN{filters}_{kernel_size}LSTM{units}")

##Training functions 
def compile_and_fit(model, train,val, patience=5, epochs=10, learning_rate=0.01, summary=False, verbose=0,name="logs",path="logs"):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.5,
                                                   patience=5,
                                                   verbose=verbose,
                                                   mode='auto',
                                                   min_delta=0.01,
                                                   cooldown=5,
                                                   min_lr=1e-12
                                                  )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./{path}/{name}")

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./{path}/{name}/tmp/checkpoint",
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )


    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.Accuracy()])
    
    history = model.fit(train, 
                        epochs=epochs,
                      validation_data=val,
                      verbose=verbose,
                      callbacks=[early_stopping,reduce_lr,tensorboard_callback,model_checkpoint_callback])
    if(summary):
        model.summary()
    
    return history

def MLP_Train(train,val,learning_rate=0.01,verbose=0,epoch=10,units=32,name="logs",path="logs"):
    model=MLP_Model(units)
    history = compile_and_fit(model,train,val,epochs=epoch,learning_rate=learning_rate,verbose=verbose,name=name, path=path)
    
    
    return model,history

def LSTM_Train(train,
               val,
               learning_rate=0.01,
               verbose=0,
               epoch=10, 
               patience=5,
               name="logs",
               path="logs",
               dropout=0.2,
               cells=3,
               units=32,
               normalization=False,
               mlp_cells=0,
               mlp_units=32
              ):
    
    model = LSTM_Model(dropout,
                       cells=cells,
                       units=units, 
                       normalization=normalization, 
                       mlp_cells=mlp_cells,
                       mlp_units=mlp_units
                      )
    history = compile_and_fit(model,
                              train,
                              val,
                              epochs=epoch,
                              learning_rate=learning_rate,
                              verbose=verbose,
                              patience=patience,
                              name=name,
                              path=path
                             )
    
    
    return model,history

def CNN_LSTM_Train(train,val,learning_rate=0.01,verbose=0,epoch=10, patience=5,name="logs",path="logs",
                   dropout=0.2, kernel_size=3,units=32, filters=32,pool_size=1, normalization=False):
    
    model = CNN_LSTM_Model(dropout,kernel_size=kernel_size,units=units,filters=filters, normalization=normalization,pool_size=pool_size)
    history = compile_and_fit(model,train,val,epochs=epoch,learning_rate=learning_rate,verbose=verbose,patience=patience,name=name,path=path)
    
    #Print_Train(history,"CNN_LSTM")
    
    return model,history