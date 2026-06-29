def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=16, patience=20, restore_best_weights=True):
    import tensorflow as tf
    cbs=[]
    if patience and patience < epochs: cbs.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=restore_best_weights))
    return model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=epochs,batch_size=batch_size,callbacks=cbs,verbose=0)
