def build_lstm_tda_model(window_size=6, n_features=4, units=64, layers=1, d_k=64, dropout=0.2, learning_rate=0.001):
    import tensorflow as tf
    from .tda import TemporalDecayAttention
    inp=tf.keras.Input(shape=(window_size,n_features)); x=inp
    for i in range(layers): x=tf.keras.layers.LSTM(units, return_sequences=True)(x)
    x=tf.keras.layers.Dropout(dropout)(x); ctx, att, beta=TemporalDecayAttention(d_k=d_k, name='tda')(x); out=tf.keras.layers.Dense(1, activation='sigmoid', name='risk_score')(ctx)
    model=tf.keras.Model(inp,out); model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse'); return model
