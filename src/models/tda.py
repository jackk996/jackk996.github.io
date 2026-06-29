try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

if tf is not None:
    class TemporalDecayAttention(tf.keras.layers.Layer):
        def __init__(self, d_k=64, beta_init=0.05, **kwargs):
            super().__init__(**kwargs); self.d_k=d_k; self.beta_init=beta_init
        def build(self, input_shape):
            d_h=int(input_shape[-1]); self.W_k=self.add_weight(name='W_k', shape=(d_h,self.d_k), initializer='glorot_uniform', trainable=True)
            self.W_v=self.add_weight(name='W_v', shape=(d_h,self.d_k), initializer='glorot_uniform', trainable=True)
            self.q=self.add_weight(name='q', shape=(self.d_k,), initializer='glorot_uniform', trainable=True)
            import numpy as np
            raw=np.log(np.exp(self.beta_init)-1.0)
            self.raw_beta=self.add_weight(name='raw_beta', shape=(), initializer=tf.keras.initializers.Constant(raw), trainable=True)
        def call(self, H):
            K=tf.matmul(H,self.W_k); V=tf.matmul(H,self.W_v); raw=tf.tensordot(K,self.q,axes=[[-1],[0]])/tf.sqrt(tf.cast(self.d_k,H.dtype))
            T=tf.shape(H)[1]; delta=tf.cast(tf.range(T-1,-1,-1),H.dtype); beta=tf.nn.softplus(self.raw_beta)
            att=tf.nn.softmax(raw-beta*delta, axis=1); ctx=tf.reduce_sum(att[...,None]*V, axis=1)
            return ctx, att, beta
        def get_config(self): return {**super().get_config(), 'd_k':self.d_k, 'beta_init':self.beta_init}
else:
    class TemporalDecayAttention: # type: ignore
        def __init__(self,*a,**k): raise ImportError('TensorFlow is required')
