import pytest
tf = pytest.importorskip('tensorflow')
from src.models.tda import TemporalDecayAttention

def test_tda_shapes_weights_beta_and_gradients():
    layer=TemporalDecayAttention(d_k=4, beta_init=0.05)
    H=tf.ones((2,5,3))
    with tf.GradientTape() as tape:
        ctx, att, beta=layer(H); loss=tf.reduce_sum(ctx)+tf.reduce_sum(att)+beta
    assert ctx.shape == (2,4); assert att.shape == (2,5)
    tf.debugging.assert_near(tf.reduce_sum(att, axis=1), tf.ones((2,)))
    assert float(beta.numpy()) > 0
    a=att.numpy()[0]
    assert all(a[i] <= a[i+1] + 1e-6 for i in range(len(a)-1))
    grads=tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads)
