def set_global_seed(seed, deterministic=True):
    import os, random, numpy as np
    os.environ['PYTHONHASHSEED']=str(seed)
    if deterministic: os.environ.setdefault('TF_DETERMINISTIC_OPS','1')
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        if deterministic: tf.config.experimental.enable_op_determinism()
    except Exception: pass
