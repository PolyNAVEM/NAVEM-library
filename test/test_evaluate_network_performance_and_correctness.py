import tensorflow as tf




class DummyInternal(tf.keras.layers.Layer):

    """
    Minimal model to simulate internal code internal_call
    """

    def __init__(self, n_features, dtype=tf.float64):
        super().__init__(dtype=dtype)
        self.layers_list = [
            tf.keras.layers.Dense(16, activation="tanh", dtype=dtype),
            tf.keras.layers.Dense(16, activation="tanh", dtype=dtype),
            tf.keras.layers.Dense(16, activation="tanh", dtype=dtype),
            tf.keras.layers.Dense(1, dtype=dtype),
        ]

    def internal_call(self, inputs):
        u = self.layers_list[0](inputs)
        for layer in self.layers_list[1:-1]:
            u = layer(u) + u
        return self.layers_list[-1](u)


def laplacian_call_v1(model, inputs, persistent_inner=True):
    """
    Nested version. Trying to calculate only what we need.
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(inputs)
        with tf.GradientTape(persistent=persistent_inner) as tape1:
            tape1.watch(inputs)
            u0 = model.internal_call(inputs)
        grad = tape1.gradient(u0, inputs)

    u_xx_pre = grad[:, 0:1]
    u_xx_full = tape2.gradient(u_xx_pre, inputs)

    return u_xx_full


def laplacian_call_v2_single_tape_jacobian(model, inputs):
    """
    Starting version that we aim to optimize
    """
    with tf.GradientTape() as tape2:
        tape2.watch(inputs)
        with tf.GradientTape() as tape1:
            tape1.watch(inputs)
            u0 = model.internal_call(inputs)
        grad = tape1.gradient(u0, inputs)

    jac = tape2.batch_jacobian(grad, inputs)

    return jac

def laplacian_call_v3_forward_over_backward(model, inputs):
    """
    Versione efficiente: calcola SOLO le 2 derivate seconde che servono (u_xx, u_yy)
    usando forward-over-backward (tf.autodiff.ForwardAccumulator + GradientTape),
    evitando di calcolare l'intero jacobiano 8x8.
    """
    n_features = inputs.shape[-1]

    def laplacian_terms(direction_onehot):
        # direction_onehot: tensor (n_features,) con 1 in una posizione, 0 altrove
        with tf.autodiff.ForwardAccumulator(inputs, tf.ones_like(inputs) * direction_onehot) as acc:
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                u0 = model.internal_call(inputs)
            grad = tape1.gradient(u0, inputs)
        # acc.jvp(grad) = directional derivative of grad along direction_onehot
        # cioè la colonna del jacobiano corrispondente a quella direzione
        return u0, grad, acc.jvp(grad)

    e_x = tf.constant([1.0, 0.0] + [0.0] * (n_features - 2), dtype=inputs.dtype)
    e_y = tf.constant([0.0, 1.0] + [0.0] * (n_features - 2), dtype=inputs.dtype)

    u0, grad, jvp_x = laplacian_terms(e_x)
    _, _, jvp_y = laplacian_terms(e_y)

    u_xx = jvp_x[:, 0:1]
    u_yy = jvp_y[:, 1:2]
    return u0, grad, u_xx, u_yy


def main():
    tf.random.set_seed(0)
    model = DummyInternal(N_FEATURES, dtype=DTYPE)
    inputs = tf.random.normal((BATCH_SIZE, N_FEATURES), dtype=DTYPE)

    print("\n=== TEST 1: eager mode, NO @tf.function, tape1 persistent=True ===")
    try:
        result = laplacian_call_v1(model, inputs, persistent_inner=True)
        print("Results:", "OK" if result is not None else "fail (None)")
    except Exception as e:
        print("exception:", repr(e))

    print("\n=== TEST 2: eager mode, NO @tf.function, tape1 persistent=False ===")
    try:
        result = laplacian_call_v1(model, inputs, persistent_inner=False)
        print("Results:", "OK" if result is not None else "fail (None)")
    except Exception as e:
        print("exception:", repr(e))

    print("\n=== TEST 3: eager mode, with batch_jacobian===")
    try:
        result = laplacian_call_v2_single_tape_jacobian(model, inputs)
        print("Results:", "OK" if result is not None else "fail (None)")
    except Exception as e:
        print("exception:", repr(e))

    print("\n=== TEST 4: @tf.function, tape1 persistent=True ===")

    @tf.function
    def wrapped_call(inputs):
        return laplacian_call_v1(model, inputs, persistent_inner=True)

    try:
        result = wrapped_call(inputs)
        print("Results:", "OK" if result is not None else "fail (None)")
    except Exception as e:
        print("exception:", repr(e))

    print("\n=== TEST 5: eager mode, input as tf.Variable and not as tf.Tensor, tape1 persistent=True ===")
    inputs_var = tf.Variable(inputs)
    try:
        result = laplacian_call_v1(model, inputs_var, persistent_inner=True)
        print("Results:", "OK" if result is not None else "fail (None)")
    except Exception as e:
        print("exception:", repr(e))

    print("\n=== TEST 6: forward-over-backward (ForwardAccumulator), SOLO u_xx e u_yy, no full jacobian ===")
    try:
        u0, grad, u_xx, u_yy = laplacian_call_v3_forward_over_backward(model, inputs)
        print("  u_xx is None?", u_xx is None, " shape:", None if u_xx is None else u_xx.shape)
        print("  u_yy is None?", u_yy is None, " shape:", None if u_yy is None else u_yy.shape)
        # Comparison with batch_jacobian (Test 3) to evaluate correctness
        jac = laplacian_call_v2_single_tape_jacobian(model, inputs)
        u_xx_ref = jac[:, 0, 0:1]
        u_yy_ref = jac[:, 1, 1:2]
        print("  max abs diff u_xx vs batch_jacobian:", tf.reduce_max(tf.abs(u_xx - u_xx_ref)).numpy())
        print("  max abs diff u_yy vs batch_jacobian:", tf.reduce_max(tf.abs(u_yy - u_yy_ref)).numpy())
        print("Results:", "OK" if u_xx is not None and u_yy is not None else "fail (None)")
    except Exception as e:
        print("exception:", repr(e))

    print("\n=== TEST 7: timing comparison (batch_jacobian vs forward-over-backward) ===")

    import time
    big_inputs = tf.random.normal((1056, N_FEATURES), dtype=DTYPE)
    @tf.function
    def full_jacobian_fn(x):
        return laplacian_call_v2_single_tape_jacobian(model, x)

    @tf.function
    def fwd_over_bwd_fn(x):
        _, _, u_xx, u_yy = laplacian_call_v3_forward_over_backward(model, x)
        return u_xx, u_yy

    # warmup (per il tracing di @tf.function)
    _ = full_jacobian_fn(big_inputs)
    _ = fwd_over_bwd_fn(big_inputs)

    n_iter = 50

    t0 = time.time()
    for _ in range(n_iter):
        _ = full_jacobian_fn(big_inputs)
    t1 = time.time()
    print(f"  batch_jacobian completo: {(t1 - t0) / n_iter * 1000:.3f} ms/iter")

    t0 = time.time()
    for _ in range(n_iter):
        _ = fwd_over_bwd_fn(big_inputs)
    t1 = time.time()
    print(f"  forward-over-backward (solo u_xx,u_yy): {(t1 - t0) / n_iter * 1000:.3f} ms/iter")


if __name__ == "__main__":

    print("TensorFlow version:", tf.__version__)

    # --- Parameters for quadrilateral B-NAVEM ---#
    N_FEATURES = 8
    DTYPE = tf.float64

    # ----- Test case ----#
    BATCH_SIZE = 16

    main()