import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import src.NAVEM.NeuralNetwork.rescaled_bfgs as rescaled_bfgs
import time

tfp.optimizer.bfgs = rescaled_bfgs

class PrintInfoCb(tf.keras.callbacks.Callback):
    def __init__(self, n_epochs, info_printer):
        super(PrintInfoCb, self).__init__()
        self.n_epochs = n_epochs
        self.info_printer = info_printer

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            self.info_printer(self.model, epoch, self.n_epochs)

    def on_train_end(self, logs=None):
        current = logs.get("loss")
        tf.print("Final loss: {:<.16e}".format(current))


class StoreLoss(tf.keras.callbacks.Callback):
    def __init__(self, list_of_losses, n_steps=100):
        super(StoreLoss, self).__init__()
        self.list_of_losses = list_of_losses
        self.n_steps = n_steps

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if epoch % self.n_steps == 0:
            self.list_of_losses.append(current)


class StoreTime(tf.keras.callbacks.Callback):
    def __init__(self, list_of_times, n_steps=100):
        super(StoreTime, self).__init__()
        self.list_of_times = list_of_times
        self.n_steps = n_steps
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.n_steps == 0:
            self.list_of_times.append(time.perf_counter() - self.t0)


def get_lr_scheduler(n_epochs, lower, upper):
    def scheduler(epoch, lr):
        return upper * np.exp(np.log(lower / upper) * epoch / n_epochs)

    return scheduler


def train_with_first_order(model, loss_f, x, y, bsize, n_epochs, shuffle, cb_list, info_printer):
    if info_printer is not None:
        cb_list.append(PrintInfoCb(n_epochs, info_printer))

    optimizer = tf.optimizers.Adam(1e-3, epsilon=1e-13)
    model.compile(optimizer=optimizer, loss=loss_f)
    model.fit(x, y, batch_size=bsize, epochs=n_epochs, shuffle=shuffle, verbose=0, callbacks=cb_list)
    del optimizer


def print_info_h_navem(model, iteration, total_iterations=None):
    if total_iterations is None:
        l2 = tf.strings.regex_replace(tf.strings.as_string(model.curr_distance_l2,
                                                           scientific=True,
                                                           precision=16), '\"', '')
        h1 = tf.strings.regex_replace(tf.strings.as_string(model.curr_distance_h1,
                                                           scientific=True,
                                                           precision=16), '\"', '')
        tf.print(tf.strings.join(["Iter: ",
                                  tf.strings.as_string(iteration),
                                  ". Current distance L2 and H1: ",
                                  l2,
                                  "\t",
                                  h1]))
    else:
        tf.print("Performed {:<d} / {:<d} epochs. Current distance L2 and H1: {:<.16e} \t {:<.16e}".
                 format(iteration, total_iterations,
                        model.curr_distance_l2.read_value(), model.curr_distance_h1.read_value()))


def print_info_p_navem(model, iteration, total_iterations=None):
    if model.loss_x_dx.read_value() == 0:
        if total_iterations is None:
            loss_one = tf.strings.regex_replace(tf.strings.as_string(model.loss_one,
                                                                     scientific=True,
                                                                     precision=16), '\"', '')
            loss_x = tf.strings.regex_replace(tf.strings.as_string(model.loss_x,
                                                                   scientific=True,
                                                                   precision=16), '\"', '')
            loss_y = tf.strings.regex_replace(tf.strings.as_string(model.loss_y,
                                                                   scientific=True,
                                                                   precision=16), '\"', '')
            tf.print(tf.strings.join(["Iter: ", tf.strings.as_string(iteration),
                                      ". Current error for 1, x and y: ",
                                      loss_one, "\t", loss_x, "\t", loss_y]))
        else:
            tf.print("Performed {:<d} / {:<d} epochs. Current error for 1, x and y: {:<.5e} \t {:<.5e} \t {:<.5e}".
                     format(iteration, total_iterations,
                            model.loss_one.read_value(), model.loss_x.read_value(), model.loss_y.read_value()))
    else:

        if model.loss_x.read_value() == 0:
            if total_iterations is None:
                loss_one_dx = tf.strings.regex_replace(tf.strings.as_string(model.loss_one_dx,
                                                                         scientific=True,
                                                                         precision=16), '\"', '')
                loss_x_dx = tf.strings.regex_replace(tf.strings.as_string(model.loss_x_dx,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                loss_y_dx = tf.strings.regex_replace(tf.strings.as_string(model.loss_y_dx,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                loss_one_dy = tf.strings.regex_replace(tf.strings.as_string(model.loss_one_dy,
                                                                         scientific=True,
                                                                         precision=16), '\"', '')
                loss_x_dy = tf.strings.regex_replace(tf.strings.as_string(model.loss_x_dy,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                loss_y_dy = tf.strings.regex_replace(tf.strings.as_string(model.loss_y_dy,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                tf.print(tf.strings.join(["Iter: ", tf.strings.as_string(iteration),
                                          ". Current errors: ",
                                          loss_one_dx, "\t", loss_x_dx, "\t", loss_y_dx, "\t",
                                          loss_one_dy, "\t", loss_x_dy, "\t", loss_y_dy]))
            else:
                tf.print(("Performed {:<d} / {:<d} epochs. Current errors: "
                          " {:<.5e} \t {:<.5e} \t {:<.5e} \t  {:<.5e} \t {:<.5e} \t {:<.5e}").
                         format(iteration, total_iterations,
                                model.loss_one_dx.read_value(), model.loss_x_dx.read_value(),
                                model.loss_y_dx.read_value(), model.loss_one_dy.read_value(),
                                model.loss_x_dy.read_value(), model.loss_y_dy.read_value()
                                ))
        else:
            if total_iterations is None:
                loss_one = tf.strings.regex_replace(tf.strings.as_string(model.loss_one,
                                                                         scientific=True,
                                                                         precision=16), '\"', '')
                loss_x = tf.strings.regex_replace(tf.strings.as_string(model.loss_x,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                loss_y = tf.strings.regex_replace(tf.strings.as_string(model.loss_y,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                loss_one_dx = tf.strings.regex_replace(tf.strings.as_string(model.loss_one_dx,
                                                                         scientific=True,
                                                                         precision=16), '\"', '')
                loss_x_dx = tf.strings.regex_replace(tf.strings.as_string(model.loss_x_dx,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                loss_y_dx = tf.strings.regex_replace(tf.strings.as_string(model.loss_y_dx,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                loss_one_dy = tf.strings.regex_replace(tf.strings.as_string(model.loss_one_dy,
                                                                         scientific=True,
                                                                         precision=16), '\"', '')
                loss_x_dy = tf.strings.regex_replace(tf.strings.as_string(model.loss_x_dy,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                loss_y_dy = tf.strings.regex_replace(tf.strings.as_string(model.loss_y_dy,
                                                                       scientific=True,
                                                                       precision=16), '\"', '')
                tf.print(tf.strings.join(["Iter: ", tf.strings.as_string(iteration),
                                          ". Current errors: ",
                                          loss_one, "\t", loss_x, "\t", loss_y, "\t",
                                          loss_one_dx, "\t", loss_x_dx, "\t", loss_y_dx, "\t",
                                          loss_one_dy, "\t", loss_x_dy, "\t", loss_y_dy]))
            else:
                tf.print(("Performed {:<d} / {:<d} epochs. Current errors: {:<.5e} \t {:<.5e} \t {:<.5e} \t"
                          " {:<.5e} \t {:<.5e} \t {:<.5e} \t  {:<.5e} \t {:<.5e} \t {:<.5e}").
                         format(iteration, total_iterations,
                                model.loss_one.read_value(), model.loss_x.read_value(), model.loss_y.read_value(),
                                model.loss_one_dx.read_value(), model.loss_x_dx.read_value(),
                                model.loss_y_dx.read_value(), model.loss_one_dy.read_value(),
                                model.loss_x_dy.read_value(), model.loss_y_dy.read_value()
                                ))


def print_info_b_navem(model, iteration, total_iterations=None):
    if total_iterations is None:
        laplacian = tf.strings.regex_replace(tf.strings.as_string(model.curr_laplacian,
                                                                  scientific=True,
                                                                  precision=16), '\"', '')
        tf.print(tf.strings.join(["Iter: ",
                                  tf.strings.as_string(iteration),
                                  ". Current Laplacian: ",
                                  laplacian]))
    else:
        tf.print("Performed {:<d} / {:<d} epochs. Current Laplacian: {:<.16e}".
                 format(iteration, total_iterations, model.curr_laplacian.read_value()))


def function_factory(model, loss, train_x, train_y, info_printer):
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.prod(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    @tf.function(experimental_relax_shapes=True)
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for j, (shape_j, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[j].assign(tf.reshape(param, shape_j))

    # now create a function that will be returned by this factory
    @tf.function(experimental_relax_shapes=True)
    def f(params_1d):
        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(train_y, model(train_x, training=True))

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        if f.iter.read_value() % 100 == 0:
            if loss_value < model.best_loss:
                model.best_w.assign(params_1d)
                model.best_loss.assign(loss_value)
            model.bfgs_regularization_grads.assign(params_1d)
        grads += model.bfgs_regularization_grads.read_value() * model.reg_coefficient

        # print out iteration & loss
        f.iter.assign_add(1)
        if f.iter.read_value() % 100 == 0:
            info_printer(model, f.iter)

        return loss_value, grads

    # store these information as members, so we can use them outside the scope
    f.iter = tf.Variable(0, dtype=tf.int32)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.losses = []
    f.errors = []
    f.errors_params = []
    f.t0 = time.perf_counter()
    f.times = []

    return f


def train_w_bfgs(model, loss, train_data, train_labels, n_epochs, bfgs=True, inv_hess=None, info_printer=None):
    if not (n_epochs > 0):
        return None

    func = function_factory(model, loss, train_data, train_labels, info_printer)

    max_iters = tf.convert_to_tensor(n_epochs, dtype=tf.int32)
    max_ls_iters = 50
    parl_iters = 1

    init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)
    if bfgs:
        results = tfp.optimizer.bfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params,
            max_iterations=max_iters,
            max_line_search_iterations=max_ls_iters, parallel_iterations=parl_iters, tolerance=1e-30,
            initial_inverse_hessian_estimate=inv_hess, validate_args=False)
    else:
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params,
            max_iterations=max_iters,
            max_line_search_iterations=max_ls_iters, parallel_iterations=parl_iters, tolerance=1e-30)

    tf.print("Training finished with Loss =  {:<.16e}".format(model.best_loss.read_value()))
    return [results, func]


def assign_best(model):
    func = function_factory(model, None, None, None, None)
    func.assign_new_model_parameters(model.best_w)


def train_adam_bfgs(model, loss, inputs, labels, epochs_adam, epochs_bfgs, cb_list, use_bfgs=True):
    if model.name == "h_navem_neural_network":
        info_printer = print_info_h_navem
    elif model.name == "b_navem_neural_network":
        info_printer = print_info_b_navem
    elif model.name == "p_navem_neural_network":
        info_printer = print_info_p_navem
    else:
        info_printer = None

    train_with_first_order(model, loss, inputs, labels, inputs.shape[0], epochs_adam, shuffle=False, cb_list=cb_list,
                           info_printer=info_printer)
    results = train_w_bfgs(model, loss, inputs, labels, epochs_bfgs, bfgs=use_bfgs, info_printer=info_printer)
    return results
