import cupy as cp
import numpy as np


def trainer(x_train, y_train, x_val, y_val, epoch_num, learning_rate, model):
    mempool = cp.get_default_memory_pool()
    num_train = len(x_train)
    batch_size = len(x_train)
    x_val_gpu = cp.array(x_val)
    y_val_gpu = cp.array(y_val)

    best_val_loss = 9999999

    for epoch in range(epoch_num):

        shuffled_indices = np.arange(num_train)
        np.random.shuffle(shuffled_indices)
        sections = np.arange(batch_size, num_train, batch_size)
        batches_indices = np.array_split(shuffled_indices, sections)
        batch_losses = np.zeros(len(batches_indices))

        for batch_id, batch_indices in enumerate(batches_indices):
            batch_X = x_train[batch_indices]
            batch_y = y_train[batch_indices]

            batch_y_gpu = cp.asarray(batch_y)
            batch_X_gpu = cp.asarray(batch_X)

            out = model.forward(batch_X_gpu)
            train_loss = cp.mean((out - batch_y_gpu) ** 2)

            grad = out - batch_y_gpu

            for param in model.params().values():
                param.grad.fill(0)
            model.backward(grad)

            for param_name, param in model.params().items():
                optimizer = model.optimizers[param_name]
                optimizer.update(param.value, param.grad, learning_rate)

            batch_losses[batch_id] = train_loss.get()
            mempool.free_all_blocks()

        val_out = model.forward(x_val_gpu)
        val_loss = cp.mean((val_out - y_val_gpu) ** 2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            for param_name, param in model.params().items():
                cp.save(f"best_model_LSTM/{param_name}.npy", param.value)

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch}:  Train loss: {batch_losses.mean():.5f}  Val loss: {val_loss:.5f}"
            )
