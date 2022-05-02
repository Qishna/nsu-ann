import matplotlib.pyplot as plt


def data_visualization(x_train, y_train, x_test, y_test):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(8, 4), sharex=True, sharey=True)
    ax1.plot(x_train, y_train)
    ax2.plot(x_test, y_test)
    ax1.title.set_text('Train data')
    ax2.title.set_text('Test data')
    plt.show()


def model_visualization(loss, x_test, y_test, predict_on_test):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(loss)

    ax2.plot(x_test, y_test, label='test')
    ax2.plot(x_test, predict_on_test, label='predict')

    ax1.title.set_text('Loss function')
    ax2.title.set_text('Test and predict')
    plt.legend()
