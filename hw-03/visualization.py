import matplotlib.pyplot as plt
from random import seed, sample


def data_visualization(x_train, x_test, RANDOM_SEED):
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
        nrows=2, ncols=4, figsize=(8, 4), sharex=True, sharey=True
    )
    plt.rc('image', cmap='gray')
    fig.suptitle('Train and test data samples')
    seed(RANDOM_SEED)
    train_samples = sample(range(len(x_train)), 4)
    test_samples = sample(range(len(x_test)), 4)

    ax1.imshow(x_train[train_samples[0]][:,:,::-1])
    ax1.axis("off")
    ax2.imshow(x_train[train_samples[1]][:,:,::-1])
    ax2.axis("off")
    ax3.imshow(x_train[train_samples[2]][:,:,::-1])
    ax3.axis("off")
    ax4.imshow(x_train[train_samples[3]][:,:,::-1])
    ax4.axis("off")

    ax5.imshow(x_test[test_samples[0]][:,:,::-1])
    ax5.axis("off")
    ax6.imshow(x_test[test_samples[1]][:,:,::-1])
    ax6.axis("off")
    ax7.imshow(x_test[test_samples[2]][:,:,::-1])
    ax7.axis("off")
    ax8.imshow(x_test[test_samples[3]][:,:,::-1])
    ax8.axis("off")

    plt.show()
