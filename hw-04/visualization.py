import matplotlib.pyplot as plt


def predict_visualization(true, predict):
    plt.figure(figsize=(30, 10))
    plt.plot(predict, label="predict")
    plt.plot(true, label="true")
    plt.legend()
    plt.show()
