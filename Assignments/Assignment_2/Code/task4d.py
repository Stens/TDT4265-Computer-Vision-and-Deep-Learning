import utils
import matplotlib.pyplot as plt
import numpy as np
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy
np.random.seed(0)


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True  # True for all tests, as this was implemented last assignment

    # Using improvements
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    def plot_metrics_loss(train_history, val_history, description=["", ""], *, use_val=False):
        # plt.subplot(1, 2, 1)  # Warning
        plt.title("Training and Validation loss")
        utils.plot_loss(train_history["loss"],
                        ("Training " + description[0]), npoints_to_average=10)
        if use_val:
            utils.plot_loss(val_history["loss"],
                            ("Validation " + description[0]))
        plt.ylim([0, .6])
        plt.legend(loc="upper left")
        plt.ylabel("Average Cross Entropy Loss")
        plt.xlabel("Training steps")

    def plot_metrics_acc(train_history, val_history, description=["", ""], *, use_val=False):
        # plt.subplot(1, 2, 2)
        plt.title("Training and Validation accuracy")
        utils.plot_loss(train_history["accuracy"],
                        ("Training " + description[1]))
        if use_val:
            utils.plot_loss(val_history["accuracy"],
                            ("Validation " + description[1]))
        plt.ylim([0.85, 1.01])
        plt.ylabel("Accuracy")
        plt.xlabel("Training steps")
        plt.legend(loc="lower right")

    # Single hidden layer
    neurons_per_layer = [64, 10]

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,)

    train_history_single, val_history_single = trainer.train(num_epochs)
    print("---------", neurons_per_layer, "----------")
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # Two hidden layers
    neurons_per_layer = [59, 59, 10]

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,)

    train_history_double, val_history_double = trainer.train(
        num_epochs)
    print("---------", neurons_per_layer, "----------")
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plot_metrics_loss(train_history_single, val_history_single, [
        "Loss (single hidden layer)", "Accuracy (single hidden layer)"], use_val=True)
    plot_metrics_loss(train_history_double, val_history_double, [
        "Loss (two hidden layers)", "Accuracy (two hidden layers)"], use_val=True)

    plt.show()

    plot_metrics_acc(train_history_single, val_history_single, [
        "Loss (single hidden layer)", "Accuracy (single hidden layer)"], use_val=True)
    plot_metrics_acc(train_history_double, val_history_double, [
        "Loss (two hidden layers)", "Accuracy (two hidden layers)"], use_val=True)

    plt.show()
