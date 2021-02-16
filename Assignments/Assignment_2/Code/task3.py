import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True  # True for all tests, as this was implemented last assignment

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    def plot_metrics(train_history, val_history, description=["", ""]):
        plt.subplot(1, 2, 1)  # Warning
        utils.plot_loss(train_history["loss"],
                        description[0], npoints_to_average=10)
        plt.ylim([0, .6])
        plt.legend(loc="upper left")

        plt.subplot(1, 2, 2)
        utils.plot_loss(val_history["accuracy"], description[1])
        plt.ylim([0.85, 1.])
        plt.ylabel("Validation Accuracy")
        plt.legend(loc="upper right")

    # Setting parameters
    use_improved_weight_init = True
    use_improved_sigmoid = False
    use_momentum = False

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    plot_metrics(train_history, val_history, [
                 "Loss improved weights", "Accuracy improved weights"])
    # Setting parameters,
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = False

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(
        num_epochs)
    plot_metrics(train_history, val_history, [
                 "Loss improved sigmoid", "Accuracy improved sigmoid"])

    # Setting parameters, all true
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(
        num_epochs)

    plot_metrics(train_history, val_history, [
                 "Loss with momentum", "Accuracy with momentum"])

    plt.show()
