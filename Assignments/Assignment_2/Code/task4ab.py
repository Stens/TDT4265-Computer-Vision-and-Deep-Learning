import utils
import matplotlib.pyplot as plt
import numpy as np
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy
np.random.seed(0)


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    # Settings for task 4
    num_epochs = 100
    learning_rate = .02
    batch_size = 32
    momentum_gamma = .9  # Task 3 hyperparameter

    shuffle_data = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    neurons_per_layer = [64, 10]

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_64, val_history_64 = trainer.train(num_epochs)
    print("\n\n --------------------------")
    print(neurons_per_layer)
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
    print("\n\n --------------------------")

    neurons_per_layer = [32, 10]

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_32, val_history_32 = trainer.train(num_epochs)
    print("\n\n --------------------------")
    print(neurons_per_layer)
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
    print("\n\n --------------------------")

    neurons_per_layer = [128, 10]

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_128, val_history_128 = trainer.train(num_epochs)

    print("\n\n --------------------------")
    print(neurons_per_layer)
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
    print("\n\n --------------------------")

    # Plot loss for first model
    plt.figure(figsize=(20, 12))
    plt.ylim([0., .5])
    # 64 units
    utils.plot_loss(train_history_64["loss"],
                    "Training Loss (64)", npoints_to_average=10)
    utils.plot_loss(val_history_64["loss"], "Validation Loss (64)")
    # 32 units
    utils.plot_loss(train_history_32["loss"],
                    "Training Loss (32)", npoints_to_average=10)
    utils.plot_loss(val_history_32["loss"], "Validation Loss (32)")
    # 128 units
    utils.plot_loss(train_history_128["loss"],
                    "Training Loss (128)", npoints_to_average=10)
    utils.plot_loss(val_history_128["loss"], "Validation Loss (128)")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.show()

    plt.figure(figsize=(20, 12))

    plt.ylim([0.90, 1.01])
    # 64 units
    utils.plot_loss(train_history_64["accuracy"], "Training Accuracy (64)")
    utils.plot_loss(val_history_64["accuracy"], "Validation Accuracy (64)")
    # 32 units
    utils.plot_loss(train_history_32["accuracy"], "Training Accuracy (32)")
    utils.plot_loss(val_history_32["accuracy"], "Validation Accuracy (32)")
    # 128 units
    utils.plot_loss(train_history_128["accuracy"], "Training Accuracy (128)")
    utils.plot_loss(val_history_128["accuracy"], "Validation Accuracy (128)")

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
