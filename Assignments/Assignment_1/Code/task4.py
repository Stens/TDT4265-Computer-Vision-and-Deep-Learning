import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
from task3 import calculate_accuracy, SoftmaxTrainer
np.random.seed(0)


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 500
    learning_rate = 0.01
    batch_size = 128
    shuffle_dataset = True
    show_plots = False
    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Train a model with L2 regularization (task 4b)
    l2_lambdas = [1, .1, .01, .001, 0]

    norm_list = []
    train_history_list = []
    val_history_list = []

    for l in l2_lambdas:
        # Intialize model
        model = SoftmaxModel(l)
        # Train model
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        train_history_list.append(train_history)
        val_history_list.append(val_history)

        print("Final Train Cross Entropy Loss:",
              cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:",
              cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
        print("Final Validation accuracy:",
              calculate_accuracy(X_val, Y_val, model))

        # Plotting of softmax weights (Task 4b)
        image_weights = model.w[:-1, :]  # remove bias
        image_weights = image_weights.reshape(28, 28, 10)
        for i in range(10):
            plt.imshow(image_weights[:, :, i], cmap="gray")
            # plt.savefig("task4b_softmax_weight_"+str(i) +
            #             "_"+str(l)+"_.png")
            if show_plots:
                plt.show()
            else:
                plt.close()

        # Save the norm for plotting later
        norm_list.append(np.linalg.norm(model.w))

    # Plotting of accuracy for differente values of lambdas (task 4c)
    for l, train_history, val_history in zip(l2_lambdas, train_history_list, val_history_list):
        plt.ylim([0.7, .93])
        plt.xlim([-100, 10000])
        utils.plot_loss(train_history["accuracy"],
                        "Training Accuracy $\lambda$ = "+str(l))
        utils.plot_loss(val_history["accuracy"],
                        "Validation Accuracy $\lambda$ = "+str(l))
        plt.xlabel("Number of Training Steps")
        plt.ylabel("Accuracy")
        plt.legend()

    plt.savefig("task4c_l2_reg_accuracy.png")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Task 4d - Plotting of the l2 norm for each weight
    plt.plot(l2_lambdas, norm_list)
    plt.title("Plot of length of w vs $\lambda$")
    plt.xlabel("$\lambda$")
    plt.ylabel("$||w|^2$")
    plt.savefig("task4d_l2_reg_norms.png")
    if show_plots:
        plt.show()
    else:
        plt.close()
