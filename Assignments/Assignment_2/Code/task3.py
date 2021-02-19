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

    def plot_metrics_loss(train_history, val_history, description=["", ""],*,use_val=False):
        #plt.subplot(1, 2, 1)  # Warning
        plt.title("Training and Validation loss")
        utils.plot_loss(train_history["loss"],
                        ("Training " + description[0]), npoints_to_average=10)
        if use_val:
            utils.plot_loss(val_history["loss"], ("Validation " + description[0]))
        plt.ylim([0, .6])
        plt.legend(loc="upper left")
        plt.xlabel("Training steps")
        plt.ylabel("Average Cross entropy Loss")
    def plot_metrics_acc(train_history, val_history, description=["", ""],*,use_val=False):
        #plt.subplot(1, 2, 2)
        plt.title("Training and Validation accuracy")
        utils.plot_loss(train_history["accuracy"], ("Training " + description[1]))
        if use_val:
            utils.plot_loss(val_history["accuracy"],("Validation " + description[1]))
        plt.ylim([0.85, 1.])
        plt.ylabel("Accuracy")
        plt.xlabel("Training steps")
        plt.legend(loc="lower right")

    
    # First nothing
    use_improved_weight_init = False
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

    train_history_nothing, val_history_nothing = trainer.train(num_epochs)
    
    
    # Adding improved weights
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

    train_history_im_weights, val_history_im_weights = trainer.train(num_epochs)
    
    
    # Adding improved sigmoid
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

    train_history_im_sigmoid, val_history_im_sigmoid = trainer.train(num_epochs)
    
    
    
    # Adding momentum
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

    train_history_momentum, val_history_momentum = trainer.train(num_epochs)
    

    # All plotting is done after the training with different tricks is done
    # Remove the ones not wanted in plot (if you want to avoid messy plot)
    
    plot_metrics_loss(train_history_nothing, val_history_nothing, [
                 "Loss (no additions)", "Accuracy (no additions)"],use_val=True)
    plot_metrics_loss(train_history_im_weights, val_history_im_weights, [
                 "Loss (improved weights)", "Accuracy (improved weights)"],use_val=True)
    plot_metrics_loss(train_history_im_sigmoid, val_history_im_sigmoid, [
                "Loss (improved sigmoid + weights)", "Accuracy (improved sigmoid + weights)"],use_val=True)
    plot_metrics_loss(train_history_momentum, val_history_momentum, [
                 "Loss (weights, sigmoid and momentum)", "Accuracy (weights, sigmoid and momentum)"],use_val=True)
    #plt.savefig("task3a_loss.png")

    plt.show()

    plot_metrics_acc(train_history_nothing, val_history_nothing, [
                 "Loss (no additions)", "Accuracy (no additions)"],use_val=True) 
    plot_metrics_acc(train_history_im_weights, val_history_im_weights, [
                 "Loss (improved weights)", "Accuracy (improved weights)"],use_val=True)
    plot_metrics_acc(train_history_im_sigmoid, val_history_im_sigmoid, [
                     "Loss (improved sigmoid + weights)", "Accuracy (improved sigmoid + weights)"],use_val=True)
    plot_metrics_acc(train_history_momentum, val_history_momentum, [
                "Loss (weights, sigmoid and momentum)", "Accuracy (weights, sigmoid and momentum)"],use_val=True) 
    
    #plt.savefig("task3a_acc.png")
    plt.show()












"""
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
                 "Loss (improved sigmoid)", "Accuracy (improved sigmoid)"])

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
                 "Loss (momentum)", "Accuracy (momentum)"])
    """