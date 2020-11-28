import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier


def generate_xor_experiment():
    set_random_seed()

    xor_dataset = make_xor_dataset(2000)

    samples = xor_dataset["samples"]
    centers = generate_centers(samples, n_clusters=7)
    kernelized_inputs = apply_rbf_kernel(samples, centers, gamma=5)

    model = create_sgd_classifier(alpha=0.001, learning_rate=0.7)

    n_iter = 10
    y = xor_dataset["classes"]

    weights_history, pred_history = partial_weights_and_prediction(
        kernelized_inputs, y, model, n_iter)

    save_array_csv(centers, file_name="centers_xor.csv")
    save_array_csv(weights_history, file_name="weight_history_xor.csv")
    save_array_csv(pred_history, file_name="pred_history_xor.csv")

    print(f"final accuracy was {model.score(kernelized_inputs, y)}")


def set_random_seed(seed=42):
    np.random.seed(seed)


def make_xor_dataset(n_samples):
    samples = _generate_elements_xor(n_samples)
    classes = _xor(samples)
    samples_with_classes = {"samples": samples, "classes": classes}
    return samples_with_classes


def _generate_elements_xor(n_samples=10):
    X = np.random.uniform(-1, 1, n_samples).reshape(-1, 1)
    Y = np.random.uniform(-1, 1, n_samples).reshape(-1, 1)
    samples = np.concatenate((X, Y), axis=1)
    return samples


def _xor(samples):
    x_samples = samples[:, 0]
    y_samples = samples[:, 1]
    x_greater_zero = x_samples > 0
    y_greater_zero = y_samples > 0

    result_xor = np.logical_xor(x_greater_zero, y_greater_zero)

    result_numeric = _convert_to_numeric(result_xor)

    return result_numeric


def _convert_to_numeric(array):
    return 2 * array.astype(np.float32).reshape(-1,) - 1


def generate_centers(samples, n_clusters=3, random_state=42):
    k_means = KMeans(n_clusters=n_clusters,
                     random_state=random_state).fit(samples)
    return k_means.cluster_centers_


def apply_rbf_kernel(samples, centers, gamma=0.1):
    result = rbf_kernel(X=samples, Y=centers, gamma=gamma)
    return result


def create_sgd_classifier(alpha=0.0, learning_rate=0.7):
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=alpha,
                          learning_rate="constant", eta0=learning_rate,
                          fit_intercept=True, shuffle=False)
    return model


def partial_weights_and_prediction(X, y, model, n_iter):
    classes = np.unique(y)

    weights_history = []
    pred_history = []

    for _ in range(n_iter):

        _perfome_one_epoch(model, X, y, classes)

        new_weights = obtain_weights_and_bias(model)
        new_pred = model.predict(X)

        weights_history.append(new_weights)
        pred_history.append(new_pred)

    return weights_history, pred_history


def _perfome_one_epoch(model, X, y, classes):
    model.partial_fit(X, y, classes=classes)


def obtain_weights_and_bias(model):
    return np.concatenate((model.intercept_, model.coef_[0]), axis=0)


def save_array_csv(array, file_name="data.csv", header=""):
    np.savetxt(file_name, array, delimiter=',',
               header=header, comments="")


def plot_dataset_2d(samples, classes):
    x = samples[:, 0]
    y = samples[:, 1]
    colors = classes

    plt.scatter(x, y, c=colors)
    plt.show()


def plot_dataset_3d(samples, classes):
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]
    colors = classes

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=colors)
    plt.show()


if __name__ == "__main__":
    generate_xor_experiment()
