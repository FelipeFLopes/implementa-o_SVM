import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier


def main():
    set_random_seed()

    s = generate_elements_xor(2000)

    xor_dataset = make_xor_dataset(s)

    centers = generate_centers(xor_dataset["samples"], n_clusters=7)

    kernelized_inputs = apply_rbf_kernel(
        xor_dataset["samples"], centers, gamma=5)

    xor_dataset["samples"] = kernelized_inputs

    X = xor_dataset["samples"]
    y = xor_dataset["classes"]

    model = _create_sgd_classifier()

    n_iter = 10

    weights_history, pred_history = partial_weights_and_prediction(
        xor_dataset["samples"], xor_dataset["classes"], model, n_iter)

    final_weights = obtain_weights_and_bias(model)

    save_array_csv(final_weights, file_name="centers.csv",
                   header="bias, dim1, dim2, dim3, dim4, dim5")

    save_array_csv(np.asarray(weights_history), file_name="weight_history.csv",
                   header="bias, dim1, dim2, dim3, dim4, dim5")

    print(f"final accuracy was {model.score(X, y)}")


def set_random_seed(seed=42):
    np.random.seed(seed)


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


def generate_elements_xor(n_samples=10):
    X = np.random.uniform(-1, 1, n_samples).reshape(-1, 1)
    Y = np.random.uniform(-1, 1, n_samples).reshape(-1, 1)
    samples = np.concatenate((X, Y), axis=1)
    return samples


def make_xor_dataset(samples):
    classes = _xor(samples)
    samples_with_classes = {"samples": samples, "classes": classes}
    return samples_with_classes


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


def _create_sgd_classifier():
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=0.2,
                          learning_rate="constant", eta0=0.4,
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


def save_array_csv(array, file_name="data.csv", header="A,B,C"):
    np.savetxt(file_name, array, delimiter=',',
               header=header, comments="")


if __name__ == "__main__":
    main()
