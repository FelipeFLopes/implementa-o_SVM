import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from train_xor import (create_sgd_classifier, generate_centers,
                       apply_rbf_kernel, partial_weights_and_prediction,
                       save_array_csv)


def train_iris():
    iris = load_iris()

    iris_dataset = make_iris_dataset(iris)

    samples = iris_dataset["samples"]
    centers = generate_centers(samples, n_clusters=7)
    kernelized_inputs = apply_rbf_kernel(samples, centers, gamma=0.2)

    model = create_sgd_classifier(alpha=0.01, learning_rate=0.7)

    n_iter = 10
    y = iris_dataset["classes"]

    weights_history, pred_history = partial_weights_and_prediction(
        kernelized_inputs, y, model, n_iter)

    save_array_csv(centers, file_name="centers_iris.csv")
    save_array_csv(weights_history, file_name="weight_history_iris.csv")
    save_array_csv(pred_history, file_name="pred_history_iris.csv")

    print(f"O score final foi {model.score(kernelized_inputs, y)}")


def make_iris_dataset(dataset):
    target_names = dataset.target_names
    y = dataset.target
    samples = dataset.data

    samples, y = _remove_virginica(samples, y, target_names)

    normalizer = train_normalizer(samples)

    std_samples = _normalize(samples, normalizer)

    classes = 2 * y - 1

    samples_with_classes = {"samples": std_samples, "classes": classes}

    return samples_with_classes


def _remove_virginica(samples, y, target_names):

    virginica_index = np.where(target_names == "virginica")[0][0]

    different_from_virginica = y != virginica_index

    target_without_virginica = y[different_from_virginica].astype(np.float32)
    samples_without_virginica = samples[different_from_virginica].astype(
        np.float32)

    return samples_without_virginica, target_without_virginica


def train_normalizer(sample):
    scaler = StandardScaler()
    scaler.fit(sample)
    return scaler


def _normalize(sample, normalizer):
    return normalizer.transform(sample)


if __name__ == "__main__":
    train_iris()
