import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train_xor import (create_sgd_classifier, generate_centers,
                       apply_rbf_kernel, partial_weights_and_prediction,
                       save_array_csv, plot_confusion_matrix)

N_ITER = 10


def train_iris():
    iris = load_iris()

    iris_dataset = make_iris_dataset(iris)

    samples = iris_dataset["samples"]
    classes = iris_dataset["classes"]
    classes_labels = ["setosa", "versicolor"]

    samples_train, samples_test, classes_train, classes_test = train_test_split(
        samples, classes, test_size=0.33, random_state=42)

    centers = generate_centers(samples_train, n_clusters=7)
    kernelized_inputs_train = apply_rbf_kernel(samples_train, centers, gamma=5)
    kernelized_inputs_test = apply_rbf_kernel(samples_test, centers, gamma=5)

    model = create_sgd_classifier(alpha=0.001, learning_rate=0.7)

    weights_history, pred_history, svm_output_history = partial_weights_and_prediction(
        kernelized_inputs_train, classes_train, model, N_ITER)

    pred_test = model.predict(kernelized_inputs_test)
    svm_output_test = model.decision_function(kernelized_inputs_test)

    save_array_csv(centers, file_name="results/centers_iris_sw.csv")
    save_array_csv(weights_history,
                   file_name="results/weight_history_iris_sw.csv")
    save_array_csv(pred_history, file_name="results/pred_history_iris_sw.csv")
    save_array_csv(svm_output_history,
                   file_name="results/svm_output_history_iris_sw.csv")
    save_array_csv(pred_test, file_name="results/pred_test_iris_sw.csv")
    save_array_csv(svm_output_test,
                   file_name="results/svm_output_test_iris_sw.csv")

    cmf_train = confusion_matrix(classes_train, pred_history[-1])

    cmf_test = confusion_matrix(classes_test, pred_test)

    plot_confusion_matrix(cmf_train, classes_labels)
    plot_confusion_matrix(cmf_test, classes_labels)

    print(
        f"final train accuracy was {model.score(kernelized_inputs_train, classes_train)}")
    print(
        f"final test accuracy was {model.score(kernelized_inputs_test, classes_test)}")


def make_iris_dataset(dataset):
    target_names = dataset.target_names
    y = dataset.target
    samples = dataset.data

    samples, y = _remove_virginica(samples, y, target_names)

    normalizer = train_normalizer(samples)

    std_samples = _normalize(samples, normalizer)[:, :2]

    classes = 2 * y - 1

    samples_with_classes = {"samples": std_samples, "classes": classes}

    return samples_with_classes


def _remove_virginica(samples, y, target_names):

    virginica_index = np.where(target_names == "virginica")[0][0]

    different_from_virginica = y != virginica_index

    target_without_virginica = y[different_from_virginica].astype(np.float32)
    samples_without_virginica = samples[different_from_virginica].astype(
        np.float32)
    labels_without_virginica = target_names != virginica_index

    return samples_without_virginica, target_without_virginica


def train_normalizer(sample):
    scaler = StandardScaler()
    scaler.fit(sample)
    return scaler


def _normalize(sample, normalizer):
    return normalizer.transform(sample)


if __name__ == "__main__":
    train_iris()
