import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from train_xor import build_confusion_matrix_plot


classes_train_xor = np.loadtxt("results/classes_train_xor.csv", delimiter=',')

classes_test_xor = np.loadtxt("results/classes_test_xor.csv", delimiter=',')

unique_classes = np.unique(classes_train_xor)

pred_train_xor_sw = np.loadtxt(
    "results/pred_history_xor_sw.csv", delimiter=',')[-1]

pred_train_xor_hw_bin_float = np.loadtxt(
    "results/pred_history_train_xor_hw_bin_float.csv", delimiter=',')[-1]

pred_train_xor_hw_bin_32 = np.loadtxt(
    "results/pred_history_train_xor_hw_bin_32.csv", delimiter=',')[-1]

pred_train_xor_hw_bin_16 = np.loadtxt(
    "results/pred_history_train_xor_hw_bin_16.csv", delimiter=',')[-1]

pred_train_xor_hw_sc_8 = np.loadtxt(
    "results/pred_train_xor_hw_sc_8.csv", delimiter=',')

pred_test_xor_sw = np.loadtxt("results/pred_test_xor_sw.csv", delimiter=',')

pred_test_xor_hw_bin_float = np.loadtxt(
    "results/pred_test_xor_hw_bin_float.csv", delimiter=',')

pred_test_xor_hw_bin_32 = np.loadtxt(
    "results/pred_test_xor_hw_bin_32.csv", delimiter=',')

pred_test_xor_hw_bin_16 = np.loadtxt(
    "results/pred_test_xor_hw_bin_16.csv", delimiter=',')

pred_test_xor_hw_sc_8 = np.loadtxt(
    "results/pred_test_xor_hw_sc_8.csv", delimiter=',')


train_sw = confusion_matrix(classes_train_xor, pred_train_xor_sw)
train_bin_float = confusion_matrix(
    classes_train_xor, pred_train_xor_hw_bin_float)
train_bin_32 = confusion_matrix(classes_train_xor, pred_train_xor_hw_bin_32)
train_bin_16 = confusion_matrix(classes_train_xor, pred_train_xor_hw_bin_16)
train_sc_8 = confusion_matrix(classes_train_xor, pred_train_xor_hw_sc_8)

test_sw = confusion_matrix(classes_test_xor, pred_test_xor_sw)
test_bin_float = confusion_matrix(classes_test_xor, pred_test_xor_hw_bin_float)
test_bin_32 = confusion_matrix(classes_test_xor, pred_test_xor_hw_bin_32)
test_bin_16 = confusion_matrix(classes_test_xor, pred_test_xor_hw_bin_16)
test_sc_8 = confusion_matrix(classes_test_xor, pred_test_xor_hw_sc_8)


fig = build_confusion_matrix_plot(train_sw, unique_classes)
plt.savefig('results/cfm_train_sw.eps', dpi=400,
            transparent=True, bbox_inches='tight')
fig = build_confusion_matrix_plot(train_bin_float, unique_classes)
plt.savefig('results/cfm_train_bin_float.eps', dpi=400,
            transparent=True, bbox_inches='tight')
fig = build_confusion_matrix_plot(train_bin_32, unique_classes)
plt.savefig('results/cfm_train_bin_32.eps', dpi=400,
            transparent=True, bbox_inches='tight')
fig = build_confusion_matrix_plot(train_bin_16, unique_classes)
plt.savefig('results/cfm_train_bin_16.eps', dpi=400,
            transparent=True, bbox_inches='tight')
fig = build_confusion_matrix_plot(train_sc_8, unique_classes)
plt.savefig('results/cfm_train_sc_8.eps', dpi=400,
            transparent=True, bbox_inches='tight')

fig = build_confusion_matrix_plot(test_sw, unique_classes)
plt.savefig('results/cfm_test_sw.eps', dpi=400,
            transparent=True, bbox_inches='tight')
fig = build_confusion_matrix_plot(test_bin_float, unique_classes)
plt.savefig('results/cfm_test_bin_float.eps', dpi=400,
            transparent=True, bbox_inches='tight')
fig = build_confusion_matrix_plot(test_bin_32, unique_classes)
plt.savefig('results/cfm_test_bin_32.eps', dpi=400,
            transparent=True, bbox_inches='tight')
fig = build_confusion_matrix_plot(test_bin_16, unique_classes)
plt.savefig('results/cfm_test_bin_16.eps', dpi=400,
            transparent=True, bbox_inches='tight')
fig = build_confusion_matrix_plot(test_sc_8, unique_classes)
plt.savefig('results/cfm_test_sc_8.eps', dpi=400,
            transparent=True, bbox_inches='tight')
