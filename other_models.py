
import numpy as np
import pandas as pd
import seaborn as sn 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# parameters: x_train, y_train, x_test, y_test

# plt.style.use("ggplot")


def norm_cm(cm):
	samples_cls = np.sum(cm, axis=0)
	cm = cm / samples_cls
	# cm = cm.T / samples_cls
	# cm = cm.T

	return cm


def save_heatmap(cm, model_name, data_tag):
	# samples_cls = np.sum(cm, axis=0)
	# cm = cm.T / samples_cls
	# cm = cm.T
	df_cm = pd.DataFrame(cm, range(10), range(10))
	#plt.figure(figsize=(12,8))
	# sn.set(font_scale=1.4)
	ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 6}, cmap=sn.cm.rocket_r)
	ax.set_xlabel("True label")
	ax.set_ylabel("Predicted label")
	plt.savefig(f"{model_name}_confusion_{data_tag}.eps")
	plt.close()

def np2latex(m):

	# return "\\hline\n" + " \\\\\n\\hline\n".join([" & ".join(map(str,line)) for line in m]) + " \\\\\n\\hline"
	return "\\\\".join([" & ".join(map(str,[round(c, 4) for c in line])) for line in m])

def lg(x_train, y_train, x_test, y_test, model, data_tag):
	print("Showing logistic regression result...")

	lg = LR()
	lg.fit(x_train, y_train)
	y_pred = lg.predict(x_test)

	# print(lg.get_params())
	# print(lg.coef_)

	acc = accuracy_score(y_pred, y_test)
	print(f"acc: {acc}")

	cm = confusion_matrix(y_pred, y_test)
	print(cm)
	cm = norm_cm(cm)
	print(np2latex(cm))

	return []

def lda(x_train, y_train, x_test, y_test, model, data_tag):
	print("Showing LDA result...")

	lda = LDA()
	lda.fit(x_train, y_train)
	y_pred = lda.predict(x_test)

	# print(lda.get_params())
	# print(lda.coef_)

	acc = accuracy_score(y_pred, y_test)
	print(f"acc: {acc}")

	cm = confusion_matrix(y_pred, y_test)
	print(cm)
	cm = norm_cm(cm)
	print(np2latex(cm))

	if data_tag == "mnist":
		save_heatmap(cm, "lda", data_tag)

	return []

def qda(x_train, y_train, x_test, y_test, model, data_tag):
	print("Showing QDA result...")

	qda = QDA()
	qda.fit(x_train, y_train)
	y_pred = qda.predict(x_test)

	# print(qda.get_params())
	# # print(qda.coef_)

	acc = accuracy_score(y_pred, y_test)
	print(f"acc: {acc}")

	cm = confusion_matrix(y_pred, y_test)
	print(cm)
	cm = norm_cm(cm)
	print(np2latex(cm))

	if data_tag == "mnist":
		save_heatmap(cm, "qda", data_tag)

	return []
	
def svm(x_train, y_train, x_test, y_test, model, data_tag):
	print("Showing SVM result...")

	svc = SVC(kernel="linear")
	svc.fit(x_train, y_train)
	y_pred = svc.predict(x_test)

	# print(svc.get_params())
	# print(svc.coef_)

	acc = accuracy_score(y_pred, y_test)
	print(f"acc: {acc}")

	cm = confusion_matrix(y_pred, y_test)
	print(cm)
	cm = norm_cm(cm)
	print(np2latex(cm))

	print("Showing KSVM result...")
	# rbf kernel 
	ksvc = SVC(kernel="rbf")
	ksvc.fit(x_train, y_train)
	y_pred = ksvc.predict(x_test)

	# print(ksvc.get_params())
	# # print(ksvc.coef_)

	acc = accuracy_score(y_pred, y_test)
	print(f"acc: {acc}")

	cm = confusion_matrix(y_pred, y_test)
	print(cm)
	cm = norm_cm(cm)
	print(np2latex(cm))

	return []

def compare_all(dataset, model, data_tag, work_magic_code, magic_code, time_stamp, is_bin=False):
	x_train, y_train, x_test, y_test = dataset


	y_train = np.argmax(y_train, axis=1)
	y_test = np.argmax(y_test, axis=1)

	# calculate the metrics of ECA
	print("Showing ECA result...")

	y_pred = model.predict(x_test)
	y_pred = np.argmax(y_pred, axis=1)

	acc = accuracy_score(y_pred, y_test)
	print(f"acc: {acc}")

	cm = confusion_matrix(y_pred, y_test)
	print(cm)	
	cm = norm_cm(cm)
	print(np2latex(cm))

	if data_tag == "mnist":
		save_heatmap(cm, "eca", data_tag)
		
	if data_tag.startswith("imdb"):
		return

	x_train = x_train.reshape(x_train.shape[0], -1)
	x_test = x_test.reshape(x_test.shape[0], -1)

	# other models
	lg_res = []
	if is_bin:
		lg_res = lg(x_train, y_train, x_test, y_test, model, data_tag)
	lda_res = lda(x_train, y_train, x_test, y_test, model, data_tag)
	qda_res = qda(x_train, y_train, x_test, y_test, model, data_tag)
	svm_res = []
	if is_bin:
		svm_res = svm(x_train, y_train, x_test, y_test, model, data_tag)

	# sort out the res

	return []



def save_compare_result(res, data_tag, work_magic_code, magic_code, time_stamp):
	pass
