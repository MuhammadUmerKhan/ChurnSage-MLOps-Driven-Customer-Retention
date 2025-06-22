import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, roc_curve


def Algorithm(model=None, X_train=None, X_test=None, y_train=None, y_test=None, params=None, save = False, name = None):
    gridsearch = GridSearchCV(model, param_grid=params, scoring='f1', cv=5)
    gridsearch.fit(X_train, y_train)
    
    best_model = gridsearch.best_estimator_
    best_params = gridsearch.best_params_

    if save:
        joblib.dump(best_model, f'{name}.joblib')
    
    y_scores = best_model.predict_proba(X_test)[:, 1]
    predictions = best_model.predict(X_test)
    
    print(f"Model Accuracy: {best_model.score(X_test, y_test) * 100:.2f}%")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_scores) * 100:.2f}%")
    
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    return model_evaluation(model=best_model, X_test=X_test, y_test=y_test), best_params

def model_evaluation(model=None, X_test=None, y_test=None):
    plt.figure(figsize=(4, 3))
    
    cm = confusion_matrix(y_true=y_test, y_pred=model.predict(X_test))
    names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cm, annot=labels, cmap='Blues', fmt='', cbar=False)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.show()

    print("-----------------------------------")
    print(classification_report(y_true=y_test, y_pred=model.predict(X_test)))