import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import KFold, LeavePGroupsOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def load_data():
    df = pd.read_csv("features_dataset.csv")
    X = df.drop(columns=["sujeito", "window", "groups"]).to_numpy()
    y = df["sujeito"].to_numpy()
    groups = df["groups"].to_numpy()
    return X, y, groups
    

def KFoldStratificationClassGroup(n_folds, groups, X, y):

    unique_groups = list(set(groups))
    unique_classes = list(set(y))

    train_indexes = []
    test_indexes = []

    for class_ in unique_classes:
        for group in unique_groups:

            indexes_to_split = np.array([
                i for i in range(len(groups))
                if y[i] == class_ and groups[i] == group
            ])

            if len(indexes_to_split) < n_folds:
                continue

            kf = KFold(n_splits=n_folds, shuffle=True, random_state=15)

            n_fold = 0
            for tr, te in kf.split(indexes_to_split):

                if len(train_indexes) < n_folds:
                    train_indexes.append(indexes_to_split[tr])
                    test_indexes.append(indexes_to_split[te])
                else:
                    train_indexes[n_fold] = np.append(train_indexes[n_fold], indexes_to_split[tr])
                    test_indexes[n_fold] = np.append(test_indexes[n_fold], indexes_to_split[te])

                n_fold += 1

    return train_indexes, test_indexes


def main():
    
    groups_selected = ["1", "2", "3","4"]
    n_groups_train = 3
    n_folds = 10

    X, y, groups = load_data()

    # Normalização e seleção de features
    preprocess_pipe = Pipeline([
        ("scaler", StandardScaler()),              # z-score
        ("fs", SelectKBest(mutual_info_classif, k=15))
    ])

    n_groups = len(set(groups))

    # Leave P Groups Out
    lpgo = LeavePGroupsOut(n_groups=n_groups - n_groups_train)

    i = 0
    for train_ind, test_ind in lpgo.split(X, y, groups):

        print(f"\nOuter Fold {i+1}/{lpgo.get_n_splits(groups=groups)}")

        print("Train groups:", Counter(groups[train_ind]))
        print("Train subjects:", Counter(y[train_ind]))

        print("Test groups:", Counter(groups[test_ind]))
        print("Test subjects:", Counter(y[test_ind]))

        i += 1

        # Inner folds 
        train_ind_folds, _ = KFoldStratificationClassGroup(
            n_folds, groups[train_ind], X[train_ind], y[train_ind]
        )

        _, test_ind_folds = KFoldStratificationClassGroup(
            n_folds, groups[test_ind], X[test_ind], y[test_ind]
        )

        for j in range(n_folds):
            print(f"\nInner Fold {j+1}/{n_folds}")

            train_ind_ = train_ind_folds[j]
            test_ind_  = test_ind_folds[j]

            X_train_fold = X[train_ind][train_ind_]
            y_train_fold = y[train_ind][train_ind_]

            X_test_fold  = X[test_ind][test_ind_]
            y_test_fold  = y[test_ind][test_ind_]

            # NORMALIZAÇÃO + SELEÇÃO (fit apenas no treino)
            X_train_sel = preprocess_pipe.fit_transform(
                X_train_fold, y_train_fold
            )
            X_test_sel = preprocess_pipe.transform(X_test_fold)

            print("Training set")
            print("\tSubjects:", Counter(y_train_fold))
            print("\tGroups:", Counter(groups[train_ind][train_ind_]))

            print("Test set")
            print("\tSubjects:", Counter(y_test_fold))
            print("\tGroups:", Counter(groups[test_ind][test_ind_]))

            # Modelos
            models = {
                "SVM Linear": SVC(kernel="linear", class_weight="balanced"),
                "SVM RBF": SVC(kernel="rbf", class_weight="balanced"),
                "KNN": KNeighborsClassifier(n_neighbors=5)
            }

            for name, model in models.items():
                # Treinar apenas com o fold corrente
                model.fit(X_train_sel, y_train_fold)
                y_train_pred = model.predict(X_train_sel)
                y_test_pred  = model.predict(X_test_sel)
                acc = accuracy_score(y_test_fold, y_test_pred)
                acc_train = accuracy_score(y_train_fold, y_train_pred)
                

                print(f"\n{name} - Training Confusion Matrix:\n", confusion_matrix(y_train_fold, y_train_pred))
                print(f"{name} - Test Confusion Matrix:\n", confusion_matrix(y_test_fold, y_test_pred))
                print(f"{name} - Training Accuracy: {acc_train:.4f}")
                print(f"{name} - Test Accuracy: {acc:.4f}")
                


if __name__ == "__main__":
    main()
