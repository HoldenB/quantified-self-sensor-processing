import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ClassificationAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix


# ------------------------------------------------------------ #
# rcParams
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# ------------------------------------------------------------ #
DATA_PATH = Path("../../data")
DATA_INTERIM_PATH = Path(DATA_PATH, "interim")
DATA_PKL_FILENAME_FEATURES_OUT = "01_75ms_feature_extract_out.pkl"


# ------------------------------------------------------------ #
# Helper function to first run a grid search on each classification algo
# then aggregate the accuracy results into an output score dataframe
def run_grid_search_on_models(
    model_orchestrator: ClassificationAlgorithms,
    X_train,
    X_test,
    y_train,
    possible_feature_sets,
    feature_set_names,
    iterations=1,
) -> pd.DataFrame:
    score_df = pd.DataFrame()

    for i, f in zip(range(len(possible_feature_sets)), feature_set_names):
        print("Feature set:", i)
        selected_train_X = X_train[possible_feature_sets[i]]
        selected_test_X = X_test[possible_feature_sets[i]]

        # First run non deterministic classifiers to average their score.
        performance_test_nn = 0
        performance_test_rf = 0

        for it in range(0, iterations):
            print("\tTraining neural network,", it)
            (
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = model_orchestrator.feedforward_neural_network(
                selected_train_X,
                y_train,
                selected_test_X,
                grid_search=False,
            )
            performance_test_nn += accuracy_score(y_test, class_test_y)

            print("\tTraining random forest,", it)
            (
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = model_orchestrator.random_forest(
                selected_train_X, y_train, selected_test_X, grid_search=True
            )
            performance_test_rf += accuracy_score(y_test, class_test_y)

        performance_test_nn = performance_test_nn / iterations
        performance_test_rf = performance_test_rf / iterations

        # And we run our deterministic classifiers:
        print("\tTraining KNN")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = model_orchestrator.k_nearest_neighbor(
            selected_train_X, y_train, selected_test_X, grid_search=True
        )
        performance_test_knn = accuracy_score(y_test, class_test_y)

        print("\tTraining decision tree")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = model_orchestrator.decision_tree(
            selected_train_X, y_train, selected_test_X, grid_search=True
        )
        performance_test_dt = accuracy_score(y_test, class_test_y)

        print("\tTraining naive bayes")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = model_orchestrator.naive_bayes(
            selected_train_X, y_train, selected_test_X
        )

        performance_test_nb = accuracy_score(y_test, class_test_y)

        # Save results to dataframe
        models = ["NN", "RF", "KNN", "DT", "NB"]
        new_scores = pd.DataFrame(
            {
                "model": models,
                "feature_set": f,
                "accuracy": [
                    performance_test_nn,
                    performance_test_rf,
                    performance_test_knn,
                    performance_test_dt,
                    performance_test_nb,
                ],
            }
        )
        score_df = pd.concat([score_df, new_scores])

    return score_df


def plot_confusion_matrix(cm: np.ndarray, classes) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)
    plt.show()


# ------------------------------------------------------------ #
# def main():
#     pass

# creating a training and test set
df: pd.DataFrame = pd.read_pickle(
    Path(DATA_INTERIM_PATH, DATA_PKL_FILENAME_FEATURES_OUT)
)

df.info()

# dropping cols we do not need -- need to keep the ex col though
df_train = df.drop(["participant", "set", "effort"], axis=1)

# dropping labels for training set
X = df_train.drop("ex", axis=1)
# labels only
y = df_train["ex"]

# training/test split
# stratify ensures that we see every label in the splits i.e
# an equal distribution of the labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

# comparing the test and train set to the total df_train
fix, ax = plt.subplots(figsize=(10, 5))
df_train["ex"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(
    kind="bar", ax=ax, color="dodgerblue", label="Train"
)
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# splitting feature subsets
origin_features = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]
square_features = ["accel_r", "gyro_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
# i.e accel_x_temp_mean_ws_6
time_features = [f for f in df_train.columns if "_temp_" in f]
# i.e accel_y_freq_0.0_Hz_ws_14, accel_z_pse
frequency_features = [
    f for f in df_train.columns if "_freq" in f or "_pse" in f
]
cluster_features = ["cluster"]

print(f"Time Features: {len(time_features)}")
print(f"Frequency Features: {len(frequency_features)}")

# creating feature-sets
fs_1 = list(set(origin_features))
fs_2 = list(set(fs_1 + square_features + pca_features))
fs_3 = list(set(fs_2 + time_features))
fs_4 = list(set(fs_3 + frequency_features + cluster_features))

# feature selection (forward feature selection) with simple decision tree
# loop over single features, starting small, in a forward selection to
# try individual features to check acc on scoring the labels
# highest acc feature, we'll start again with additional features + the best
# performing one
# over time we'll verify an increase in acc as we stack in additional features
# at which point we can check to see if acc declines as we hit a feature threshold
# i.e we see diminishing returns
# Occam's razor - simple model is better
model_orchestrator = ClassificationAlgorithms()

max_features = 10
# commenting out for now: this takes a lot of computational resources
# selected_features, ordered_features, ordered_scores = (
#     model_orchestrator.forward_selection(max_features, X_train, y_train)
# )

# result from forward_selection
selected_features = [
    "duration_s",
    "accel_y_freq_0.0_Hz_ws_14",
    "pca_2",
    "accel_x_freq_2.786_Hz_ws_14",
    "accel_y_freq_1.857_Hz_ws_14",
    "gyro_x_pse",
    "gyro_x_temp_mean_ws_6",
    "gyro_x",
    "gyro_r_pse",
    "gyro_z_max_freq",
]
# result from forward_selection
ordered_scores = [
    0.7122593718338399,
    0.9766970618034447,
    0.9918946301925026,
    0.9959473150962512,
    0.9979736575481256,
    0.9979736575481256,
    0.9979736575481256,
    0.9989868287740629,
    0.9989868287740629,
    0.9989868287740629,
]

# visualizing the acc vs # features
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1), ordered_scores)
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.show()

# using grid search to select the best hyperparams and model combinations
# we will use a 5-fold cross validation on the training set
possible_feature_sets = [fs_1, fs_2, fs_3, fs_4, selected_features]
feature_set_names = ["fs_1", "fs_2", "fs_3", "fs_4", "selected_fs"]

score_df = run_grid_search_on_models(
    model_orchestrator,
    X_train,
    X_test,
    y_train,
    possible_feature_sets,
    feature_set_names,
    iterations=1,
)

score_df.sort_values(by="accuracy", ascending=False)

# grouped bar plot to visualize the results
plt.figure(figsize=(10, 10))
# x/y/hue need to match the cols in the score_df
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

# selecting best model and evaluating the results - output was fairly close
# between RF - fs_4 & NN - fs_4
# we'll start with testing RF
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = model_orchestrator.random_forest(
    X_train[fs_4], y_train, X_test[fs_4], grid_search=True
)

# exploring the confusion matrix for the RF classifier
acc_rf = accuracy_score(y_test, class_test_y)
classes_rf = class_test_prob_y.columns
conf_mat_rf = confusion_matrix(y_test, class_test_y, labels=classes_rf)

plot_confusion_matrix(conf_mat_rf, classes_rf)

# subtracting participant A from the training data -- our goal
# is to train on the set minus A and then validate if the model
# can generalize to the data specific to participant A
# i.e. everyone does exercises differently and we need to ensure
# that the model can generalize across different participants
participant_df = df.drop(["set", "ex"], axis=1)

# ------------------------------------------------------------ #
# for now we'll comment out because of periodic
# runs in jupyter interactive
# if __name__ == "__main__":
#     main()
