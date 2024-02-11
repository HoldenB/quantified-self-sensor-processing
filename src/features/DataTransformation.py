# Referenced from:
# Mark Hoogendoorn and Burkhardt Funk (2017)
# Machine Learning for the Quantified Self (Springer, Ch. 4)
# ------------------------------------------------------------ #
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy


class DataTransformUtils:
    @staticmethod
    def normalize_dataset(data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max() - data_table[col].min()
            )

        return dt_norm


# This class removes the high frequency data (that might be considered noise) from the data.
# We can only apply this when we do not have missing values (i.e. NaN).
class LowPassFilter:

    @staticmethod
    def low_pass_filter(
        data_table,
        col,
        sampling_frequency,
        cutoff_frequency,
        order=5,
        phase_shift=True,
    ):
        # http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
        # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq

        # Apply preliminary butterworth filter
        b, a = butter(order, cut, btype="low", output="ba", analog=False)

        filter_func = filtfilt if phase_shift else lfilter
        data_table[col + "_lowpass"] = filter_func(b, a, data_table[col])

        return data_table


# Class for Principal Component Analysis. We can only apply this when we do not have missing values (i.e. NaN).
# For this we have to impute these first, be aware of this.
class PCA_Helper:

    def __init__(self):
        # latest cached PCA result
        self.pca = []

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data_table, cols):

        # Normalize the data first.
        dt_norm = DataTransformUtils.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_

    # Apply a PCA given the number of components we have selected.
    # We add new pca columns.
    def apply_pca(self, data_table, cols, number_comp):

        # Normalize the data first.
        dt_norm = DataTransformUtils.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        # And add the new ones:
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        return data_table
