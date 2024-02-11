# Referenced from:
# Mark Hoogendoorn and Burkhardt Funk (2017)
# Machine Learning for the Quantified Self (Springer, Ch. 4)
# ------------------------------------------------------------ #
import numpy as np


# Class to abstract a history of numerical values we can use as an attribute.
class NumericalAbstraction:

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std')
    @staticmethod
    def aggregate(aggregation_function):
        # Compute the values and return the result.
        if aggregation_function == "mean":
            return np.mean
        elif aggregation_function == "max":
            return np.max
        elif aggregation_function == "min":
            return np.min
        elif aggregation_function == "median":
            return np.median
        elif aggregation_function == "std":
            return np.std
        else:
            return np.nan

    # Abstract numerical columns specified given a window size (i.e. the number of time points from
    # the past considered) and an aggregation function.
    @staticmethod
    def abstract(
        data_table,
        cols,
        window_size,
        aggregation_functions,
    ):
        # Create new columns for the temporal data, pass over the dataset and compute values
        for col in cols:
            for fn in aggregation_functions:
                data_table[col + "_temp_" + fn + "_ws_" + str(window_size)] = (
                    data_table[col]
                    .rolling(window_size)
                    .apply(NumericalAbstraction.aggregate(fn))
                )

        return data_table
