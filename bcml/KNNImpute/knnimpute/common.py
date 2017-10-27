# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function, division
import time

from six.moves import range
import numpy as np

from .normalized_distance import all_pairs_normalized_distances


def knn_initialize(X, missing_mask, verbose=False, distance=False):
    """
    Fill X with NaN values if necessary, construct the n_samples x n_samples
    distance matrix and set the self-distance of each row to infinity.
    """
    X_row_major = X.copy("C")
    if missing_mask.sum() != np.isnan(X_row_major).sum():
        # if the missing values have already been zero-filled need
        # to put NaN's back in the data matrix for the distances function
        X_row_major[missing_mask] = np.nan
    if distance is False:
        D = all_pairs_normalized_distances(X_row_major, verbose=verbose)
    else:
        D = distance
    # set diagonal of distance matrix to infinity since we don't want
    # points considering themselves as neighbors
    np.fill_diagonal(D, np.inf)
    #for i in range(X.shape[0]):
    #    D[i, i] = np.inf
    return X_row_major, D
