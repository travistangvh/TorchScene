import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=35)

# Average CV score on the training set was: 0.9895260981411951
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=0.01, fit_intercept=True, l1_ratio=0.5, learning_rate="constant", loss="squared_hinge", penalty="elasticnet", power_t=0.0)),
    ZeroCount(),
    XGBClassifier(learning_rate=0.5, max_depth=7, min_child_weight=13, n_estimators=100, n_jobs=1, subsample=0.15000000000000002, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 35)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
