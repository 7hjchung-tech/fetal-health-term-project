from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_multinomial_logistic(C=1.0, random_state=42):
    return LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        penalty='l2',
        C=C,
        max_iter=1000,
        random_state=random_state
    )

def get_xgb_multiclass(num_class=3, random_state=42, **kwargs):
    return XGBClassifier(
        objective='multi:softprob',
        num_class=num_class,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=random_state,
        **kwargs
    )
