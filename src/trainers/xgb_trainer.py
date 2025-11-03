from typing import Any, Dict

import joblib
from xgboost import XGBClassifier

def train(X_train, y_train, X_val, y_val, config: Dict[str, Any], logger ) -> Dict[str, Any]:
    model_config = config["model"].get("params",{})
    logger.info("Building XGBClassifier with params: %s", model_config)

    classifier = XGBClassifier(
        n_estimators = model_config.get("n_estimators", 200),
        max_depth = model_config.get("max_depth", 6),
        learning_rate = model_config.get("learning_rate",0.05),
        use_label_encoder = False,
        eval_metric = model_config.get("eval_metric","logloss"),
    )

    logger.info("Fitting XGBoost model...")
    classifier.fit(X_train,
                   y_train,
                   eval_set = [(X_val, y_val)],
                   early_stopping_rounds = model_config.get("early_stopping_rounds",20),
                   verbose = False,
                   )
    
    # Compute validation probabilities for shared metrics
    try:
        val_probs = classifier.predict_proba(X_val)[:, 1]
    except Exception:
        logger.exception("Model does not support predict_proba; returning model only")
        val_probs = None
    
    result = {

        "model": classifier,
        "val_probs":val_probs,
        "metadata": {"num_boost_round": getattr(classifier,"best_iteration",None)},

    }

    return result