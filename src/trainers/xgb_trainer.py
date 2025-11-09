
from typing import Any, Dict, Tuple
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

def train(X_train, y_train, X_val, y_val, config: Dict[str, Any], logger) -> Tuple[Any, Dict[str, Any]]:
    """
    Standard trainer entrypoint for XGBoost models.
    
    """
    model_config = config.get("model", {}).get("params", {})
    logger.info("Building XGBClassifier with params: %s", model_config)

    classifier = XGBClassifier(
        n_estimators=model_config.get("n_estimators", 200),
        max_depth=model_config.get("max_depth", 6),
        learning_rate=model_config.get("learning_rate", 0.05),
        subsample=model_config.get("subsample", 0.8),
        colsample_bytree=model_config.get("colsample_bytree", 0.8),
        gamma=model_config.get("gamma", 0),
        reg_alpha=model_config.get("reg_alpha", 0.0),
        use_label_encoder=False,
        eval_metric=model_config.get("eval_metric", "logloss"),
        #enable_categorical=True,
        random_state=model_config.get("random_state", 42),
        n_jobs=-1
    )

    logger.info("Fitting XGBoost model...")
    classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=model_config.get("early_stopping_rounds", 20),
        verbose=False,
    )

    # Validation metrics
    try:
        val_probs = classifier.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_probs)
    except Exception as e:
        logger.warning("Failed to compute PR-AUC: %s", e)
        val_probs = None
        pr_auc = None

    metrics = {
        "val_pr_auc": pr_auc,
        "best_iteration": getattr(classifier, "best_iteration", None),
    }

    logger.info("Training complete. PR-AUC=%.4f", pr_auc if pr_auc is not None else float("nan"))

    return classifier, metrics
