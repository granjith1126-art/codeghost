"""Classical Machine Learning Baseline for CodeGhost Risk Scoring (Milestone 4).

Implements Model A (Logistic Regression and Random Forest) with pure NumPy implementations
to guarantee 100% portability and prevent Windows Application Control / WDAC DLL load blocks.
Maps continuous probabilities to calibrated 1.0-10.0 risk scores with strict feature scaling.
"""

import json
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from backend.retrieval.schemas import RetrievedMatch
from backend.risk.features import CodeRiskFeatureExtractor, FEATURE_NAMES
from backend.risk.schemas import RiskCategory, RiskExample, RiskPrediction, RiskThresholds

logger = logging.getLogger(__name__)


def map_probability_to_score(prob: float) -> float:
    """
    Map probability in [0.0, 1.0] to a calibrated risk score in [1.0, 10.0].

    Formula: risk_score = 1.0 + 9.0 * prob
    - prob = 0.00 -> score = 1.0 (lowest risk)
    - prob = 0.50 -> score = 5.5 (medium risk)
    - prob = 1.00 -> score = 10.0 (critical risk)
    """
    clamped_prob = max(0.0, min(1.0, float(prob)))
    score = 1.0 + (9.0 * clamped_prob)
    return round(score, 1)


class NumpyStandardScaler:
    """Pure NumPy implementation of feature standardization (z = (x - u) / s)."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "NumpyStandardScaler":
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class NumpyLogisticRegression:
    """Pure NumPy regularized Logistic Regression with class-weight balancing."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        num_iterations: int = 500,
        l2_reg: float = 0.01,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.l2_reg = l2_reg
        self.class_weight = class_weight
        self.random_state = random_state
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.coef_: Optional[np.ndarray] = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow
        z = np.clip(z, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0.0

        # Calculate sample weights if balanced
        if self.class_weight == "balanced":
            pos_count = np.sum(y == 1)
            neg_count = np.sum(y == 0)
            w_pos = (n_samples / (2.0 * pos_count)) if pos_count > 0 else 1.0
            w_neg = (n_samples / (2.0 * neg_count)) if neg_count > 0 else 1.0
            sample_weights = np.where(y == 1, w_pos, w_neg)
        else:
            sample_weights = np.ones(n_samples)

        # Gradient descent with Adam-like momentum
        m_w = np.zeros(n_features)
        v_w = np.zeros(n_features)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.num_iterations + 1):
            linear = np.dot(X, self.weights) + self.bias
            preds = self._sigmoid(linear)
            error = (preds - y) * sample_weights

            grad_w = (np.dot(X.T, error) / n_samples) + (self.l2_reg * self.weights)
            grad_b = np.sum(error) / n_samples

            # Adam update
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
            m_hat = m_w / (1 - beta1 ** t)
            v_hat = v_w / (1 - beta2 ** t)

            self.weights -= (self.learning_rate * m_hat) / (np.sqrt(v_hat) + eps)
            self.bias -= self.learning_rate * grad_b

        self.coef_ = np.array([self.weights])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet.")
        linear = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear)


class _DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Probability of class 1

    @property
    def is_leaf(self):
        return self.value is not None


class NumpyDecisionTree:
    """Decision Tree Classifier building binary splits based on Gini impurity."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 4, max_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root: Optional[_DecisionTreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState) -> "NumpyDecisionTree":
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0, rng=rng)
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.RandomState) -> _DecisionTreeNode:
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Leaf conditions
        prob_1 = float(np.mean(y == 1)) if n_samples > 0 else 0.0
        if depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split:
            return _DecisionTreeNode(value=prob_1)

        # Feature subsampling
        m_feat = self.max_features or int(np.sqrt(n_feats))
        feat_indices = rng.choice(n_feats, size=min(m_feat, n_feats), replace=False)

        best_feat, best_thresh = None, None
        best_gini = 1.0

        current_gini = 1.0 - (prob_1 ** 2) - ((1.0 - prob_1) ** 2)

        for feat in feat_indices:
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 10:
                # Sample percentiles for speed
                thresholds = np.percentile(thresholds, np.linspace(10, 90, 8))

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_l, y_r = y[left_mask], y[right_mask]
                p_l = np.mean(y_l == 1)
                p_r = np.mean(y_r == 1)
                gini_l = 1.0 - (p_l ** 2) - ((1.0 - p_l) ** 2)
                gini_r = 1.0 - (p_r ** 2) - ((1.0 - p_r) ** 2)
                weighted_gini = (len(y_l) / n_samples) * gini_l + (len(y_r) / n_samples) * gini_r

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None or (current_gini - best_gini) < 1e-4:
            return _DecisionTreeNode(value=prob_1)

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1, rng)
        return _DecisionTreeNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _predict_row(self, row: np.ndarray, node: _DecisionTreeNode) -> float:
        if node.is_leaf:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.root) for row in X], dtype=np.float32)


class NumpyRandomForest:
    """Pure NumPy Random Forest ensemble classifier with out-of-bag feature importances."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 5,
        min_samples_split: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees: List[NumpyDecisionTree] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyRandomForest":
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.trees = []
        feat_counts = np.zeros(n_features)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
            tree = NumpyDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X[boot_indices], y[boot_indices], rng)
            self.trees.append(tree)

            # Accumulate used features for importance estimation
            def _accumulate_feats(node: _DecisionTreeNode):
                if node and not node.is_leaf:
                    feat_counts[node.feature] += 1
                    _accumulate_feats(node.left)
                    _accumulate_feats(node.right)

            _accumulate_feats(tree.root)

        total_nodes = np.sum(feat_counts)
        self.feature_importances_ = feat_counts / total_nodes if total_nodes > 0 else np.ones(n_features) / n_features
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Forest has no trees fitted.")
        all_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_preds, axis=0)


class BaselineRiskModel:
    """Model A: Classical ML risk scoring model trained on extracted AST and retrieval features."""

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42,
        thresholds: Optional[RiskThresholds] = None,
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.thresholds = thresholds or RiskThresholds()
        self.scaler = NumpyStandardScaler()
        self.feature_extractor = CodeRiskFeatureExtractor()
        self.feature_names = FEATURE_NAMES
        self.is_fitted = False

        if self.model_type in ("logistic_regression", "lr"):
            self.model = NumpyLogisticRegression(random_state=self.random_state)
            self.name = "LogisticRegression"
        elif self.model_type in ("random_forest", "rf", "gradient_boosting", "gb"):
            self.model = NumpyRandomForest(n_estimators=50, max_depth=5, random_state=self.random_state)
            self.name = "RandomForest"
        else:
            raise ValueError(f"Unsupported model type '{model_type}'. Choose 'logistic_regression' or 'random_forest'.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaselineRiskModel":
        if len(X_train) == 0:
            raise ValueError("Training dataset is empty.")

        # Fit scaler strictly on training set to prevent data leakage
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True

        logger.info(
            "Trained %s on %d samples (positive: %d, negative: %d)",
            self.name,
            len(X_train),
            int(np.sum(y_train == 1)),
            int(np.sum(y_train == 0)),
        )
        return self

    def fit_examples(
        self,
        train_examples: List[RiskExample],
        historical_stats: Optional[Dict[str, Any]] = None,
    ) -> "BaselineRiskModel":
        X_train, y_train, _ = self.feature_extractor.extract_dataset_matrix(
            train_examples, historical_stats=historical_stats
        )
        return self.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict_example(
        self,
        example: RiskExample,
        historical_stats: Optional[Dict[str, Any]] = None,
        decision_threshold: float = 0.5,
    ) -> RiskPrediction:
        feat_dict = self.feature_extractor.extract_from_example(example, historical_stats=historical_stats)
        X_vec = np.array([self.feature_extractor.features_to_vector(feat_dict)], dtype=np.float32)
        prob = float(self.predict_proba(X_vec)[0])
        score = map_probability_to_score(prob)
        category = self.thresholds.categorize(score)
        is_risky = prob >= decision_threshold

        feature_importances = self.get_feature_importances()

        return RiskPrediction(
            function_name=example.function_name,
            file_path=example.file_path,
            risk_probability=round(prob, 4),
            risk_score=score,
            risk_category=category,
            is_risky=is_risky,
            model_name=self.name,
            features_used={k: round(v, 4) for k, v in feat_dict.items()},
            top_retrieved_matches=example.retrieved_matches[:3],
            explanation_context={
                "similarity_scores": example.similarity_scores[:3],
                "top_feature_importances": {
                    k: round(v, 4) for k, v in list(feature_importances.items())[:5]
                },
                "commit_sha": example.commit_sha,
                "timestamp": example.timestamp,
            },
            reasoning_summary=f"Predicted risk score {score}/10 ({category.value}) based on {self.name} classifier.",
        )

    def predict_function(
        self,
        code: str,
        function_name: str,
        file_path: str,
        language: str = "python",
        retrieved_matches: Optional[List[RetrievedMatch]] = None,
        historical_stats: Optional[Dict[str, Any]] = None,
        decision_threshold: float = 0.5,
    ) -> RiskPrediction:
        retrieved_matches = retrieved_matches or []
        feat_dict = self.feature_extractor.extract_features(
            code=code,
            language=language,
            retrieved_matches=retrieved_matches,
            function_name=function_name,
            file_path=file_path,
            historical_stats=historical_stats,
        )
        X_vec = np.array([self.feature_extractor.features_to_vector(feat_dict)], dtype=np.float32)
        prob = float(self.predict_proba(X_vec)[0])
        score = map_probability_to_score(prob)
        category = self.thresholds.categorize(score)
        is_risky = prob >= decision_threshold

        feature_importances = self.get_feature_importances()
        sims = [m.similarity_score for m in retrieved_matches]

        return RiskPrediction(
            function_name=function_name,
            file_path=file_path,
            risk_probability=round(prob, 4),
            risk_score=score,
            risk_category=category,
            is_risky=is_risky,
            model_name=self.name,
            features_used={k: round(v, 4) for k, v in feat_dict.items()},
            top_retrieved_matches=retrieved_matches[:3],
            explanation_context={
                "similarity_scores": sims[:3],
                "top_feature_importances": {
                    k: round(v, 4) for k, v in list(feature_importances.items())[:5]
                },
            },
            reasoning_summary=f"Predicted risk score {score}/10 ({category.value}) using {self.name}.",
        )

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {name: 0.0 for name in self.feature_names}

        if hasattr(self.model, "feature_importances_") and self.model.feature_importances_ is not None:
            weights = self.model.feature_importances_
        elif hasattr(self.model, "coef_") and self.model.coef_ is not None:
            weights = self.model.coef_[0]
        else:
            weights = np.zeros(len(self.feature_names))

        sorted_pairs = sorted(
            zip(self.feature_names, weights),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return {name: float(weight) for name, weight in sorted_pairs}

    def save(self, file_path: Union[str, Path]) -> str:
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "name": self.name,
            "feature_names": self.feature_names,
            "thresholds": self.thresholds.model_dump(),
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        logger.info("Saved %s model bundle to %s", self.name, path)
        return str(path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "BaselineRiskModel":
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        thresholds = RiskThresholds.model_validate(bundle["thresholds"])
        instance = cls(
            model_type=bundle["model_type"],
            thresholds=thresholds,
        )
        instance.model = bundle["model"]
        instance.scaler = bundle["scaler"]
        instance.name = bundle["name"]
        instance.feature_names = bundle["feature_names"]
        instance.is_fitted = bundle["is_fitted"]
        logger.info("Loaded %s model from %s", instance.name, path)
        return instance
