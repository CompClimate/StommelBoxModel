import shap


def explain(model, X, n_features):
    features = [f'\(t - {i}\)' for i in reversed(range(1, n_features + 1))]
    
    e = shap.GradientExplainer(model, X)
    e.features = features
    shap_values = e(X)
    
    return shap_values


def heatmap(shap_values):
    shap.heatmap_plot(shap_values, show=False)
