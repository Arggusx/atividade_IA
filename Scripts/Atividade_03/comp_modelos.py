import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)

# Importa resultados já calculados
from regresLinear import linear_results
from regresLogistica import logistica_results

# Extrai dados da regressão linear
y_test_lin = linear_results["y_test"]
y_pred_cont = linear_results["y_pred_cont"]   # valores contínuos
y_pred_lin_class = linear_results["y_pred_class"]  # valores 0/1

# Extrai dados da regressão logística
y_test_log = logistica_results["y_test"]
y_pred_log_class = logistica_results["y_pred_class"]
y_prob_log = logistica_results["y_prob"]  # probabilidades para ROC

# ============================
# MÉTRICAS REGRESSÃO LINEAR
# ============================

mae = mean_absolute_error(y_test_lin, y_pred_cont)
mse = mean_squared_error(y_test_lin, y_pred_cont)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_lin, y_pred_cont)

# ============================
# MÉTRICAS REGRESSÃO LOGÍSTICA
# ============================

acc = accuracy_score(y_test_log, y_pred_log_class)
prec = precision_score(y_test_log, y_pred_log_class)
rec = recall_score(y_test_log, y_pred_log_class)
f1 = f1_score(y_test_log, y_pred_log_class)
auc_roc = roc_auc_score(y_test_log, y_prob_log)

# ============================
# TABELA DE COMPARAÇÃO
# ============================

comparacao = pd.DataFrame({
    "Tarefa": ["Regressão Linear (contínuo)", "Classificação (logística)"],
    "Métrica 1": [f"MAE = {mae:.4f}", f"Acurácia = {acc:.4f}"],
    "Métrica 2": [f"RMSE = {rmse:.4f}", f"Precision = {prec:.4f}"],
    "Métrica 3": [f"R² = {r2:.4f}", f"Recall = {rec:.4f}"],
    "Métrica 4": ["-", f"F1 = {f1:.4f}"],
    "Métrica 5": ["-", f"AUC-ROC = {auc_roc:.4f}"]
})

print("\n===== Comparação entre Regressão e Classificação =====\n")
print(comparacao.to_string(index=False))
