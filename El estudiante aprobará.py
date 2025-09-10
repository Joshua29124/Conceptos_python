import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Datos de ejemplo
datos = {
    'horas_estudio': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'asistencia':    [30, 40, 50, 60, 65, 70, 75, 80, 90, 95],
    'aprobado':      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(datos)

# Variables
X = df[['horas_estudio', 'asistencia']]
y = df['aprobado']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear red neuronal
modelo = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predecir nuevo caso
nuevo = [[6, 85]]  # 6 horas de estudio y 85% asistencia
pred = modelo.predict(nuevo)[0]
prob = modelo.predict_proba(nuevo)[0][1]

print(f"Probabilidad de aprobar: {prob:.2f}")
print("Resultado:", "✅ Aprobó" if pred == 1 else "❌ No aprobó")
