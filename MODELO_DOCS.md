# Documentación técnica del modelo de predicción de demanda

Sistema de forecasting de demanda farmacéutica basado en XGBoost con pipeline de feature engineering, evaluación por holdout temporal y forecasting recursivo.

---

## 1. Arquitectura del pipeline

```
Dataset (CSV/XLSX)
    │
    ▼
StorageDatasetLoader        → Lee el archivo, detecta delimitador, normaliza columnas
    │
    ▼
DynamicTransformationStage  → Aplica el mapeo de columnas del usuario (product/quantity/date/...)
    │                          Filtra filas con fecha/cantidad nulas o cantidad < 0
    │                          Agrupa transacciones del mismo día (sum de quantity)
    ▼
HistoricalDataManager       → Fusiona con snapshots históricos de ejecuciones anteriores
    │                          Permite reentrenamiento acumulativo por botica
    ▼
FeatureEngineeringStage     → Resamplea a frecuencia diaria, rellena días sin venta con 0
    │                          Calcula features de calendario, lags y promedios móviles
    ▼
XGBoostTrainingStage        → Entrena un único modelo con todos los productos
    │                          OneHotEncoder para product_id + PassThrough para numéricos
    ▼
ForecastingStage            → Genera N días de predicción por producto (batch por paso)
    │
    ▼
MetricsCalculationStage     → Evalúa calidad sobre holdout temporal (último 20% de fechas)
```

---

## 2. Feature engineering aplicado

### 2.1 Features de calendario

| Feature | Descripción | Por qué importa en farmacia |
|---|---|---|
| `year` | Año de la fecha | Tendencias de largo plazo |
| `month` | Mes (1–12) | Estacionalidad mensual (ej: antigripales en invierno) |
| `day_of_week` | Día de la semana (0=lunes, 6=domingo) | Patrones semanales (más ventas ciertos días) |
| `week_of_year` | Semana del año (1–53) | Ciclos estacionales más granulares |
| `day_of_month` | Día del mes (1–31) | Ciclos de pago (quincena, fin de mes) |
| `is_weekend` | 1 si sábado o domingo, 0 si no | Diferencias en afluencia de clientes |
| `is_end_of_month` | 1 si día >= 25, 0 si no | Pico de demanda post-pago de sueldos |

### 2.2 Features de series de tiempo

Configuradas en `settings.lags = [1, 7, 14]` y `settings.rolling_windows = [7, 14]`.

| Feature | Descripción | Por qué importa |
|---|---|---|
| `lag_1` | Ventas del día anterior | Autocorrelación de corto plazo |
| `lag_7` | Ventas del mismo día hace 1 semana | Patrón semanal directo |
| `lag_14` | Ventas del mismo día hace 2 semanas | Confirma si el patrón semanal es recurrente |
| `rolling_mean_7` | Promedio de los últimos 7 días | Nivel de demanda reciente |
| `rolling_mean_14` | Promedio de los últimos 14 días | Tendencia de 2 semanas |
| `growth` | (hoy − ayer) / ayer | Si la demanda está acelerando o desacelerando |

### 2.3 Features opcionales (cuando se mapean en el request)

| Field key | Columna típica | Aporte al modelo |
|---|---|---|
| `price` | precio_unitario | Elasticidad precio-demanda |
| `stock` | stock_actual | Restricción de disponibilidad |
| `promotion` | promocion | Pico de demanda por oferta |
| `season` | temporada | Estacionalidad cualitativa |
| `holiday` | feriado | Días sin atención o mayor demanda |

---

## 3. Modelo XGBoost

```python
XGBRegressor(
    n_estimators=300,       # 300 árboles de decisión
    max_depth=3,            # Profundidad máxima por árbol
    learning_rate=0.05,     # Tasa de aprendizaje (shrinkage)
    subsample=0.9,          # 90% de filas por árbol (reduce sobreajuste)
    colsample_bytree=0.9,   # 90% de features por árbol
    objective="reg:squarederror",
    n_jobs=-1,
)
```

**Preprocesamiento:**
- `product_id`: One-Hot Encoding (cada producto → columnas binarias)
- Features numéricas: PassThrough (sin escalar, XGBoost no lo requiere)

---

## 4. Evaluación — Holdout temporal

### 4.1 Cómo se divide el dataset

```
Dataset completo (N fechas ordenadas cronológicamente)
│
├── Train set: primeras 80% fechas → modelo de evaluación
└── Test set:  últimas  20% fechas → se compara predicho vs real
```

**No es un split aleatorio.** Un split aleatorio en series de tiempo provocaría data leakage (el modelo vería el futuro como features de lag). El split siempre respeta el orden temporal.

### 4.2 Por qué hay dos modelos

| Modelo | Entrena con | Predice sobre | Para qué |
|---|---|---|---|
| Modelo de evaluación | 80% histórico | Últimas 20% fechas (reales) | Calcular WAPE/MAPE/RMSE/MAE |
| Modelo de producción | 100% histórico | N días futuros (no existen) | El forecast real devuelto al usuario |

Las métricas reflejan **qué tan bien el modelo habría predicho** si las últimas fechas históricas hubieran sido el futuro. Es un proxy de confianza, no una medición exacta de la predicción futura.

### 4.3 Fórmulas de las métricas

Sea `y` = valores reales, `ŷ` = valores predichos, `n` = número de observaciones.

**WAPE — Weighted Absolute Percentage Error** (métrica principal)
```
WAPE = Σ|y - ŷ| / Σ|y| × 100
```
Pondera el error por el volumen de ventas. Productos de alta demanda contribuyen más.
No explota cuando `y = 0`. Estándar en forecasting farmacéutico y de retail.

**MAPE — Mean Absolute Percentage Error**
```
MAPE = mean(|y - ŷ| / y) × 100   [excluyendo y ≈ 0]
```
Sensible a valores bajos: si `y = 1` y `ŷ = 2`, el error es 100%. Con datos dispersos (muchos días con 0 o 1 unidad vendida), MAPE tiende a valores muy altos que no reflejan la calidad real del modelo.

**MAE — Mean Absolute Error**
```
MAE = mean(|y - ŷ|)
```
Error promedio en unidades absolutas. Fácil de interpretar ("el modelo se equivoca en X unidades por día en promedio").

**RMSE — Root Mean Squared Error**
```
RMSE = sqrt(mean((y - ŷ)²))
```
Penaliza errores grandes más que los pequeños. Útil para detectar predicciones muy fuera de rango.

### 4.4 Métrica global (overall)

La métrica overall se calcula sobre **todos los productos concatenados** en el test set, no como promedio de las métricas individuales:

```python
# Todos los productos apilados en un solo array
actual    = test_frame["quantity"].to_numpy()   # [prod_A_dia1, prod_A_dia2, ..., prod_B_dia1, ...]
predicted = model.predict(test_features)        # misma forma

WAPE_global = Σ|actual - predicted| / Σ|actual| × 100
```

Esto es un **micro-promedio ponderado por volumen**: los productos de mayor venta dominan la métrica. Es el comportamiento correcto para farmacia — los medicamentos de alta rotación deben tener más peso que los esporádicos.

### 4.5 "Precisión Estimada" en el dashboard

```
Precisión Estimada = max(0%,  100% - WAPE)
```

Ejemplo: WAPE = 35% → Precisión = 65%

Se usa WAPE en lugar de MAPE porque:
1. MAPE es indefinido cuando `y = 0` (días sin ventas)
2. MAPE explota con ventas bajas (1 unidad real → cualquier diferencia da error enorme)
3. WAPE es estable y es el estándar de la industria farmacéutica

---

## 5. Horizonte de predicción

### 5.1 Relación entre datos disponibles y horizonte

El horizonte de predicción no afecta las métricas de evaluación (que son sobre el holdout histórico). Pero sí afecta **la calidad real del forecast**: cada día adicional usa como `lag_1` la predicción del día anterior, acumulando error.

**Regla práctica — ratio entrenamiento/horizonte:**

| Ratio | Evaluación | Ejemplo con 96 días de entrenamiento |
|---|---|---|
| ≥ 14× | Óptimo | Horizonte ≤ 7 días |
| 7–14× | Aceptable | Horizonte 7–14 días |
| 3–7× | Riesgoso | Horizonte 14–32 días |
| < 3× | No recomendado | Horizonte > 32 días |

### 5.2 Recomendaciones por tamaño del dataset

| Días históricos disponibles | Horizonte óptimo | Horizonte máximo razonable |
|---|---|---|
| < 90 días | 3–5 días | 7 días |
| 90–180 días | 7 días | 14 días |
| 180–365 días | 14 días | 30 días |
| 365–730 días (1–2 años) | 30 días | 60–90 días |
| > 730 días | 60–90 días | 180 días |

### 5.3 Opciones disponibles en el sistema

El sistema permite horizontes de 1 a 90 días (validado en el backend). La interfaz ofrece:

| Opción | Uso recomendado |
|---|---|
| **7 días** | Dataset < 6 meses. Predicción semanal de pedidos. |
| **14 días** | Dataset 6–12 meses. Planificación quincenal. |
| **30 días** | Dataset > 1 año. Planificación mensual de inventario. |

---

## 6. Preparación del dataset

### 6.1 Columna de cantidad — decisión crítica

La columna `quantity` determina qué se predice. Para datasets transaccionales de farmacia:

| Situación | Columna recomendada |
|---|---|
| Dataset solo tiene ventas por pack completo | `packs_completos` |
| Dataset tiene ventas fraccionadas (unidad suelta) | `unidades_fraccionadas` |
| **Dataset tiene ambas (caso típico de farmacia)** | **`packs_completos + unidades_fraccionadas`** |

Usar solo `packs_completos` cuando hay ventas fraccionadas puede ocultar el 50–80% de la demanda real (productos como Paracetamol, analgésicos, ansiolíticos suelen venderse por unidades sueltas).

### 6.2 Filas que no deben ir al modelo

| Tipo de fila | Razón | Solución |
|---|---|---|
| Transacciones anuladas (`anulada=True`) | Representan ventas que no ocurrieron | Filtrar antes de subir |
| Fechas futuras | Contaminarían el entrenamiento | El pipeline las rechaza (fecha > hoy se ignoraría) |
| Cantidad negativa | Errores de registro o devoluciones | El pipeline las filtra automáticamente (`quantity >= 0`) |
| Producto vacío o nulo | No puede asignarse a ninguna serie temporal | El pipeline los elimina |

El pipeline ML **sí limpia automáticamente** nulos, negativos y product_id vacío. Lo que **no limpia** (por no conocer la semántica del negocio) son: anuladas, devoluciones, y campos de negocio específicos.

### 6.3 Productos con muy pocos datos

Con datasets con muchos productos de baja rotación (1–5 ventas en todo el período):
- El modelo genera predicciones para todos los productos (incluyendo los esporádicos)
- Las métricas WAPE/MAPE se ven afectadas por estos productos
- Esto **no es un defecto del modelo** — refleja la dificultad inherente de predecir productos con demanda muy irregular

Para estos productos, el forecast es una estimación de tendencia basada en los pocos datos disponibles. La botica debe interpretar las predicciones de productos esporádicos como "demanda esperada baja", no como cero garantizado.

---

## 7. Acumulación histórica y reentrenamiento

Cada ejecución guarda un snapshot de los datos procesados en `almacen/historical-datasets/{datasetId}/`. En la siguiente ejecución, el pipeline:

1. Carga todos los snapshots históricos para esa botica
2. Los fusiona con los datos nuevos
3. Deduplica por `(date, product_id)` manteniendo el valor más reciente
4. Reentrena el modelo con el dataset combinado

Esto permite que el modelo mejore con el tiempo sin necesidad de re-subir datos históricos en cada predicción.

**`isRetrain: true`** en la respuesta indica que se usaron datos de ejecuciones anteriores.

---

## 8. Limitaciones conocidas

- **Datos < 90 días:** Los lags de 7 y 14 días tienen poco histórico. Las predicciones para horizontes > 7 días acumulan error significativo.
- **Productos nuevos:** Un producto sin historial solo puede basarse en features de calendario. La predicción será muy imprecisa hasta acumular al menos 2–4 semanas de datos.
- **Cambios estructurales de demanda:** El modelo asume que los patrones del pasado se mantienen. Pandemias, cambios de precio drásticos o nuevas regulaciones no se reflejan hasta que el nuevo comportamiento quede en los datos de entrenamiento.
- **Estacionalidad con < 1 año de datos:** El modelo no puede aprender ciclos anuales (verano/invierno) si solo tiene datos de unos pocos meses.
