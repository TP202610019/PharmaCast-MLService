# Métricas de demanda acumulada

Se calculan sobre el mismo periodo de evaluación histórica de 30, 60 o 90 días que WAPE, para XGBoost y NaiveWeekly. Solo participan productos conocidos antes del corte. No miden todavía el resultado futuro ni el plan de compras con inventario.

Por producto: `actualTotal = sum(real)`, `predictedTotal = sum(predicción)`, `absoluteTotalError = abs(predictedTotal - actualTotal)`, `base = max(actualTotal, predictedTotal)`, `totalErrorPercentage = 100 * absoluteTotalError / base`.

Global: sumar los errores absolutos de los productos y dividir por la suma de sus bases `max(actualTotal, predictedTotal)`. Así los excesos y faltantes de medicamentos diferentes no se cancelan. No se promedian porcentajes individuales. Dentro de cada producto sí se compensan los errores diarios: se evalúa exclusivamente el total del periodo.

El backend devuelve `volumeAccuracy = 100 - totalErrorPercentage`. Es un indicador de ajuste volumétrico definido para la app, no una probabilidad ni una garantía de precisión futura. Para cantidades no negativas está entre 0 y 100%. El frontend solo redondea su presentación a un decimal.

Real cero y previsto positivo: error porcentual 100% y precisión 0% a nivel de producto; esas unidades contribuyen al error y a la base global. Real y previsto ambos cero: por convención, error porcentual 0% y ajuste 100% a nivel de producto; esto expresa coincidencia sin demanda, no evidencia de capacidad predictiva. Si todas las bases globales son cero se aplica la misma convención. Una evaluación ausente o insuficiente conserva sus métricas ausentes; no se transforma en 100%. Se conservan todos los días cero.

El WAPE diario existente no cambia. Las ejecuciones anteriores conservan los valores almacenados con su fórmula anterior; esta fórmula se aplica a nuevas evaluaciones. No hace falta cambiar el esquema de la base de datos.

Ejemplo: real [0,0,0,10], predicción [1,1,1,8] => WAPE diario 50%, error acumulado normalizado 9.09%, precisión volumétrica 90.91%.
