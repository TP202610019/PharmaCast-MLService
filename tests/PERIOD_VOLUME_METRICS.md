# Métricas de demanda acumulada

Se calculan sobre el mismo periodo de evaluación histórica de 30, 60 o 90 días que WAPE, para XGBoost y NaiveWeekly. Solo participan productos conocidos antes del corte. No miden todavía el resultado futuro ni el plan de compras con inventario.

Por producto: `actualTotal = sum(real)`, `predictedTotal = sum(predicción)`, `absoluteTotalError = abs(predictedTotal - actualTotal)`, `totalErrorPercentage = 100 * absoluteTotalError / actualTotal`.

Global: sumar los totales reales, los totales predichos y los errores absolutos de los productos; dividir la suma de errores absolutos por la suma de unidades reales. Así los excesos y faltantes de medicamentos diferentes no se cancelan. No se promedian porcentajes individuales.

El backend devuelve `volumeAccuracy = 100 - totalErrorPercentage`. Es un indicador definido para la app, no una probabilidad ni una garantía. Puede ser negativo si el error supera el 100%. El frontend solo redondea su presentación a un decimal.

Si el total real es cero, el porcentaje y su precisión son null. Las unidades predichas incorrectamente para productos sin ventas sí contribuyen al error absoluto global cuando existe demanda real en otros productos. Se conservan todos los días cero.

El WAPE diario existente no cambia. Las métricas antiguas permanecen null en las nuevas columnas y muestran un guion en precisión hasta una nueva evaluación.

Ejemplo: real [0,0,0,10], predicción [1,1,1,8] => WAPE diario 50%, error acumulado 10%, precisión volumétrica 90%.
