# Swagger local para pruebas cruzadas

Ejecutar desde el repositorio de ML con el entorno Python del proyecto:

    python -m uvicorn app.qa:app --host 127.0.0.1 --port 8101

Abrir http://127.0.0.1:8101/docs. Las operaciones llaman al código real con datos sintéticos. Cada evaluador puede ejecutarlo en su equipo o acceder mediante un túnel autenticado a un entorno QA.

| Caso | Pasos y evidencia |
|---|---|
| CP183 | Ejecutar Variables temporales con el ejemplo. Para 29-dic, A tiene lag_1=10 y B tiene lag_1=100. Cambiar sus cantidades del 29-dic y posteriores y repetir. Los lags del 29-dic conservan el valor. Los días sin ventas se rellenan con cero. |
| CP184 | Ejecutar Variables temporales. Para 28-dic, comprobar mes 12, semana ISO 52, día de semana 6. Para 29-dic, comprobar 12, 1, 0. Para 1-ene, comprobar 1, 1, 3. La numeración empieza en lunes=0. Esta operación comprueba el generador diario. |
| CP189 | Ejecutar Métricas con los pares del ejemplo. Esperar MAE=15, RMSE≈15.8114 y MAPE=10%. Para cerrar el caso, comprobar también la persistencia y lectura mediante el backend en QA. |
| CP186 y CP188 | Entrenar en QA y recuperar el artefacto. Para CP188, inducir un fallo de entrenamiento en una prueba interna con almacenamiento aislado y volver a cargar el artefacto previo. |
| CP185 y CP187 | Ejecutar con historial insuficiente. Verificar por separado la advertencia de evaluación y el rechazo antes de publicar un modelo cuando el caso exige cancelar el entrenamiento. |
| CP190 | Ejecutar Evaluate Weekly Forecast con ventas sintéticas suficientes. Enviar max_wape_percent por debajo y luego igual o por encima del WAPE global obtenido. Verificar LOW_FORECAST_PERFORMANCE solo cuando el WAPE excede el límite. El servicio normal lee MAX_EVALUATION_WAPE_PERCENT (20 por defecto). |
| CP169 | Alinear el caso con el flujo actual, que entrena antes de pronosticar, o definir una operación de inferencia separada. |
| CP191 | Corresponde al backend. Consultar la predicción guardada y comparar las métricas y su identificador. |
| CP192–CP195 | Corresponden al backend y a la vista de evaluación de compras. Probar archivos, brechas, clasificación y relectura de evaluaciones guardadas allí. |

## Local evidence commands

    python -m unittest discover -s tests -p test_temporal_acceptance.py -v
    python -m unittest tests.test_real_pipeline.RealPipelineTests.test_training_failure_preserves_readable_previous_artifact -v
    python -m unittest tests.test_real_pipeline.RealPipelineTests.test_known_metrics_are_saved_in_execution_metadata -v

The metrics test writes and reads the execution's model metadata. PostgreSQL
metrics are owned by the application backend and can be checked via the
prediction detail endpoint.

## Save and read synthetic metrics in Swagger

Execute **Save Error Metrics** with actual [100, 200] and predicted [110, 180].
Copy the returned executionId into **Read Saved Error Metrics**, then verify
MAE 15, RMSE approximately 15.8114 and MAPE 10%. The QA instance stores its last
500 evaluations in SQLite. Set ML_QA_METRICS_DB to a writable file on a QA
volume if records must survive container replacement. This tests the QA
workflow. The application uses its PostgreSQL prediction records separately.
