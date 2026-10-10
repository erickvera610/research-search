# Evidencia de validación automatizada

Se ejecutaron diez escenarios diferentes con datos ficticios o sintéticos. Este informe describe ejecuciones reales del código mediante un script automatizado. No representa tráfico de usuarios, experiencia laboral ni contribuciones realizadas manualmente por Erik.

Resultado: **10 de 10 escenarios aprobados**.

Consulta `validation-results.json` para ver cada entrada, resultado y fecha UTC de ejecución.

## Repetir las ejecuciones

Desde la carpeta del repositorio ejecuta:

```bash
python validate_scenarios.py
```

En Mercado Aprende instala primero las dependencias del README. En los otros cinco proyectos basta Python. El script utiliza almacenamiento temporal: no cambia registros personales ni realiza compras reales. Regenera el informe local; para conservar nuevos resultados en GitHub debes revisarlos y hacer un commit.

## Qué se comprobó

| Escenario | Entrada | Resultado |
|---|---|---|
| 1 | {"query": "machine learning validation"} | Aprobado |
| 2 | {"query": "SQLite"} | Aprobado |
| 3 | {"query": "logistic classification"} | Aprobado |
| 4 | {"query": "software tests"} | Aprobado |
| 5 | {"query": "Sobel edges"} | Aprobado |
| 6 | {"query": "graph routes"} | Aprobado |
| 7 | {"query": "data leakage"} | Aprobado |
| 8 | {"query": "REST APIs"} | Aprobado |
| 9 | {"query": "convolution"} | Aprobado |
| 10 | {"query": "unknownxyz"} | Aprobado |
