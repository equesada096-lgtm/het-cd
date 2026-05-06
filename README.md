# HET-CD · Herramienta de Evaluación Técnica del Complemento de Destino

Aplicación Streamlit para configurar puestos, evaluar puestos existentes y ejecutar el análisis sistemático de niveles de complemento de destino sobre la RPT.

## 1. Contenido del repositorio

```text
app.py                            # interfaz Streamlit
het_cd_engine.py                  # motor de cálculo HET-CD
het_cd_batch.py                   # análisis masivo por consola
requirements.txt                  # dependencias Python
data/het_cd_classifier_data.xlsx  # archivo base de datos HET-CD
docs/manual_tecnico_het_cd.docx   # manual técnico completo
docs/guia_rapida_het_cd.docx      # guía rápida de operación
outputs/.gitkeep                  # carpeta local de salida
```

## 2. Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

La aplicación carga por defecto `data/het_cd_classifier_data.xlsx` y permite sustituirlo desde la barra lateral por una versión actualizada del archivo de datos HET-CD.

## 3. Ejecución batch

```bash
python het_cd_batch.py data/het_cd_classifier_data.xlsx
```

La salida se genera en `resultados_het_cd_batch/`.

## 4. Despliegue en Streamlit Community Cloud

1. Crear un repositorio privado en GitHub.
2. Subir todo el contenido de esta carpeta al repositorio.
3. Comprobar que `app.py` y `requirements.txt` están en la raíz.
4. Abrir Streamlit Community Cloud y crear una nueva app.
5. Seleccionar repositorio, rama y archivo principal `app.py`.
6. Revisar la versión de Python en las opciones avanzadas. Se recomienda Python 3.12 salvo incompatibilidad específica.
7. Desplegar y revisar logs.
8. Probar: carga de Excel, evaluador individual, análisis RPT completo, introducción de dotaciones, informes y descargas.

## 5. Actualización del archivo de datos

Para actualizar criterios, patrones, puestos o importes:

1. Modificar el Excel de datos HET-CD validado.
2. Sustituir `data/het_cd_classifier_data.xlsx` por la nueva versión si se desea que sea la carga por defecto.
3. Mantener el mismo nombre de archivo para evitar cambios en código.
4. Probar localmente antes de desplegar.
5. Subir cambios a GitHub y reiniciar la app si fuera necesario.

## 6. Cautelas

- No subir datos personales o información sensible a repositorios públicos.
- Mantener `patrones_vector` con control de `activo_calculo`.
- No activar patrones no validados.
- Documentar cualquier cambio de ponderaciones, criterios o rangos.
- El cálculo económico por defecto puede ser unitario si no se introducen dotaciones afectadas.

## 7. Versiones

- App: 1.0.0
- Motor HET-CD: 1.0.0
- Modelo de datos: HET-CD v1.8
