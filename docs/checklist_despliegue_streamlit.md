# Checklist de despliegue HET-CD en Streamlit

## Preparación del repositorio

- [ ] Repositorio privado creado en GitHub.
- [ ] `app.py` en la raíz.
- [ ] `het_cd_engine.py` en la raíz.
- [ ] `het_cd_batch.py` en la raíz.
- [ ] `requirements.txt` en la raíz.
- [ ] `data/het_cd_classifier_data.xlsx` incluido.
- [ ] Manual técnico y guía rápida incluidos en `docs/`.
- [ ] No hay archivos temporales ni salidas con datos sensibles.

## Prueba local

- [ ] `pip install -r requirements.txt` ejecuta sin errores.
- [ ] `streamlit run app.py` abre la app.
- [ ] La app carga el Excel base.
- [ ] Configurador funcional operativo.
- [ ] Evaluador HET-CD operativo.
- [ ] Análisis RPT completo operativo.
- [ ] Se pueden introducir dotaciones afectadas.
- [ ] Descargas de informes funcionan.
- [ ] Batch ejecuta con `python het_cd_batch.py data/het_cd_classifier_data.xlsx`.

## Despliegue cloud

- [ ] App creada en Streamlit Community Cloud.
- [ ] Repositorio y rama seleccionados.
- [ ] Archivo principal: `app.py`.
- [ ] Python configurado en versión compatible.
- [ ] Logs sin errores de dependencias.
- [ ] App accesible mediante URL asignada.

## Validación posterior

- [ ] Inicio carga sin errores.
- [ ] Excel base detectado.
- [ ] Se puede sustituir el Excel por una versión actualizada.
- [ ] Informe individual generado.
- [ ] Informe global generado.
- [ ] Cálculo económico con dotaciones revisado.
- [ ] Textos y tablas revisados en pantalla.
- [ ] Permisos de acceso revisados.
