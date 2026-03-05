# Analisis Profundo

Usar este documento solo cuando `reglas_rapidas.md` no sea suficiente para que el agente tome decisiones seguras.

## Marco de control
- No cambiar arquitectura, contratos API ni logica de negocio sin aprobacion.
- No ejecutar cambios estructurales por iniciativa propia.
- Cualquier mejora mayor se propone primero y se espera confirmacion.

## Alcance por defecto
- Documentacion antes que implementacion.
- Si la tarea pide solo documentar, no tocar codigo.
- Si se autoriza codigo, limitarse exclusivamente a `aida`.

## Cambios importantes, nuevos o complejos (requieren confirmacion previa)
Se considera importante si afecta:
- Contratos API/JSON.
- Flujo de edicion/versionado.
- Seguridad, costos o rendimiento.
- Dependencias, despliegue o almacenamiento.

Antes de ejecutar, presentar siempre:
- Que se cambiara.
- Pros.
- Contras.
- Posibles errores/regresiones.
- Plan de implementacion.
- Impacto en compatibilidad OCR <-> editor.
- Solicitud explicita de autorizacion.

## Reglas de seguridad
- No exponer claves, tokens o credenciales.
- No guardar secretos en codigo fuente.
- Configuracion sensible en `.env` no versionado.
- Si se agregan variables nuevas, documentar uso e impacto.

## Trazabilidad y validacion
- Indicar archivos a modificar.
- Explicar riesgo estimado y pruebas de verificacion.
- Si algo no puede verificarse, reportarlo de forma clara.

## Flujo operativo acordado
- El agente solo implementa cambios locales en codigo y documentacion.
- El usuario realiza `git push` y las pruebas/despliegue en Render.
- No ejecutar `push`, releases o cambios en infraestructura sin autorizacion explicita.
- Cuando una validacion no pueda correrse localmente, entregar pasos de verificacion para Render.

## Formato esperado de respuesta del agente
1. Resumen del cambio.
2. Pros.
3. Contras.
4. Posibles errores.
5. Plan de implementacion.
6. Solicitud final de confirmacion.

