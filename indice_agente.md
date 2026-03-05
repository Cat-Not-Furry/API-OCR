# Indice de trabajo para agentes

Este archivo define el orden de lectura y uso de reglas para cualquier agente que trabaje en `aida`.

## Orden obligatorio
1. Leer `reglas_rapidas.md`.
2. Ejecutar la tarea con ese marco.
3. Si hay ambiguedad, riesgo alto o cambio complejo, escalar a `analisis_profundo.md`.

## Cuando usar cada documento

### `reglas_rapidas.md`
Usar por defecto para:
- Actualizaciones de documentacion.
- Cambios pequenos y acotados.
- Tareas de bajo riesgo y sin impacto arquitectonico.

### `analisis_profundo.md`
Usar cuando:
- Las reglas rapidas no sean suficientes.
- El cambio afecte contratos API/JSON.
- El cambio impacte flujo OCR <-> editor.
- Haya riesgo en seguridad, costos, despliegue o rendimiento.
- Sea un cambio nuevo, importante o complejo.

## Regla de confirmacion previa
Antes de ejecutar cambios importantes, nuevos o complejos, el agente debe presentar:
- Resumen del cambio.
- Pros.
- Contras.
- Posibles errores/regresiones.
- Impacto en compatibilidad OCR <-> editor.
- Solicitud explicita de autorizacion.

## Limites de accion
- No modificar arquitectura ni logica de negocio sin autorizacion.
- No salir del alcance de `aida`.
- No inventar endpoints, variables o tablas no aprobadas.

## Resultado esperado
Todo agente debe producir salidas claras, accionables y trazables, priorizando estabilidad y compatibilidad del proyecto.

