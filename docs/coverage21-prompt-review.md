# Revisión de los 21 casos de integración

Revisión local de implementación por Codex, en sandbox. No constituye aprobación humana independiente, autorización de despliegue ni activación del bloqueo remoto.

## Cobertura exigida

La política conserva los 43 casos existentes y hace obligatoria su ejecución completa en Spring/H2: antes había 22 ejecutables y 21 `NOT_RUN`. No se elimina ningún test obligatorio ni turno requerido de la política anterior, incluido el cierre adicional `__complete` de SC-012-failed-edit. Se agregan 10 pruebas unitarias del agente y 3 del backend. Los 17 tests de rechazo del bloqueo siguen siendo obligatorios.

Los nuevos adaptadores ejercitan idioma y datos, fecha ambigua de SPA, reintento de entrega, renovación de ruta, aislamiento de huésped/estancia, dos tareas abiertas, seis planes inválidos, FAQ aprobada y desconocida, recepción reconfigurada, selección ambigua, un servicio genérico configurado, límites de botones, conservación de productos y mensajes en cola. Las comprobaciones del backend leen operaciones, tareas, mensajes y outbox persistidos; no se satisfacen copiando valores esperados de las fixtures.

El replay Python aumenta de 28 a 35 casos ejecutados. Mantiene 8 `NOT_RUN` y declara las comprobaciones y eventos que requieren Spring como pendientes. Un caso sin prefijo ejecutado nunca se cuenta como aprobado. La captura de prompts omite esos casos que empiezan en un evento de backend, pero continúa rechazando los fallos de cualquier replay ejecutado.

## Cambios de comportamiento

- Renovar la ruta del mismo huésped y estancia conserva el resumen y los campos capturados. Se invalidan las confirmaciones antiguas. Un inicio de servicio aún pendiente o fallido no se convierte en una solicitud nueva. Cambiar de estancia conserva el aislamiento.
- Con dos tareas abiertas, la evidencia del servicio indicado en el mensaje actual prevalece sobre un foco histórico. Los identificadores explícitos de respuesta conservan prioridad. La decisión requiere evidencia exacta, confianza suficiente y ausencia de tema ajeno; el contexto de interpretación se descarta al terminar cada turno. Una respuesta ambigua sigue sin completar tareas.
- Un “Confirmar” recibido antes de emitir el resumen de habitación no lo aprueba retroactivamente. Spring transmite ese orden temporal, el agente vuelve a presentar el resumen y los validadores Python/Java bloquean el inicio prematuro. Una confirmación nueva sí crea exactamente una operación.

## Revisión del snapshot

Comparado con el baseline del punto 7: 53 documentos cambian y el inventario pasa de 234 a 257, sin eliminar documentos. Permanecen las 12 familias y las 14 anclas críticas. No se eliminan instrucciones de los prompts. Las fuentes de aplicación modificadas son el planificador y el contexto de interpretación, por los controles descritos arriba. Los nuevos prompts efectivos provienen de rutas de prueba ahora ejecutables; el único prompt efectivo anterior que cambia contiene un identificador sintético distinto, por la nueva secuencia de generación determinista. Las instrucciones y opciones anteriores se conservan.

Se revisaron también las 21 fixtures, los adaptadores y la política protegida. Se corrigió el tipo de tarea sintética de mantenimiento para usar el nombre real del dominio. El caso de reasignación de contacto comprueba el rechazo que realmente ofrece la API y la ausencia de sesiones/turnos nuevos, sin simular una transferencia administrativa inexistente. El caso de cola ahora exige cero operaciones ante la confirmación prematura y una operación ante la confirmación nueva. Los planes inválidos se introducen deliberadamente después de la validación Python, para probar la defensa Java; cada rechazo debe tener la causa esperada y no alterar el estado de negocio.

## Evidencia y límites

Antes de aceptar el snapshot, los 43 casos pasaron dos veces consecutivas con scripts sintéticos (`target/ci/coverage21-repeat-final.json`). Las diez nuevas pruebas Python pasaron. La batería Python completa ejecutó 434 tests y sólo falló por la diferencia de baseline todavía no revisada; diez evaluaciones de modelo permanecen explícitamente omitidas. El cierre antiguo restaurado se comprueba por separado y la batería completa debe pasar después de aceptar esta revisión, con un directorio nuevo y hashes de fuentes estables.

La ejecución final pasó y se registra en `target/ci/coverage21-gate/summary.json`; su resumen durable está en `tests/conversation_regression/reports/coverage21-validation.json`. Resultado: 424 tests del agente, 569 del backend y 43 casos de integración con 110 turnos obligatorios, sin pasos ni assertions pendientes en Spring. El cierre antiguo restaurado también pasó. Los hashes de ambos repositorios permanecieron iguales durante la ejecución. Estos reportes son evidencia, no sustituyen futuras ejecuciones del bloqueo.

Spring, H2, el planificador Python, los validadores, las operaciones, las tareas, la búsqueda/validación de FAQ y el worker de outbox usan código real. Catálogos y esquemas, datos de huésped/estancia, respuestas del modelo, traducción externa, BPM y proveedor de mensajes se sustituyen por datos o clientes sintéticos. Los mensajes en cola se persisten juntos y se procesan secuencialmente: no es una prueba de concurrencia real de workers. No se prueba PostgreSQL, WhatsApp real, exactitud del modelo ni calidad de traducciones. `INTEGRATION_PARTIAL_PASS` conserva ese significado; los pasos y assertions de integración sí deben quedar todos ejecutados. No se hicieron llamadas a OpenAI ni cambios remotos.
