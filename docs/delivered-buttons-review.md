# Botones antiguos después de la entrega

El usuario reportó que un botón Cancelar produjo una afirmación de cancelación después de entregar room service. Se encontró el acuse exacto en el sandbox mediante consultas de sólo lectura. Los cambios anteriores están publicados en PR en borrador, sin desplegar.

El 21 de septiembre, a las 07:53:14 (UTC-6), la operación quedó `COMPLETED / DELIVERED`, versión 4. A las 07:53:33 llegó `confirmation:ROOM_SERVICE:CANCEL`; a las 07:53:48 el agente respondió que se había cancelado. El turno recibió la operación entregada en `recentOperations`, no tenía operaciones activas y no propuso ninguna herramienta. La operación persistida conservó su estado de entrega y su última actualización anterior al botón: fue un acuse falso, no una cancelación real.

El replay local de la entrada exacta persistida, con modelo y efectos externos bloqueados, respondió: «Ese menú ya no está vigente. Continúa con la solicitud actual.» Sin herramientas, llamadas a modelo ni cambios de datos. Ese formato ya lo rechaza el control pendiente de despliegue; la ampliación de hoy cubre otros tres identificadores heredados. La evidencia sin anonimizar permanece sólo en el directorio local `target/delivered-button-incident`, fuera de ambos repositorios.

## Defecto comprobado y corrección

La prueba local encontró un hueco en V2: `CANCEL_ORDER`, `CHANGE_ORDER` y `CONFIRM_ORDER`, emitidos por los prompts antiguos, no pasaban por el control de vigencia que ya reconocía `confirmation:ROOM_SERVICE:*` y `room-service:*`. Llegaban al planificador del modelo sin una capacidad vigente del borrador. La prueba negativa falló ante esa llamada no autorizada por su guion; no requiere OpenAI ni simula que el modelo haya dicho una frase concreta.

El cambio de aplicación sólo incorpora esos tres IDs antiguos a la clasificación de botones de confirmación. Al carecer de token válido, pasan por el control existente de menú vencido. No ejecutan acciones, no pueden borrar otro borrador y no dejan que el modelo improvise un acuse de cancelación. Los botones actuales conservan su comportamiento. Las decisiones de cocina `room-service-change:*` no cambian en esta corrección.

El backend ya rechaza acciones y nuevas transiciones sobre operaciones terminales. No se modifica esa protección. El adaptador de pruebas ahora prepara las versiones intermedias como activas antes de completar la operación, y lee su estado y versión persistidos para comprobar que ningún botón los altera. El replay Python separa operaciones finalizadas de las activas, como hace Spring.

## Cobertura y revisión del snapshot

Dos regresiones unitarias comprueban los formatos antiguos y actuales después de la entrega y la conservación de un nuevo borrador ante `CANCEL_ORDER`. Los 13 tests de versiones de confirmación pasan después del cambio. Se agregan dos casos obligatorios SC-009 (español e inglés), cinco turnos cada uno. Spring debe conservar `COMPLETED`, `DELIVERED` y versión 3, sin inicios de procesos. Las comprobaciones de base de datos permanecen explícitamente pendientes en el replay Python.

El snapshot propuesto cambia ocho documentos: un helper de aplicación, su archivo de tests existente, el adaptador offline, dos fixtures, sus enlaces en el contrato, la política del bloqueo y el token sintético de un prompt efectivo. No cambia ninguna instrucción ni opción del modelo. Se conservan las 14 anclas y las 12 familias; los documentos aumentan de 257 a 259 sin eliminaciones. Permanecen todos los tests y turnos previamente obligatorios.

La aceptación del snapshot es una revisión de implementación local por Codex, no aprobación humana independiente ni autorización de despliegue. La evidencia final se guarda en `target/ci/delivered-buttons-gate-final/summary.json` y `tests/conversation_regression/reports/delivered-buttons-validation.json`. No se hicieron llamadas a OpenAI ni mutaciones de servicios reales. La consulta real se limitó a leer el acuse coincidente, su conversación, operaciones y turnos. El archivo de contrato conserva las 14 reglas anteriores literalmente; un problema de codificación detectado al revisar el diff se corrigió antes de la validación definitiva.

Validación final: `PASS`, 426 pruebas unitarias del agente y 569 del backend; 45 conversaciones y 120 turnos obligatorios de integración, sin comprobaciones pendientes. Se omiten explícitamente diez pruebas de modelo real y el punto de entrada de integración en la suite general (ejecutado en la etapa dedicada). Las huellas de los dos repositorios coinciden con las fuentes validadas. El bloqueo remoto y el despliegue siguen pendientes.
