# Contrato de comportamiento conversacional — sandbox

Versión **1.0.0** · Fecha **2026-09-20** · Runtime V2.

Este documento implementa el punto 1 del plan de prevención de regresiones:
definir las garantías y sus escenarios de aceptación. El catálogo estructurado es
[conversation-behavior.v1.json](../contracts/behavior/conversation-behavior.v1.json).
Ambos archivos se mantienen juntos en el repositorio del agente.

**Definir un escenario no significa que pase.** El punto 1 fijó la especificación.
El [punto 2](../tests/conversation_regression/README.md) agrega 40 casos y una
reproducción offline parcial, con fallos y pendientes explícitos. La evaluación
completa y el bloqueo de despliegue todavía corresponden a los siguientes puntos.

## Garantías

La unidad de protección es un comportamiento observable, no una frase exacta del
prompt. Una regla puede necesitar código, prompt y persistencia del backend.
Conservar su texto sin probar el resultado no demuestra cumplimiento.

| ID | Garantía | Responsable | Escenarios |
| --- | --- | --- | --- |
| BC-001 | Conservar captura parcial | Agente | SC-001, SC-002 |
| BC-002 | Resolver respuestas según la pregunta pendiente | Agente | SC-001, SC-003, SC-015 |
| BC-003 | Aplicar cambios únicamente a campos solicitados | Agente | SC-004, SC-005 |
| BC-004 | Conservar solicitudes durante interrupciones | Agente + backend | SC-006, SC-015 |
| BC-005 | Separar idioma de estado de negocio | Agente + backend | SC-007, SC-008 |
| BC-006 | Confirmar la versión vigente y ejecutar una sola vez | Agente + backend | SC-009, SC-010 |
| BC-007 | Preguntar por lo realmente pendiente | Agente | SC-001, SC-003, SC-011, SC-012 |
| BC-008 | Respetar autorización y evidencia al usar herramientas | Agente + backend | SC-010, SC-015, SC-016 |
| BC-009 | Recuperar errores sin borrar datos ni duplicar efectos | Agente + backend | SC-010, SC-012 |
| BC-010 | Mantener continuidad con límites de identidad | Backend | SC-013, SC-014 |
| BC-011 | Responder con conocimiento aprobado | Agente + backend | SC-017 |
| BC-012 | Respetar la configuración de offerings | Agente + backend | SC-004, SC-018, SC-019 |
| BC-013 | Preservar datos originales en presentación | Agente | SC-009, SC-020 |
| BC-014 | Procesar el mensaje y la solicitud correctos | Agente + backend | SC-015, SC-021 |

Las 14 garantías iniciales son críticas: incumplirlas puede perder datos, impedir
completar un flujo válido, inventar hechos o producir una acción incorrecta.
Una preferencia estética o una redacción distinta no son por sí solas regresiones
críticas. No se exige un número fijo de mensajes para todos los recorridos; se
comprueba que una pregunta adicional tenga una razón concreta.

## Escenarios iniciales

Cada escenario JSON contiene contexto, pasos, expectativas, resultados prohibidos
y niveles de prueba. Los pasos que describen ramas independientes se convertirán
en casos separados, no en una única secuencia. Las expectativas son de negocio y
no obligan a mantener el almacenamiento actual dentro de `summary`.

| ID | Escenario | Origen |
| --- | --- | --- |
| SC-001 | Producto y cantidad en mensajes separados | Incidente anonimizado y variantes |
| SC-002 | Pedido con un artículo incompleto | Incidente anonimizado y variantes |
| SC-003 | Fecha y hora de spa por separado | Incidente anonimizado y variantes |
| SC-004 | Editar una ubicación ya capturada | Incidente anonimizado y variantes |
| SC-005 | Editar cantidad conservando restricciones | Recorrido a proteger |
| SC-006 | Interrumpir pedido con FAQ y recepción | Recorrido a proteger |
| SC-007 | Cambiar idioma durante captura | Recorrido a proteger |
| SC-008 | No inventar preferencia explícita de idioma | Incidente anonimizado y variantes |
| SC-009 | Confirmación con espacios y versión de borrador | Incidente anonimizado y variantes |
| SC-010 | Idempotencia y acuse veraz | Caso límite |
| SC-011 | Aclaraciones específicas de spa | Incidente anonimizado y variantes |
| SC-012 | Fallo técnico durante edición o traducción | Caso límite |
| SC-013 | Retomar tras cambio técnico de sesión | Incidente anonimizado y variantes |
| SC-014 | No heredar datos entre identidades o estancias | Caso límite |
| SC-015 | Dos tareas abiertas y respuesta ambigua | Caso límite |
| SC-016 | Restricciones de herramientas y datos requeridos | Caso límite |
| SC-017 | FAQ exacta y derivación sin inventar | Recorrido a proteger |
| SC-018 | Recepción sin preguntas innecesarias | Recorrido a proteger |
| SC-019 | Equivalencia entre botón y texto de catálogo | Recorrido a proteger |
| SC-020 | Traducción preserva datos y decisiones | Incidente anonimizado y variantes |
| SC-021 | Mensajes consecutivos y turno disparador | Caso límite |

Los ejemplos derivados de incidentes no contienen identificadores, teléfonos,
folios ni URLs de huéspedes o instancias reales. Las fechas usan un reloj fijo
de prueba, independiente del día de evaluación. Una salida incorrecta observada
no se convierte en respuesta esperada.

## Cobertura existente y brechas

El catálogo enlaza **23 pruebas existentes** por ruta, clase y método, y explica
qué cubren y qué falta. Son referencias de cobertura **parcial**, no resultados de
evaluación de estos escenarios completos.

- El test de cantidad faltante recupera el pedido haciendo que el huésped repita
  producto y cantidad. No acredita que «Uno» complete el producto anterior (SC-001).
- Resolver una opción de catálogo faltante no demuestra que se pueda editar una
  ubicación ya guardada (SC-004).
- Simular la respuesta correcta del clasificador no prueba su interpretación real
  de los mensajes (SC-008, SC-015).
- La caché del endpoint Python no demuestra que Spring evite crear dos operaciones
  por el mismo evento (SC-010).
- Una prueba de presentación no acredita recuperación del outbox ni continuidad
  entre sesiones (SC-012, SC-013).

En el punto 2, `executionStatus` distingue `fixtures_only` de `partial_offline`.
Cada escenario enlaza sus conversaciones mediante `fixturePaths`; el cargador
verifica esas referencias. `sandboxResult=not_evaluated` sigue refiriéndose al
escenario completo en todos sus niveles, no a sus reproducciones parciales.
Los resultados parciales están en el reporte enlazado desde `library`.
Un incidente de Telware **no se declara automáticamente reproducido en sandbox**:
las ramas y modelos pueden diferir.

## Cómo se comprobará

1. **Offline:** transformaciones, validaciones y reproducción con dependencias
   controladas; sin llamadas al modelo ni servicios de hotel.
2. **Modelo real:** clasificación, extracción y respuesta con información sintética.
   Se registran todos los resultados, incluidos fallos; no ejecuta herramientas
   reales ni envía mensajes al huésped.
3. **Integración:** backend, agente, persistencia y herramientas en entorno aislado,
   con entrega simulada; comprueba efectos en la base de pruebas.

Cada regla tiene al menos un escenario. Un escenario solo se declara verificado
cuando existe una prueba ejecutable y evidencia de todos sus niveles requeridos,
identificando commits, contrato, prompts, modelo y configuración. Un test omitido,
un mock del componente evaluado o un resultado favorable aislado no constituye una
garantía completa.

En secuencias de varios turnos, cada turno consume el estado producido por el
anterior: no se construye manualmente un estado ideal entre respuestas. Se
comprueban datos, acciones, necesidad de las preguntas y fidelidad de presentación.
No se exige una frase exacta cuando varias redacciones cumplen el contrato.

Idiomas base de la futura biblioteca: español e inglés, con italiano para el caso
de respuesta corta observado. Otras variantes se incorporarán explícitamente;
estos casos no acreditan cobertura universal multilenguaje.

## Cambios y fallos conocidos

- Los IDs son estables y no se reutilizan. Retirar una regla conserva su historial
  y motivo; no desaparece silenciosamente.
- Cambiar expectativas exige explicar el cambio de producto, escenarios afectados
  y evidencia. No se debilitan expectativas para obtener un resultado verde.
- Cada corrección agrega o amplía un escenario que falle antes y pase después.
  Se conservan también recorridos exitosos.
- Los fallos conocidos se registrarán con escenario, entorno, evidencia y condición
  de cierre. No se contabilizan como éxitos ni excepciones permanentes.
- Modelo, configuración de offering y normalización pueden alterar comportamiento
  aunque no cambie un prompt; también requieren evaluación.
- Conservar una frase en un prompt es una comprobación estructural, no una prueba
  de comprensión del modelo.
- Ante ambigüedad real, aclarar es correcto. Un fallo técnico exige conservar el
  progreso y explicar la recuperación necesaria; no autoriza inventar datos.

Estas son reglas del proceso propuesto; todavía no existe enforcement automático
ni bloqueo de despliegue en esta entrega.

## Continuidad y límites de identidad

SC-013 permite recuperar una captura inequívoca o pedir confirmación explícita de
su reanudación dentro del mismo contacto, huésped y estancia. SC-014 prohíbe
heredar silenciosamente datos entre identidades o estancias. El contrato no impone
todavía un diseño de migración; se implementará con pruebas de integración.

No se interpreta un `close_reason` interno como prueba de vencimiento real de
la ventana de WhatsApp.

## Criterio de cierre del punto 1

- 14 garantías con IDs estables, responsables y escenarios asociados.
- 21 escenarios con resultados observables y comportamientos prohibidos.
- 23 referencias a pruebas existentes, con límites de cobertura explícitos.
- Integridad de IDs, relaciones y referencias verificada localmente.
- La biblioteca del punto 2 incorpora fixtures y replay parcial. Siguen pendientes
  las evaluaciones completas, adaptadores, correcciones y barrera de promoción.
  La especificación no modifica ni publica servicios.

## Historial

| Versión | Cambio |
| --- | --- |
| 1.0.0 | Contrato inicial en sandbox: requisitos, escenarios y cobertura parcial. |
