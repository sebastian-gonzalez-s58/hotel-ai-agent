# Biblioteca de regresión conversacional — punto 2

Este documento conserva la entrega y baseline del punto 2. Para los tres niveles
implementados y sus resultados actuales, consultar
[Evaluación conversacional — punto 3](../../docs/conversation-evaluation.md).
Las [correcciones posteriores](../../docs/conversation-fixes.md) amplían la
biblioteca a 42 casos; la corrección 6 ejecuta interrupción y reanudación.

Biblioteca inicial para sandbox: **40 casos de 21 escenarios**, enlazados al
[contrato 1.0.0](../../contracts/behavior/conversation-behavior.v1.json).
Incluye incidentes anonimizados, recorridos exitosos y condiciones límite.
Cada rama independiente tiene su propio caso.
El [índice completo](INDEX.md) permite abrir las 40 conversaciones.

Los mensajes son reproducciones sintéticas basadas en los problemas analizados,
no transcripciones completas ni exportaciones de huéspedes. Los IDs, nombre,
habitación y hotel son ficticios; las URLs usan `.example`. El reloj inicial es
`2026-10-01T16:00:00Z`, zona `America/Mexico_City`, para evitar que una fecha de
reserva se vuelva pasada al ejecutar las pruebas más adelante.

## Qué contiene cada caso

- `scenario_id`: garantía de negocio a la que pertenece.
- `profile`: snapshot V2 con offerings, campos, catálogos y herramientas.
- `initial_summary`, `initial_operations`, `locale`: preparación inicial.
- `steps`: mensajes y eventos del entorno en orden.
- `checks`: resultados observables con rutas JSON Pointer y comparaciones.
- `review`: criterios semánticos pendientes de evaluación; no cuentan como aprobados.
- `model_replies`: respuestas sintéticas del modelo para reproducción offline.
  `null` significa que falta el guion; `[]` exige que no haya llamada al modelo.
- `required_levels`: niveles exigidos por el contrato, conservados sin reducirlos.

[library.py](library.py) valida el formato, IDs únicos, referencias al contrato,
snapshots y herramientas propuestas. Rechaza campos no reconocidos. Los datos
internos de acciones del entorno se describen en [adapters.md](adapters.md);
su ejecución e interpretación todavía requieren adaptadores.

## Estado inicial medido

[sandbox-baseline.json](reports/sandbox-baseline.json) contiene la ejecución local,
observaciones por turno, diferencias y huellas del código, fixtures y contrato.
Los éxitos offline tienen cobertura parcial; no acreditan el contrato completo.

| Resultado | Casos | Significado |
| --- | ---: | --- |
| OFFLINE_PARTIAL_PASS | 8 | Pasan las aserciones automáticas con modelo simulado |
| FAILED | 4 | Incumplimiento reproducido bajo las condiciones del guion |
| NOT_RUN | 28 | Faltan guiones de modelo o adaptadores de entorno |
| HARNESS_ERROR | 0 | No hubo errores del ejecutor en esta ejecución |

Los cuatro fallos reproducidos son:

| Caso | Resultado observado |
| --- | --- |
| [SC-001](fixtures/cases/SC-001-split-quantity.json) | Descarta el artículo sin cantidad; la secuencia se detiene en ese primer fallo |
| [SC-004](fixtures/cases/SC-004-edit-location.json) | Conserva POOL_1 tras pedir entrega en la habitación |
| [SC-009](fixtures/cases/SC-009-double-space.json) | Rechaza confirmar el pedido cuyo nombre contiene doble espacio |
| [SC-011](fixtures/cases/SC-011-spa-name.json) | La aclaración base muestra serviceName y solicita fecha/hora al aclarar el servicio |

Las ocho reproducciones parciales satisfactorias cubren fecha/hora separadas de
spa, edición de cantidades/eliminación conservando restricciones, cambio explícito
de idioma conservando el pedido, recuperación de un fallo del extractor, recepción
directa y selección por botón, código o etiqueta inglesa. No son ocho evaluaciones
del modelo real.

La suite offline de unidad se ejecutó aparte: **329 pruebas, 319 aprobadas y 10
omitidas** por requerir habilitación explícita. Incluye 18 pruebas nuevas de
integridad del ejecutor. Que esa suite pase no convierte en verdes los cuatro
fallos de la biblioteca: se mantienen en el reporte y el replay retorna error.

## Cómo ejecutar

Desde la raíz del repositorio del agente, con sus dependencias instaladas:

```powershell
python -m tests.conversation_regression.replay validate
python -m unittest tests.test_conversation_regression_library
python -m tests.conversation_regression.replay replay --report tests/conversation_regression/reports/sandbox-baseline.json
python -m tests.conversation_regression.replay replay --case SC-005-edit-items
```

El último comando permite investigar un solo caso; no evalúa toda la biblioteca.
El replay retorna `1` si hay fallos o errores del ejecutor, `2` si solo quedan
casos sin ejecutar y `0` si todos los casos seleccionados pasan las aserciones
offline. Ninguno de estos resultados habilita por sí mismo un despliegue.
`validate` comprueba estructura y referencias; no ejecuta conversaciones.

## Evitar resultados engañosos

[session.py](session.py) alimenta cada turno con el resumen y las respuestas
realmente devueltos por el agente anterior. Las expectativas nunca escriben el
estado. Conserva los mensajes originales, agrega las respuestas reales y limita
el snapshot a los últimos 100 mensajes. Sigue la semántica actual de resúmenes
nulos/vacíos y conserva la decisión de idioma con evidencia del mensaje.

El adaptador no ejecuta herramientas: si un turno propone una y después intenta
continuar sin procesar el resultado, falla. Los eventos de backend no se simulan
como éxitos. Los casos no disponibles aparecen como NOT_RUN; no son skips verdes.
Cada caso se detiene al primer fallo: los siguientes turnos no están evaluados.

Una aserción negativa sobre un campo ausente falla. Las comparaciones distinguen
booleanos de números. Un guion con llamadas adicionales, fuera de orden o sin
consumir produce HARNESS_ERROR y exige revisar el ejecutor o el guion.

Las rutas `/state/...` reflejan el almacenamiento actual del agente; son un
adaptador del requisito de conservar información. Si cambia su representación,
se debe adaptar la observación manteniendo el requisito y una prueba de migración.
Las preguntas no exigen una redacción literal. En spa comprobamos estado de
confirmación e interacción porque su etiqueta interna actual es CLARIFICATION
aunque el mensaje sí pide confirmar.

## Límites y siguiente etapa

El replay ejecuta el planner real con respuestas sintéticas en la frontera del
modelo y conexiones bloqueadas. **No evalúa prompts ni comprensión del modelo**.
La localización de la respuesta se omite para aislar la orquestación; no se
certifica la traducción ni el texto final del canal. El reporte declara esos
límites también para cada conjunto de resultados aprobados parcialmente.

El punto 3 agrega ejecución con modelo real y Spring/H2 para los casos soportados.
Los eventos de renovación de sesión, aislamiento de identidades, colas, outbox y
otras ramas siguen pendientes de adaptadores; ver el alcance exacto en su documento.
También falta automatizar los criterios de `review`. No existe todavía barrera
de promoción a dev ni una garantía de ausencia de regresiones.

## Agregar o modificar casos

1. Elegir un escenario del contrato o agregar uno con justificación.
2. Crear un JSON con ID estable, datos sintéticos y un evento por paso.
3. Especificar tanto conservación de datos como acciones permitidas/prohibidas.
4. Separar otras ramas en otros casos. No reemplazar estado entre turnos.
5. Agregar guiones solo cuando su propósito y evidencia correspondan al mensaje.
   Identificarlos como sintéticos, nunca como resultados observados del modelo.
6. Ejecutar validación, pruebas del ejecutor y replay; conservar fallos y pendientes.

Cambiar expectativas requiere explicar el cambio de producto o el error del
oráculo. No se borran casos ni se convierten en éxitos para aprobar un cambio.

## Protección de prompts (punto 4)

El [verificador de cambios](../../docs/prompt-change-protection.md) mantiene un
inventario vinculado a estas conversaciones y captura prompts efectivos con datos
sintéticos. `python -m tests.prompt_contract.guard check` detecta cambios respecto
a la referencia; `propose` genera el diff para revisión sin aceptarlo. También se
ejecuta dentro de la suite offline. Esta comprobación no evalúa al modelo real ni
convierte en aprobados los casos pendientes.
