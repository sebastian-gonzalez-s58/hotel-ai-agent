# Evaluación conversacional en tres niveles — punto 3

Este documento conserva los resultados anteriores a las correcciones del agente.
Los cambios y resultados posteriores están en [Correcciones de regresión](conversation-fixes.md).

El ejecutor ya permite recorrer conversaciones, conservar el estado real entre
turnos y reportar el primer incumplimiento del contrato. Se implementaron los
tres niveles en los checkouts de sandbox; la cobertura de los 40 casos todavía
es parcial. No se modificaron los prompts ni el comportamiento del agente.

## Resultados observados

| Nivel | Alcance ejecutado | Resultado |
| --- | --- | --- |
| Offline | 18 casos | 14 aprobaciones parciales, 4 fallos; otros 22 pendientes |
| Modelo real | 3 casos, 2 repeticiones cada uno | 2 aprobaciones parciales, 4 fallos |
| Spring + Python + H2 | 12 casos | 8 aprobaciones parciales, 4 fallos; otros 28 pendientes |

Los seis intentos con modelo real usaron **gpt-4.1-mini**, tomado de la
configuración local de sandbox: **18 llamadas y 24,785 tokens**, dentro del límite
autorizado de 30 llamadas. Se conservaron todas las repeticiones, incluidas las
fallidas. No se evaluaron los otros 37 casos con modelo real.

Reportes:

- [Offline](../tests/conversation_regression/reports/point3-offline-final.json).
- [Integración](../tests/conversation_regression/reports/point3-integration-final.json).
- [Modelo real: spa y edición](../tests/conversation_regression/reports/point3-live-connected.json).
- [Modelo real: cambio de idioma](../tests/conversation_regression/reports/point3-live-language.json).

En offline e integración se reprodujeron los cuatro fallos conocidos: pérdida
del producto sin cantidad, cambio de ubicación ignorado, confirmación rechazada
con doble espacio y aclaración base de spa que expone campos internos.

La comparación revela problemas adicionales al usar el modelo real:

| Caso | Con respuestas controladas | Con modelo real |
| --- | --- | --- |
| SC-003: tratamiento/fecha → 6pm | Conserva los campos y pide confirmar | Una repetición perdió el tratamiento; la otra no resolvió la hora |
| SC-005: cambiar cantidad → quitar sopa | Aplica los dos cambios | Ambas repeticiones conservaron la sopa |
| SC-007: pedir inglés → confirmar | Conserva el pedido | Ambas repeticiones conservaron el pedido y propusieron su creación |

Las pruebas de integridad del agente suman **342 pruebas: 332 aprobadas y 10
omitidas** por habilitación explícita. Incluyen 13 nuevas del ejecutor del punto
3, además de las 18 del punto 2. Esas pruebas comprueban que el arnés funciona;
los errores del agente permanecen fallidos en los reportes conversacionales.
Además pasaron 15 pruebas existentes del backend sobre contexto, turnos,
límites transaccionales e idempotencia/persistencia de servicios.

## Ejecutar

Desde la raíz del agente, con sus dependencias Python instaladas:

```powershell
python -m tests.conversation_regression.evaluate --level offline --report tests/conversation_regression/reports/my-offline-run.json
python -m tests.conversation_regression.evaluate --level integration --report tests/conversation_regression/reports/my-integration-run.json
python -m tests.conversation_regression.evaluate --level live_model --model gpt-4.1-mini --case SC-003-spa-split-time --case SC-005-edit-items --repeat 2 --max-calls 30 --max-tokens 60000 --report tests/conversation_regression/reports/my-live-run.json
```

El nivel real requiere establecer `REGRESSION_OPENAI_API_KEY` explícitamente
mediante el mecanismo local de secretos. No carga automáticamente la credencial
de sandbox. Envía prompts y entradas sintéticas a OpenAI y genera consumo; debe
ejecutarse con autorización y presupuesto definidos. `--model` es obligatorio.
No usa endpoints ni credenciales del hotel.

Para integración se necesitan Maven, Java 17+ y el checkout hermano
`sandbox-backend` (se puede indicar `--backend`). Maven funciona en modo offline
y requiere dependencias ya descargadas. `REGRESSION_MAVEN_REPOSITORY` permite
indicar otra caché local cuando el entorno de ejecución usa un home diferente.
La base H2 se crea en memoria; no se usa la base del sandbox desplegado.

Cada comando exige un nombre de reporte nuevo. No sobrescribe resultados
anteriores. `--case` admite patrones y puede repetirse. `--repeat` registra cada
ejecución por separado; no se repite un caso hasta conseguir un verde.

## Qué ejecuta cada nivel

**Offline:** usa el planner y validadores reales con respuestas sintéticas en
la frontera del modelo. Se agregaron seis pruebas que entregan planes inválidos
concretos al validador Python: herramienta prohibida, offering desconocido,
versión obsoleta, evidencia ajena, campos requeridos ausentes y confirmación
ausente. Se comprueba el motivo del rechazo, no solo que ocurra una excepción.
Esto no acredita por sí mismo el rechazo independiente de Spring.

**Modelo real:** utiliza el planner, prompts, extracción, clasificación y
localización candidatos. No sustituye las respuestas por los guiones offline.
Conserva solicitudes, respuestas, propósito y hash del prompt de cada llamada,
ID de respuesta, consumo y duración. Solo admite el endpoint oficial de OpenAI,
deshabilita telemetría hacia hoteles y no ejecuta herramientas de negocio.
La caché de traducción se comparte dentro de la corrida, como declara el
ejecutor; la interpretación de cada turno vuelve a invocar el modelo.

El límite de llamadas se verifica antes de invocar el modelo. El umbral de tokens
detiene nuevas llamadas después de medir la última; puede superarse por el
consumo de esa llamada. Cada respuesta tiene un máximo de 4,096 tokens de salida.
Los reintentos de transporte del SDK están deshabilitados. Las reparaciones
propias del agente cuentan como nuevas llamadas.

**Integración:** la [prueba Java](../../sandbox-backend/src/test/java/com/chatbotinn/agent/v2/service/ConversationRegressionIntegrationTest.java)
usa los servicios reales de contexto, conversación, turnos, registro/ejecución de
herramientas y creación de operaciones. El cliente HTTP de Spring llama al planner
Python por un puente enlazado exclusivamente a 127.0.0.1. La base H2 guarda el
estado y el próximo turno se construye desde ella, no desde las expectativas.

Cuando hay una confirmación, se pulsa el botón realmente presentado por el
agente y se compara el input de la operación creada con lo capturado. Reenviar
el mismo inbound comprueba que no se cree otra operación. La entrega simulada
persiste los mensajes generados sin conectarse a router o WhatsApp.

Los proveedores de huésped/estancia, schemas y catálogos son fixtures. El puerto
de procesos BPM y la entrega externa son simulados; el modelo también está
controlado en este nivel. No se certifican autenticación del endpoint Python,
migraciones PostgreSQL, ejecución BPM completa ni entrega de WhatsApp.

## Lectura de resultados

- `first_failure` identifica el primer turno fallido y `rule_ids` lo vincula
  con las garantías. Integración incluye también las trazas de sus intercambios.
- `FAILED` indica una expectativa incumplida. Los turnos posteriores no se ejecutan.
- `DEPENDENCY_ERROR` separa problemas del proveedor de errores de comportamiento.
- `HARNESS_ERROR` identifica problemas del ejecutor, guion o entorno.
- `NOT_RUN` indica falta de adaptador o guion; nunca cuenta como aprobado.
- `*_PARTIAL_PASS` aprueba únicamente las comprobaciones automáticas ejecutadas.
  Los criterios libres de `review` y demás niveles siguen pendientes.

El proceso termina con código 1 ante fallos, errores del arnés, dependencias o
presupuesto agotado; con 2 si faltan casos y no hubo otros fallos; con 0 si todas
las comprobaciones seleccionadas pasan. **Todavía no está conectado a una
barrera de despliegue.** Eso pertenece al punto 7.

Se mantienen las expectativas de negocio. Por ejemplo, cambiar la cantidad de un
artículo no autoriza perder sus modificaciones. La comparación distingue tipos
también dentro de listas y objetos: `true` no satisface una cantidad esperada de 1.
Un dato ausente no satisface una comprobación negativa.

## Pendientes concretos de cobertura

Se añadieron posteriormente los recorridos de interrupción con FAQ sin respuesta
aprobada, cambio de idioma y confirmaciones antiguas de Room Service. Un evento
guest con `reply_from_turn` selecciona una opción realmente emitida: un entero
positivo apunta a una respuesta previa (base 1), y -1 busca la última coincidencia.
`reply_id` indica el prefijo y la acción que se seleccionan; sin referencia se
envía literalmente, permitiendo probar identificadores inválidos o heredados.
Los casos que usan un índice positivo después de resultados de herramientas deben
tener presente que el adaptador Python registra cada respuesta, mientras Spring
registra la respuesta final de cada turno entrante. SC-009 referencia únicamente
las tres primeras respuestas, que no ejecutan herramientas en ambos niveles.

Quedan por implementar adaptadores para FAQ con respuesta aprobada, nueva
configuración de offerings, tareas simultáneas,
fallo/reintento de outbox, renovación de sesión,
aislamiento de identidades y mensajes en cola. El reenvío probado al completar
un pedido no cubre todos esos escenarios. Las pruebas existentes del backend
aportan evidencia parcial, no sustituyen sus conversaciones completas.

SC-010 añade adaptadores Spring para repetir el mismo evento y herramienta,
rechazar argumentos modificados con el mismo ID, inyectar un fallo de inicio y
perder su resultado después del commit para recuperar la cadena original.
Offline ejecuta solo el prefijo anterior al primer evento backend y registra
explícitamente `not_run_steps`; `OFFLINE_PARTIAL_PASS` en estos casos no acredita
idempotencia ni recuperación. El caso de fallo técnico usa un resultado simulado
ligado al ID de la herramienta realmente propuesta. Esos fallos se excluyen de live.

Los casos formados solo por mensajes de huésped pueden ejecutarse con modelo
real aun sin guion offline. Los que contienen eventos del entorno siguen
marcados NOT_RUN. Tampoco se simula silenciosamente un fallo técnico al ejecutar
un caso real: los escenarios de inyección de errores pertenecen al nivel controlado.

Los reportes `point3-live-smoke`, `point3-integration-smoke` y
`point3-integration-http` conservan los intentos de puesta a punto (red no
disponible, framing HTTP y metadatos de botones incompletos en el arnés).
No son evidencia de aceptación del producto. Los reportes enlazados arriba son
la referencia de esta entrega.

El diseño de combinar comprobaciones deterministas con evaluaciones de tareas
representativas sigue las [buenas prácticas de evaluación de OpenAI](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
Las evaluaciones no garantizan ausencia de errores ni sustituyen revisión humana.

El [check obligatorio del punto 7](conversation-ci-gate.md) ejecuta los niveles
deterministas y compara su cobertura con una política explícita. Un `NOT_RUN`
solo es admisible si corresponde a una brecha ya registrada; una conversación
obligatoria omitida bloquea el resultado. El informe conserva los pendientes y
las limitaciones de las pruebas parciales. El modelo real sigue siendo una
evaluación separada, con autorización y presupuesto propios.
