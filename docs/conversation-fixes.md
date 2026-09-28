# Correcciones de regresión en sandbox

Estado actual: implementadas las correcciones 1 a 6 en el checkout local de sandbox.
La sección 7 contiene la cobertura más reciente: cambio de idioma con solicitudes
pendientes, sin modificaciones adicionales al agente. No se ha desplegado.

## 1. Confirmar un pedido con espacios dobles (SC-009)

Garantías protegidas: **BC-006** (confirmar la versión actual) y **BC-013**
(preservar datos de negocio). El pedido confirmado debe conservar producto,
cantidad, restricciones y ubicación del borrador que revisó el huésped.

### Causa y cambio

La confirmación compactaba espacios del nombre mediante `_coerce_order_items`,
pero el validador comparaba el input resultante con el borrador original.
`Pechuga  Herbal` pasaba a `Pechuga Herbal` y la validación rechazaba el pedido.

En el flujo multilenguaje, `_room_service_draft_plan` ahora entrega los campos
capturados sin reescribirlos. La coerción sigue comprobando que haya artículos
y los validadores de schema y confirmación siguen activos. Se conserva la
comparación exacta con el borrador actual; no se flexibilizó el validador.
El flujo anterior a multilenguaje conserva su comportamiento.

Archivo de producción: [v2_turn_planner.py](../app/agents/v2_turn_planner.py).
**Prompts modificados: ninguno.** No se requiere una llamada al modelo para
confirmar este botón. No se hicieron llamadas a OpenAI para esta corrección.

### Pruebas y evidencia

- La nueva prueba `UnderstandingFlowTest.test_confirmation_preserves_exact_captured_order`
  en [test_multilingual_understanding.py](../tests/test_multilingual_understanding.py)
  falló antes de aplicar el arreglo con `Room service requires explicit
  confirmation of the current captured order` y pasó después.
- Conserva espacios del nombre y de las restricciones, no modifica la solicitud
  recibida y no invoca el extractor de artículos al confirmar.
- Rechaza mutaciones de producto, cantidad, ubicación o restricciones. También
  rechaza una confirmación desarmada, un mensaje sin confirmación y evidencia ajena.
- El caso [SC-009-double-space](../tests/conversation_regression/fixtures/cases/SC-009-double-space.json)
  añade una comprobación del nombre exacto. No se relajaron las expectativas previas.
- La integración creó **una operación**, con el nombre exacto `Pechuga  Herbal`,
  cantidad 1, restricciones vacías y ubicación ROOM. Reenviar el mismo inbound
  no creó otra operación (`duplicate_inbound_verified: true`).

| Comprobación | Antes, punto 3 | Después, corrección 1 |
| --- | --- | --- |
| SC-009: offline e integración | Fallaba | Pasa las comprobaciones ejecutadas |
| Biblioteca offline | 14 pasan, 4 fallan, 22 pendientes | 15 pasan, 3 fallan, 22 pendientes |
| Biblioteca Spring + agente + H2 | 8 pasan, 4 fallan, 28 pendientes | 9 pasan, 3 fallan, 28 pendientes |
| Suite unitaria del agente | 332 pasan, 10 omitidas | 333 pasan, 10 omitidas |

Reportes posteriores: [offline](../tests/conversation_regression/reports/fix1-offline.json),
[integración](../tests/conversation_regression/reports/fix1-integration.json).
Los [reportes originales del punto 3](conversation-evaluation.md) se conservan
como referencia anterior al cambio. Los resultados parciales no certifican
los escenarios completos ni los casos que aún carecen de adaptador.

### Pendientes al cerrar la corrección 1

Esta entrega corrige únicamente SC-009-double-space. Siguen fallando SC-001
(producto sin cantidad), SC-004 (cambio de ubicación) y SC-011-spa-name en las
comprobaciones controladas. Los fallos previos con modelo real de spa y eliminación
de artículos tampoco están corregidos ni reevaluados.

El siguiente arreglo corresponde al cambio de ubicación. Cada arreglo debe
conservar su evidencia anterior/posterior y verificar los recorridos relacionados.
La protección automática de cambios de prompts del punto 4 y la barrera de
despliegue del punto 7 siguen pendientes. Todo este cambio está en el checkout
local de sandbox; no se publicó ni desplegó.

## 2. Cambiar la ubicación de entrega sin perder el pedido (SC-004)

Garantías protegidas: **BC-003** (editar solo el campo solicitado), **BC-006**
(confirmar el borrador vigente) y **BC-013** (conservar datos originales).

### Causa y cambio

El selector de catálogo solo consideraba campos vacíos y se desactivaba al
esperar confirmación. Una vez guardada POOL_1, «I want to receive in my room»
terminaba en el extractor de artículos, que no podía cambiar la ubicación.

- [catalog_selection.py](../app/services/catalog_selection.py) ahora expone la
  ubicación actual y sus opciones configuradas durante la edición de un borrador
  multilenguaje de ROOM_SERVICE. No introduce destinos fijos ni traduce códigos.
  Las operaciones que ya arrancan y las respuestas dirigidas a tareas conservan
  sus exclusiones; los otros offerings conservan la selección de campos pendientes.
- [v2_turn_planner.py](../app/agents/v2_turn_planner.py) reutiliza la selección de
  catálogo y pide una nueva confirmación. Conserva los artículos exactamente,
  incluidos espacios y restricciones. Una ubicación ambigua conserva el borrador
  y bloquea su confirmación hasta aclararse; editar cantidades mientras tanto no
  resuelve esa ambigüedad.
- El prompt **V2_HOTEL_SCOPE**, en
  [v2_scope_router.py](../app/agents/v2_scope_router.py), recibe `currentValue`
  además de las opciones. Se añadieron tres líneas para distinguir una ubicación
  nueva de una petición genérica de editar o confirmar. **Cero líneas previas
  eliminadas**, comprobado contra HEAD. El [registro del cambio de prompt](../tests/conversation_regression/reports/fix2-prompt-change.json)
  guarda hashes, conteos y pruebas asociadas, sin copiar el prompt completo.

### Evidencia y protecciones

Las nuevas pruebas de edición y ambigüedad fallaron antes del cambio. Las pruebas
de [test_catalog_selection.py](../tests/test_catalog_selection.py) cubren edición
directa y después de «Cambiar», conservación exacta de artículos, confirmación
posterior, evidencia incompleta, destinos desconocidos, confianza baja,
alternativas ambiguas y recuperación mediante una selección válida. Un caso
adicional comprueba que cambiar cantidades después de una ubicación ambigua
mantenga pendiente la selección del destino.

El [caso SC-004](../tests/conversation_regression/fixtures/cases/SC-004-edit-location.json)
conserva las entradas del huésped y todas sus expectativas previas. Se actualizó
el guion del modelo a la selección de ubicación —antes simulaba el envío al
extractor equivocado—, se comprueban todos los artículos y se añadió el turno de
confirmación con comparación exacta del input. La interpretación de la frase
se comprobó además con el modelo real, sin usar ese guion.

En Spring + Python + H2 se creó una operación con destino ROOM, un taco y la
restricción «sin cebolla». Reenviar el inbound no creó otra operación.

| Comprobación | Antes, corrección 1 | Después, corrección 2 |
| --- | --- | --- |
| SC-004: offline e integración | Fallaba | Pasa las comprobaciones ejecutadas |
| Biblioteca offline | 15 pasan, 3 fallan, 22 pendientes | 16 pasan, 2 fallan, 22 pendientes |
| Biblioteca Spring + agente + H2 | 9 pasan, 3 fallan, 28 pendientes | 10 pasan, 2 fallan, 28 pendientes |
| Suite unitaria del agente | 333 pasan, 10 omitidas | 338 pasan, 10 omitidas |
| SC-004 con gpt-4.1-mini, versión final | No evaluado | 2 repeticiones aprobadas |

Reportes finales: [offline](../tests/conversation_regression/reports/fix2-offline-final.json),
[integración](../tests/conversation_regression/reports/fix2-integration-final.json),
[modelo real](../tests/conversation_regression/reports/fix2-live-final.json) y
[suite unitaria](../tests/conversation_regression/reports/fix2-unit-final.log).
Los reportes `fix2-offline`, `fix2-integration` y `fix2-live` conservan la corrida
intermedia, antes de añadir la protección de edición de cantidades tras una
ubicación ambigua.

Las dos corridas con modelo real usaron 3 llamadas y 5,119 tokens cada una:
**6 llamadas adicionales**, para un acumulado de **24 de las 30 autorizadas**
(35,023 tokens contando el punto 3). La interpretación se ejecutó por separado
en cada repetición; la caché de traducción se comparte dentro de cada corrida,
como declara el reporte.

### Alcance y siguiente corrección

Este arreglo cubre cambios de ubicación por separado, con código, etiqueta,
botón o frase interpretada contra el catálogo. No acredita mensajes que mezclen
simultáneamente cambio de ubicación y cambios de artículos, ni todas las
variantes lingüísticas. Las pruebas con modelo real aquí usan la frase en inglés
del incidente; los casos ambiguos están verificados con respuestas controladas.

Siguen pendientes SC-001 (producto sin cantidad), SC-011-spa-name y los fallos
previos con modelo real de spa y eliminación de artículos. El siguiente arreglo
es conservar un producto incompleto y resolver la respuesta «Uno».
La protección automática de prompts del punto 4 y la barrera de despliegue del
punto 7 siguen pendientes. Esta comparación puntual de prompts no las sustituye.

La combinación de pruebas controladas, integración y evaluación del modelo
sigue las [buenas prácticas de evaluación de OpenAI](https://developers.openai.com/api/docs/guides/evaluation-best-practices).

## 3. Conservar productos sin cantidad y completar «Uno» (SC-001 / SC-002)

Garantías protegidas: **BC-001** (captura parcial), **BC-002** (respuestas cortas
con referente inequívoco), **BC-003** (conservar los otros artículos) y **BC-006**
(no ejecutar un pedido incompleto).

### Causa y cambio

El extractor identificaba el producto y sus restricciones, pero el validador
descartaba el resultado entero cuando faltaba una cantidad. El siguiente mensaje
llegaba sin ese artículo en el borrador; además, la coerción de artículos exigía
cantidades completas incluso al preparar el contexto de captura.

[input_understanding.py](../app/services/input_understanding.py) permite ahora
artículos parciales únicamente cuando el llamador habilita `allow_partial`.
El borrador conserva el nombre y las restricciones, y omite la cantidad hasta
recibir evidencia válida. No introduce cantidades por defecto. Los cambios de
cocina siguen exigiendo un pedido completo, incluso si se intenta combinar los
flags de captura parcial y reemplazo completo.

Al completar una cantidad, una respuesta sin nombre solo puede dirigirse al
único artículo pendiente. Si hay varios pendientes, o el modelo elige otro
artículo completo sin identificarlo, se rechaza esa asignación y se pide aclarar.
El contexto enviado al extractor incluye los artículos incompletos.

[v2_turn_planner.py](../app/agents/v2_turn_planner.py) conserva el borrador durante
la captura inicial, las respuestas posteriores y la selección de ubicación.
Pregunta por los nombres que aún carecen de cantidad. Solo presenta confirmación
cuando tiene cantidades y ubicación; completar cantidades tampoco resuelve una
ubicación que había quedado ambigua.

El prompt **V2_ORDER_UNDERSTANDING** recibió seis líneas nuevas y ninguna línea
previa eliminada. La política `ORDER_QUANTITY_POLICY` se conserva. Se añadió una
plantilla de pregunta por cantidades pendientes y se incrementó la versión del
registro de mensajes de 4 a 5. El [registro de cambio](../tests/conversation_regression/reports/fix3-prompt-change.json)
incluye hashes del prompt y del archivo de plantillas, reglas y pruebas vinculadas.

### Evidencia

Las pruebas nuevas reprodujeron la pérdida de artículos antes del cambio. Ahora
cubren producto → cantidad corta → confirmación, restricciones conservadas,
selección de ubicación con un artículo incompleto, canasta mixta, asignación
ambigua de cantidades y la exigencia de pedido completo para cocina.

SC-001 conserva su guion de extracción y sus expectativas originales; añade
comprobaciones de cantidad ausente y confirmación deshabilitada, y el turno final
de confirmación con input exacto. SC-002, antes pendiente de guion, ahora ejecuta
su conversación completa: «gorditas y una sopa de pollo» → «Dos gorditas» →
habitación → confirmar. Sus expectativas originales se conservan y se amplían.
Los estados de ejecución anotados durante el punto 2 en el contrato son
históricos; los reportes de esta sección acreditan la cobertura actual de SC-002.

| Comprobación | Antes, corrección 2 | Después, corrección 3 |
| --- | --- | --- |
| SC-001: offline e integración | Fallaba | Pasa las comprobaciones ejecutadas |
| SC-002: offline e integración | Sin ejecutar | Pasa las comprobaciones ejecutadas |
| Biblioteca offline | 16 pasan, 2 fallan, 22 pendientes | 18 pasan, 1 falla, 21 pendientes |
| Biblioteca Spring + agente + H2 | 10 pasan, 2 fallan, 28 pendientes | 12 pasan, 1 falla, 27 pendientes |
| Suite unitaria del agente | 338 pasan, 10 omitidas | 342 pasan, 10 omitidas |
| SC-001 con gpt-4.1-mini | No evaluado | 1 conversación aprobada |

Reportes: [offline](../tests/conversation_regression/reports/fix3-offline.json),
[integración](../tests/conversation_regression/reports/fix3-integration.json),
[modelo real](../tests/conversation_regression/reports/fix3-live.json) y
[suite unitaria](../tests/conversation_regression/reports/fix3-unit.log).

Spring creó una operación para cada caso con las cantidades correctas y los
artículos completos; reenviar su inbound no duplicó ninguna. La corrida real
mantuvo el idioma italiano y el nombre original «Gorditas de pulpo», preguntó
únicamente por su cantidad y resolvió «Uno» como 1 antes de pedir confirmación.
Se revisaron esos mensajes además de las aserciones automáticas.

Esta corrida real consumió **6 llamadas y 8,069 tokens**. El acumulado autorizado
queda en **30 de 30 llamadas**, 43,092 tokens. No queda presupuesto de llamadas
bajo aquella autorización. No se ejecutó la canasta mixta con modelo real ni se
repitió la conversación italiana: esos alcances siguen sin certificarse.

### Pendientes

El fallo controlado restante es SC-011-spa-name. También siguen pendientes los
fallos observados con modelo real en captura de spa y eliminación de artículos,
y los casos sin adaptador. Las pruebas de espacios dobles y cambio de ubicación
siguen pasando en las corridas de esta corrección.

Todo continúa en el checkout local de sandbox. No se ha desplegado ni completado
la protección automática de cambios de prompts o la barrera de despliegue.

## 4. Tratamiento, hora y aclaraciones de spa (SC-003 / SC-011)

Esta es la cuarta **corrección de errores**, no el punto 4 del plan general de
protección automática de prompts, que sigue pendiente.

Garantías protegidas: **BC-002**, **BC-003**, **BC-006**, **BC-007** y **BC-013**.

### Causas y cambios

Los reportes del modelo real mostraban dos respuestas concretas: una extraía
«Masaje herbal» pero citaba la oración con la fecha, y otra marcaba «6pm» como
ambiguo. La primera chocaba con una comparación que exigía que nombre y cita
fueran idénticos. Además, el mensaje inglés de aclaración concatenaba directamente
el identificador del campo con una instrucción genérica sobre fechas y horas.

En [spa_turns.py](../app/agents/spa_turns.py):

- El tratamiento puede ser una subcadena exacta de una cita original más larga.
  La cita debe seguir perteneciendo al mensaje actual; no se aceptan traducciones
  del nombre. Las fechas ISO dentro de citas más largas también se comprueban
  contra el valor extraído, evitando aceptar una fecha diferente.
- Una respuesta formada únicamente por una hora explícita AM/PM o una hora de
  24 horas inequívoca se interpreta sin extracción del modelo. «6pm» es 18:00;
  «12am» es 00:00 y «12pm» es 12:00. Conserva los demás campos y ambigüedades y
  aplica las comprobaciones existentes del reloj y zona horaria del hotel.
  Horas como «6» o «6:00», rangos, negaciones, condiciones y formatos inválidos
  no se resuelven por esa vía automática.
- Las aclaraciones en inglés preguntan específicamente por tratamiento, fecha
  u hora. No muestran `serviceName`, `reservationDate` o `reservationTime` ni
  piden una fecha para aclarar un tratamiento.

El prompt **V2_SPA_EXTRACTION** recibió cuatro líneas nuevas, sin eliminar
instrucciones anteriores. El [registro del cambio](../tests/conversation_regression/reports/fix4-prompt-change.json)
incluye hashes, garantías y pruebas vinculadas. Las reglas de baja confianza,
ambigüedad, evidencia original y reloj del hotel permanecen.

### Pruebas y resultados

Las pruebas nuevas reprodujeron el rechazo del tratamiento y las aclaraciones
incorrectas antes del arreglo. Después cubren citas amplias, traducciones y fechas
alteradas rechazadas, AM/PM, dígitos no latinos, horarios imposibles, fechas pasadas,
conservación de otras ambigüedades y preguntas específicas.

SC-003 conserva sus expectativas previas y añade la comprobación del tratamiento
después de recibir la hora. Su guion deja de simular la extracción de «6pm», porque
esa llamada ya no existe. La prueba Java de integración ahora reconoce el botón
real de confirmación de spa con su token, lo pulsa y comprueba el input persistido.
Creó una sola operación para Masaje herbal, 2026-10-03, 18:00; repetir el inbound
no la duplicó.

Se añadió [SC-011-spa-treatment-alternatives](../tests/conversation_regression/fixtures/cases/SC-011-spa-treatment-alternatives.json),
enlazado al contrato: el huésped aún no decide entre masaje y facial, pero ya tiene
fecha y hora. La biblioteca actual contiene **41 casos de 21 escenarios**.

| Comprobación | Antes, corrección 3 | Después, corrección 4 |
| --- | --- | --- |
| Biblioteca offline | 18 pasan, 1 falla, 21 pendientes | 20 pasan, 0 fallan, 21 pendientes |
| Spring + agente + H2 | 12 pasan, 1 falla, 27 pendientes | 14 pasan, 0 fallan, 27 pendientes |
| Suite unitaria del agente | 342 pasan, 10 omitidas | 347 pasan, 10 omitidas |
| Modelo real: 3 casos de spa, 2 repeticiones cada uno | SC-003 fallaba en el punto 3 | 6 aprobaciones parciales |

Reportes actuales: [offline](../tests/conversation_regression/reports/fix4-offline-final.json),
[integración](../tests/conversation_regression/reports/fix4-integration-final.json),
[modelo real original](../tests/conversation_regression/reports/fix4-live.json),
[modelo real: alternativas](../tests/conversation_regression/reports/fix4-live-ambiguity.json)
y [suite unitaria](../tests/conversation_regression/reports/fix4-unit.log).
Los reportes sin sufijo `final` de offline e integración corresponden a los 40
casos originales, antes de incorporar la nueva conversación de alternativas.

En ambas repeticiones de SC-003 el agente pidió solo la hora, conservó el tratamiento
y la fecha, mostró 18:00 y solicitó confirmación. En SC-011-spa-name, «Un Masaje
herbal» fue reconocido como un tratamiento claro y se pidió la fecha: esas dos
corridas no prueban una ambigüedad del nombre. Por eso se añadió el caso de dos
tratamientos alternativos. En sus dos repeticiones el modelo pidió únicamente
aclarar el tratamiento, conservó fecha y hora y no intentó iniciar el servicio.
Se revisaron los mensajes además de las comprobaciones automáticas.

La primera corrida utilizó **13 llamadas / 17,161 tokens** y la de alternativas
**5 llamadas / 6,439 tokens**: **18 de las 20 llamadas adicionales autorizadas**.
Quedan 2 llamadas de esa ampliación. Acumulado de este trabajo: 48 llamadas y
66,692 tokens. La interpretación se repitió por conversación; la caché de
traducción se comparte dentro de cada corrida, como declara el ejecutor.

### Alcance pendiente

No hubo fallos entre las comprobaciones controladas ejecutadas; los 21 casos
offline y 27 de integración marcados pendientes no cuentan como aprobados.
La rama conversacional SC-011-spa-date sigue sin adaptador/guion completo, aunque
hay pruebas unitarias de fechas ambiguas y horarios inválidos.

El siguiente fallo ya reproducido es **eliminar un artículo con el modelo real**
(SC-005). También siguen pendientes completar la protección automática de prompts,
los adaptadores de conversaciones restantes y la barrera de despliegue. No se ha
publicado ni desplegado esta corrección.

## 5. Eliminar un artículo conservando el resto del pedido (SC-005)

Garantías protegidas: **BC-003** (edición específica), **BC-006** (confirmar el
pedido actualizado) y **BC-007** (no volver a preguntar lo ya indicado).

### Causa y corrección

En las dos corridas originales, el clasificador devolvía simultáneamente
`hasRequestDetails=true` y `replyAction=CHANGE` para «Quita la sopa». El agente
interpretaba CHANGE como una petición genérica y preguntaba qué modificar sin
invocar el extractor de artículos.

[input_understanding.py](../app/services/input_understanding.py) ahora rechaza
esa decisión contradictoria: si hay detalles concretos, no registra CHANGE como
una acción genérica. El mensaje sigue hacia la extracción y validación del cambio.
Una petición sin detalles mantiene el comportamiento anterior y pide especificarlos.
No se sustituyó la interpretación por una lista de verbos en español o inglés.

El prompt **V2_HOTEL_SCOPE** en
[v2_scope_router.py](../app/agents/v2_scope_router.py) recibe cuatro líneas que
distinguen la eliminación concreta de la petición genérica de editar. No se
eliminaron instrucciones anteriores, incluidas las añadidas para cambiar ubicación.
El [registro del cambio](../tests/conversation_regression/reports/fix5-prompt-change.json)
encadena el hash anterior con el de la corrección 2 y vincula reglas y pruebas.

### Evidencia

La nueva prueba reprodujo el fallo antes de modificar el código. Ahora comprueba
que incluso con el resultado contradictorio del clasificador se elimine la sopa,
se conserven dos hamburguesas sin queso y la ubicación, y se pida una nueva
confirmación antes de proponer la creación del pedido.

Se añadieron controles para conservar la pregunta de cambio genérico y para que
intentar eliminar el último artículo no cancele ni confirme automáticamente el
pedido. Este último conserva el comportamiento restrictivo existente: mantiene
el borrador y solicita aclaración, sin iniciar un pedido vacío.

SC-005 mantiene las entradas y expectativas originales. Su guion ahora reproduce
la contradicción observada en el modelo real, en lugar de simular una clasificación
ideal. Añade la comprobación de confirmación pendiente y el turno final que verifica
el input exacto del pedido actualizado.

| Comprobación | Resultado actual |
| --- | --- |
| Suite unitaria | 350 aprobadas, 10 omitidas |
| Biblioteca offline | 20 aprobaciones parciales, 0 fallos, 21 pendientes |
| Spring + agente + H2 | 14 aprobaciones parciales, 0 fallos, 27 pendientes |
| SC-005 con gpt-4.1-mini | 2 conversaciones completas aprobadas |

Reportes: [offline](../tests/conversation_regression/reports/fix5-offline.json),
[integración](../tests/conversation_regression/reports/fix5-integration.json),
[modelo real](../tests/conversation_regression/reports/fix5-live.json) y
[suite unitaria](../tests/conversation_regression/reports/fix5-unit.log).

En ambas repeticiones reales, «Quita la sopa» se clasificó con detalles concretos
y acción NONE. El agente presentó únicamente las dos hamburguesas sin queso y
pidió confirmar; no volvió a preguntar qué cambiar. La integración persistió una
operación con ese input exacto y el reenvío no la duplicó. La ruta de clasificación
contradictoria queda probada por la regresión controlada, aunque ya no apareció
en esas dos repeticiones del modelo.

Esta evaluación consumió **12 llamadas / 16,892 tokens**, dentro de las 20
adicionales autorizadas. Quedan 8 de esa ampliación más las 2 anteriores:
**10 llamadas disponibles** bajo las autorizaciones de esta conversación.
Acumulado: 60 llamadas y 83,584 tokens.

### Siguiente paso

Los fallos reproducidos que motivaron estas cinco correcciones tienen ahora
evidencia favorable en las comprobaciones descritas. Esto no certifica todas las
formas de expresarse ni los casos aún sin adaptador: el extractor sigue siendo
responsable de identificar correctamente el artículo, las negaciones y el alcance
del cambio. Las dos conversaciones reales de esta corrección están en español.

Corresponde retomar el **punto 4 del plan general: protección automática de cambios
de prompts**, y después continuar con la cobertura pendiente y la barrera de
despliegue. Las auditorías puntuales guardadas aquí todavía no constituyen ese
bloqueo automático. Todo sigue en el checkout local de sandbox, sin despliegue.

## 6. Interrumpir un pedido y retomarlo sin perder datos (SC-006)

Después de implementar el punto 4, se retomó la cobertura pendiente del punto 5.
Garantías: **BC-004** (interrupciones conservan solicitudes), **BC-001/002/007**
(conservar captura parcial y preguntar solo lo que falta) y **BC-006/013**
(confirmación actual y datos originales).

### Fallo reproducido y corrección

El primer turno de SC-006 falló antes del cambio:
[fix6-before.json](../tests/conversation_regression/reports/fix6-before.json).
Al preguntar por la alberca, el estado activo cambiaba de ROOM_SERVICE a FAQ.
Después de responder o derivar la FAQ, el resumen quedaba vacío. Los datos del
pedido no quedaban disponibles como un borrador independiente para retomarlo.

[v2_turn_planner.py](../app/agents/v2_turn_planner.py) ahora conserva el pedido
interrumpido en `roomServiceDraft`, con sus artículos, cantidades, restricciones,
ubicación y fase pendiente. El borrador suspendido pierde su autorización anterior
de confirmar. Volver a room service recupera esos datos: si están completos presenta
la confirmación; si falta cantidad, ubicación o artículos pregunta por ese dato.
La copia suspendida se elimina al reactivarse. Una cancelación o un inicio exitoso
no vuelve a crear el borrador ya cerrado.

No se cambiaron instrucciones de prompts. Los snapshots efectivos preexistentes
permanecen idénticos. El verificador bloqueó la suite por el cambio de fuentes y
pruebas, y luego se actualizó deliberadamente la referencia de sandbox tras
revisar el [diff](../tests/conversation_regression/reports/fix6-prompt-review/review.md)
y registrar la [revisión local](../tests/conversation_regression/reports/fix6-prompt-review/acceptance.json).
Se conservaron las 14 anclas críticas. Esto no constituye aprobación humana ni
autorización de despliegue.

### Nuevas pruebas y adaptadores

- SC-006-interrupt-resume ya ejecuta pedido → FAQ sin respuesta aprobada →
  recepción → retomar → confirmar. Comprueba datos conservados y una operación
  por cada servicio; el input persistido de ROOM_SERVICE coincide con el pedido.
- SC-006-partial-resume interrumpe un pedido sin cantidad y comprueba que, al
  volver, se pregunte por esa cantidad sin volver a pedir la ubicación.
- Ocho pruebas unitarias cubren además reanudación por botón, pedido aún sin
  artículos, confirmación obsoleta, cancelación, cierre, conservación frente a
  resúmenes ajenos y resultados de herramientas vinculados a llamadas reales.

El adaptador de agente admite resultados **simulados** y explícitos de búsqueda
FAQ vacía e inicio exitoso, asociados al `toolCallId` realmente propuesto. Nunca
copia el estado esperado de las aserciones. La observación `environment.source`
distingue esa simulación de `spring-h2`.

En integración, Spring ejecuta la cadena de herramientas y persiste las operaciones
con su código real. Los pasos de resultados inspeccionan los turnos capturados y
el estado persistido al terminar el turno del huésped; no se presentan como una
lectura de commits intermedios. Conocimiento, BPM y entrega externa siguen simulados.

### Resultados actuales

| Comprobación | Resultado |
| --- | --- |
| Suite offline completa | 374 aprobadas, 10 opt-in omitidas |
| Biblioteca offline, 42 casos | 22 aprobaciones parciales, 20 NOT_RUN, 0 fallos |
| Spring + agente + H2 | 16 aprobaciones parciales, 26 NOT_RUN, 0 fallos |
| SC-006, ambas variantes | Aprobadas en replay e integración |
| Modelo real para SC-006 | Ambas variantes aprobadas, una ejecución por variante |

Evidencia: [suite](../tests/conversation_regression/reports/fix6-unit.log),
[offline](../tests/conversation_regression/reports/fix6-offline.json) e
[integración](../tests/conversation_regression/reports/fix6-integration.json).
La corrida completa de integración conserva también los recorridos anteriores.

El intento inicial con OpenAI fue bloqueado por la revisión automática de permisos
y no consumió API. Después, el usuario autorizó explícitamente esta verificación
con un máximo de 10 llamadas. Los reportes anteriores al permiso se conservan como
evidencia histórica; el cierre se registra en
[fix6-validation.json](../tests/conversation_regression/reports/fix6-validation.json).

Con `gpt-4.1-mini`, el [recorrido completo](../tests/conversation_regression/reports/fix6-live.json)
pasó sus ocho pasos usando 5 llamadas y 5,874 tokens. Al retomar, presentó los dos
tacos y la habitación para una nueva confirmación; no pidió repetir esos datos.
La [variante parcial](../tests/conversation_regression/reports/fix6-live-partial.json)
pasó sus cuatro pasos usando 4 llamadas y 5,291 tokens. Preguntó únicamente por
la cantidad de tacos y conservó la ubicación, sin autorizar el inicio del pedido.

Total de esta verificación: **9 llamadas / 11,165 tokens**, dentro del máximo
autorizado de 10; queda 1 llamada de esta autorización específica. El acumulado
de las evaluaciones realizadas es de 69 llamadas y 94,749 tokens. No se cambiaron
fuentes, fixtures, instrucciones ni la referencia de prompts durante esta corrida.
Las huellas de ambos reportes coinciden con el código validado en sandbox.

En estas corridas el modelo y la localización son reales; los resultados de
herramientas y las operaciones son simulados. La persistencia real se comprobó
por separado con Spring/H2. Una ejecución por variante no demuestra estabilidad
estadística; la FAQ con respuesta aprobada y otras combinaciones de interrupción
siguen pendientes. No hubo despliegue ni contacto con servicios reales del hotel.

## 7. Cobertura de cambio de idioma con solicitud pendiente (SC-007)

Este paso amplía pruebas; **no se encontró un fallo del agente en los recorridos
evaluados ni se cambió código de producción o instrucciones de prompts**.
Reglas principales: BC-005 (idioma separado del estado), BC-001/002/007 (captura
parcial y preguntas pertinentes), BC-006/013 (confirmación y datos originales).

SC-007-language-partial deja de ser un caso pendiente: cambia de español a inglés
con una sopa sin cantidad, recibe «Two», conserva el nombre original y la ubicación,
pide confirmar e inicia el pedido con dos sopas. SC-007-language-switch conserva
un pedido ya completo durante el cambio de idioma y confirma ese mismo pedido.
Ambos verifican hasta el resultado de inicio, incluidos idioma e input exacto.

Las tres pruebas nuevas en [test_pending_language_switch.py](../tests/test_pending_language_switch.py)
comprueban además que el cambio de idioma no complete tareas abiertas de spa o
mantenimiento, que «Dos» no deshaga la preferencia explícita de inglés y que una
segunda petición expresa pueda volver al español sin perder restricciones.
La comprobación de tareas simultáneas es unitaria; la integración de SC-007 usa
un pedido pendiente, sin sembrar operaciones ajenas.

La preparación inicial de la fixture parcial duplicó una aserción incompatible
(una sopa y dos sopas en el mismo input). Se corrigió la expectativa a dos, como
indica «Two». [sc007-initial-offline.json](../tests/conversation_regression/reports/sc007-initial-offline.json)
conserva ese diagnóstico del oráculo, que no fue un fallo del producto.

### Evidencia

| Comprobación | Resultado |
| --- | --- |
| Suite offline completa | 377 aprobadas, 10 opt-in omitidas |
| Biblioteca offline, 42 casos | 23 aprobaciones parciales, 19 NOT_RUN, 0 fallos |
| SC-007 en Spring + agente + H2 | Ambas variantes aprobadas |
| SC-007 con gpt-4.1-mini | Ambas variantes aprobadas, una ejecución por variante |

Reportes: [suite](../tests/conversation_regression/reports/sc007-unit.log),
[offline](../tests/conversation_regression/reports/sc007-offline.json),
[integración de SC-007](../tests/conversation_regression/reports/sc007-focused-integration.json),
[modelo real](../tests/conversation_regression/reports/sc007-live.json) y
[cierre](../tests/conversation_regression/reports/sc007-validation.json).

En el modelo real, el pedido incompleto pasó a «2 x sopa», con ubicación Room y
confirmación en inglés. El caso completo conservó una sopa. No se repitió la
pregunta de producto ni ubicación. Se verificaron las aserciones de los siete
pasos de ambas conversaciones y se revisaron sus mensajes visibles.

Consumo: **6 llamadas / 9,142 tokens**, dentro de las 12 autorizadas específicamente
para SC-007. Quedan 6 de esa autorización. Acumulado realizado: 75 llamadas y
103,891 tokens. El caché de traducción se compartió dentro de la corrida, como
declara la configuración del reporte; las clasificaciones se ejecutaron en ambos casos.

El [diff de protección](../tests/conversation_regression/reports/sc007-prompt-review/review.md)
registra fixtures, pruebas y nuevas capturas efectivas; todos los archivos del
agente y snapshots efectivos anteriores permanecen idénticos. Se revisó y aceptó
la nueva referencia local con sus 14 anclas críticas intactas, sin aprobación de
despliegue. La integración se ejecutó antes de actualizar esta referencia; sus
huellas de agente, biblioteca y contrato conductual coinciden con las finales.

La prueba con modelo real simula resultados de herramientas; Spring/H2 comprueba
la persistencia por separado con respuestas del modelo controladas. Una ejecución
por variante no demuestra estabilidad estadística ni cobertura de todos los idiomas.
No se desplegó ni se ejecutaron servicios reales del hotel.

## 8. Botones antiguos y usados de Room Service (SC-009)

Se reprodujo un fallo real de BC-006: el huésped recibió la confirmación de una
sopa, cambió a dos y pulsó el primer botón. El agente propuso START_SERVICE con
dos sopas. [sc009-before.json](../tests/conversation_regression/reports/sc009-before.json)
conserva esa ejecución anterior a la corrección.

Cada menú de confirmación emitido ahora tiene un identificador nuevo, guardado en
el resumen junto con una huella del pedido exacto y la conversación. Confirmar,
Cambiar y Cancelar solo actúan sobre el menú vigente. Un botón anterior conserva
los datos y presenta la confirmación actual; uno ya usado no inicia otro servicio.
Volver de un pedido A a B y luego a A tampoco reactiva el primer menú. Cambiar
únicamente de idioma conserva la autorización del menú actual. La aprobación por
texto sigue pasando por la interpretación y validación existentes.

La comprobación se realiza antes del planificador y también al validar START_SERVICE:
un botón debe ser Confirmar y su input debe coincidir exactamente con lo mostrado.
Los botones heredados sin versión requieren una confirmación nueva. Esto es una
decisión de migración deliberada: su identificador antiguo no permite saber qué
versión del pedido vio el huésped.

La fixture deja de ser NOT_RUN. Sus seis pasos obtienen los identificadores de las
respuestas realmente emitidas, mediante `reply_from_turn`; no fabrican tokens ni
inyectan expectativas entre turnos. Spring persiste mensajes, resumen, herramientas
y operaciones en H2. El adaptador reenvía el identificador anterior como otro
mensaje entrante, por lo que no depende de deduplicar un mismo evento.

Las once pruebas nuevas cubren los tres botones antiguos, cambio/cancelación
vigentes, migración, cancelación sin resurrección, A→B→A, cambio de idioma,
confirmación por texto, otra conversación, datos modificados y planes inválidos.
Los tests anteriores de confirmación positiva seleccionan ahora un botón emitido;
los negativos mantienen sus identificadores inválidos.

### Evidencia final

| Comprobación | Resultado |
| --- | --- |
| Suite sin red | 388 aprobadas, 10 opt-in omitidas |
| Biblioteca offline, 42 casos | 24 aprobaciones parciales, 18 NOT_RUN, 0 fallos |
| Spring + agente + H2, 42 casos | 18 aprobaciones parciales, 24 NOT_RUN, 0 fallos |
| Protección de prompts | 226 documentos, 14 anclas críticas conservadas |

Reportes: [suite](../tests/conversation_regression/reports/sc009-unit.log),
[offline](../tests/conversation_regression/reports/sc009-offline.json),
[integración](../tests/conversation_regression/reports/sc009-final-integration.json),
[revisión](../tests/conversation_regression/reports/sc009-final-prompt-review/review.md)
y [cierre](../tests/conversation_regression/reports/sc009-validation.json).
La revisión inicial se conserva en `sc009-prompt-review`; la referencia aceptada
es la revisión final e incluye la comprobación adicional de input y acción.
No se eliminaron instrucciones de prompts. Cambiaron el código, los guiones y el
contexto efectivo que contiene los identificadores y mensajes nuevos.

No se consumió API: SC-009 exige offline e integración; las respuestas de modelos
y la entrega externa permanecen simuladas. No se afirma cobertura de concurrencia
ni de todos los menús de otros servicios. No hubo despliegue.

No se encontró una opción oficial documentada para deshabilitar visualmente un
botón de WhatsApp después del envío. Se dejó fuera, conforme a lo solicitado.
La [referencia de mensajes interactivos alojada por Meta](https://whatsapp.github.io/WhatsApp-Nodejs-SDK/api-reference/messages/interactive/)
describe IDs de respuesta, pero es de un SDK archivado y no basta para afirmar
que ninguna versión futura de la API pueda ofrecer esa función. La protección
implementada funciona en el agente aunque el cliente siga mostrando el botón.

## 9. Duplicados, fallos de inicio y recuperación del turno (SC-010)

Se hicieron ejecutables las pruebas de evento/herramienta repetidos, conflicto de
argumentos y fallo de inicio. Se añadió `SC-010-retry-after-commit`: el servicio se
crea y confirma su transacción, pero se interrumpe el ejecutor antes de guardar el
resultado de la herramienta. El reintento usa el mismo turno, propuesta y evidencia.

La [corrida anterior a las correcciones](../tests/conversation_regression/reports/sc010-before-integration.json)
mostró dos garantías ya existentes y dos fallos:

- Repetir evento y herramienta reutilizaba el resultado, con una sola operación.
- Reutilizar el ID con otra cantidad producía `IdempotencyConflictException`.
- Un fallo técnico de START_SERVICE interrumpía el runtime antes de entregar una
  respuesta al huésped.
- Tras recuperar una interrupción posterior a la creación, quedaba una sola
  operación, pero el resumen volvía a AWAITING_CONFIRMATION: el runtime recuperaba
  el plan guardado sin restaurar su resumen en el contexto de la siguiente iteración.

El runtime ahora reconstruye el contexto de una cadena interrumpida desde el plan
persistido. Esto no vuelve a escribir resúmenes antiguos en la base de datos cuando
solo se reproduce un turno ya terminado. Los resultados de START_SERVICE fallidos
llegan al agente, y el backend retira START_SERVICE de las herramientas permitidas
durante el resto de esa ejecución.

El agente responde de forma determinista que no pudo confirmar el inicio, conserva
los datos y registra el resultado pendiente por ID de herramienta y offering. No
anuncia un folio exitoso ni propone otro inicio para ese offering mientras exista
el fallo pendiente. El registro sobrevive a cambios de resumen y otros servicios;
las operaciones independientes que sí terminaron siguen recibiendo su confirmación.
El backend aporta el offering desde los argumentos persistidos, no desde texto del modelo.

Un fallo técnico persistido sigue siendo **no reintentable automáticamente**. El
caso recuperado automáticamente es una cadena interrumpida con resultado aún sin
guardar: el servicio de dominio puede reutilizar su resultado ya confirmado. No se
añadió una consola de conciliación ni un proceso que resuelva fallos terminales;
esa revisión sigue siendo necesaria antes de volver a iniciar el servicio afectado.

### Verificación y alcance

- Ocho pruebas nuevas del agente cubren el mensaje de fallo, preservación, botones,
  reintento por texto, validación contra planes incorrectos, varios fallos pendientes
  y éxito independiente. Una prueba del arnés impide presentar eventos backend como
  comprobados por el ejecutor offline.
- Dos pruebas nuevas del runtime verifican la restauración de la cadena y la
  entrega del fallo sin permiso para iniciar de nuevo. Las 27 pruebas seleccionadas
  de runtime y persistencia de conversación/servicios pasaron.
- Los tests existentes de acuses mantienen las aserciones contra éxitos falsos;
  para FAILED ahora esperan cero llamadas al modelo y el registro del fallo.
  REJECTED y los errores de otras herramientas conservan sus rutas anteriores.
- La suite final y los conteos de cobertura están en el
  [cierre](../tests/conversation_regression/reports/sc010-validation.json), la
  [suite](../tests/conversation_regression/reports/sc010-unit.log), la
  [biblioteca offline](../tests/conversation_regression/reports/sc010-offline.json)
  y la [integración final](../tests/conversation_regression/reports/sc010-final-integration.json).

En offline, tres casos SC-010 comprueban únicamente el prefijo del agente y enumeran
los pasos restantes en `not_run_steps`: repetir eventos, conflictos y pérdida del
resultado se ejecutan en Spring/H2. No se simula una base de datos en Python para
aprobar esas garantías. El caso de fallo de inicio sí se recorre completo con un
resultado sintético explícito. La inyección no puede ejecutarse como evaluación live.

La protección detectó los cambios antes de aceptar una referencia nueva. La
[revisión aceptada](../tests/conversation_regression/reports/sc010-reviewed-prompts/acceptance.json)
conserva las 14 anclas y todos los prompts efectivos previos: cambió el runtime,
el contrato de fixtures, los adaptadores y sus pruebas. Los intentos anteriores de
revisión quedan como evidencia histórica. El diagnóstico offline inicial detuvo
un llamado no guionado del planificador; no demuestra por sí solo un mensaje falso.
La reproducción del fallo de producto proviene de Spring/H2.

Todo queda en sandbox, sin despliegue ni consumo adicional de OpenAI. H2 comprueba
la persistencia local; el puerto BPM y la entrega están simulados. Esto cubre
repeticiones secuenciales y la interrupción definida, no carreras concurrentes ni
efectos remotos fuera de la transacción. El caso original FRONT_DESK de fallo se
concretó como ROOM_SERVICE para verificar también que el pedido se conserva;
los otros tipos de inicio fallido tienen cobertura unitaria, no este recorrido H2.
