# Eventos de entorno pendientes

Este documento define cómo interpretar los eventos no `guest` de los fixtures.
Son operaciones del futuro arnés de pruebas, **no endpoints de producción ni
herramientas nuevas del agente**. El replay actual los marca NOT_RUN.

Cada caso empieza en un entorno aislado con el perfil, resumen y operaciones
declarados. Solo esa preparación inicial puede sembrar datos. Los eventos
posteriores deben pasar por el componente que se quiere probar y observar sus
efectos reales; nunca copiar `checks` a la base de datos.

La observación combina `state`, `response`, `guest`, `text`, `tools` y
`tool_names` del adaptador V2 con un bloque `backend` calculado a partir de
persistencia y outbox. Por ejemplo, `createdOperations` cuenta filas nuevas
de cada offering respecto al inicio del caso; no cuenta llamadas propuestas.
`mutationCount` cuenta efectos de negocio, excluyendo auditoría del rechazo.
`resumableDrafts` normaliza el almacenamiento durable sin imponer una tabla.

## Resultados de herramientas

Estos eventos deben vincularse a los IDs de herramientas realmente propuestas,
guardar el resultado con la semántica de Spring y disparar TOOL_RESULTS.
Sin una propuesta pendiente inequívoca, el adaptador debe fallar.

| Acción | Preparación o ejecución |
| --- | --- |
| reply_to_last_tool | Entregar exactamente status/result/error del fixture a la última herramienta pendiente |
| provide_approved_faq | Devolver un match aprobado con sourceId, pregunta y respuesta del fixture al SEARCH_KNOWLEDGE pendiente |
| complete_faq_handoff | Completar la propuesta de FAQ para intervención humana con resultado exitoso y operación sintética |
| complete_service_start | Ejecutar START_SERVICE en el entorno aislado, devolver operación creada y actualizar snapshot |

Los resultados de inicio deben respetar el esquema real de Spring; no basta
retornar status=SUCCEEDED sin operación. Los casos FAQ verifican también la
respuesta posterior. Las URLs de conocimiento se mantienen en `.example`.

## Acciones de persistencia e identidad

| Acción | Efecto controlado |
| --- | --- |
| remember_current_confirmation | Guardar bajo alias la confirmación presentada y la versión del borrador inicial; si el borrador se sembró, materializar su mensaje de confirmación mediante el flujo real |
| deliver_saved_confirmation | Reenviar ese botón con su identidad y versión originales después de la edición; no regenerarlo desde el nuevo pedido |
| execute_proposed_tools | Ejecutar las herramientas pendientes con los IDs e idempotencia reales |
| redeliver_same_event_and_tool_ids | Repetir exactamente el evento y toolCallId anteriores, sin regenerar UUIDs |
| replay_tool_with_changed_arguments | Reutilizar toolCallId e idempotencia, sustituyendo input.items por data.items |
| execute_proposed_tools_then_fail_localization | Ejecutar y persistir el servicio; provocar fallo de localización antes de entregar el mensaje |
| retry_localization_successfully | Reintentar la entrega pendiente con localización exitosa; medir operación y outbox |
| change_route_same_guest_and_stay | Cerrar ruta/sesión actual y crear otra conservando contactThreadId, guestId y stayId |
| reassign_contact_to_new_guest | Cambiar guestId y stayId del contacto a otros UUIDs sintéticos |
| start_new_stay_same_contact | Conservar contacto y huésped; cambiar stayId a otro UUID sintético |

Los cambios de identidad deben usar el mecanismo normal del backend. No borrar
manualmente el resumen para hacer pasar el caso de aislamiento. La renovación
de ruta no implica afirmar que haya vencido la ventana de WhatsApp.

## Planes inválidos

`submit_invalid_plan` incorpora `request_message` al historial y al trigger,
conservando su ID. Aplica `allowed_tools` si existe, completa un sobre V2 válido
y envía exactamente `tool_call` al validador correspondiente. Las seis variantes
tienen datos concretos: herramienta prohibida, offering desconocido, versión
2 contra operación 3, evidencia ajena, input obligatorio ausente o confirmación
ausente. El formato del tool call se valida ya al cargar la biblioteca.

Para evaluar el lado Python puede llamarse a su validador sin ejecutar herramientas.
Para evaluar Spring se debe entrar por su frontera de aplicación de resultados,
medir rechazo y ausencia de mutaciones. Un rechazo en Python no acredita el
control independiente de Spring.

## Configuración y presentación

| Acción | Efecto controlado |
| --- | --- |
| set_offering_requirements | En FRONT_DESK agregar el campo string handoffReason al schema requerido y activar confirmación; preservar el resto del offering |
| activate_synthetic_catalog_offering | Crear ENTREGA con campo string location requerido y captura SINGLE_SELECT: ROOM/Habitación y POOL_1/Pool 1; confirmationRequired=false |
| localize_overlong_button | Crear un mensaje de confirmación con data.actionId y simular traducción que devuelve data.label (>24 caracteres); comprobar recuperación y conservación del actionId y datos de negocio |

Para `localize_overlong_button`, usar un pedido sintético de dos sopas sin sal en
ROOM como datos de negocio. El control de presentación puede ejecutarse offline
mediante el localizador y validador reales; no necesita una conexión a WhatsApp.
Para ENTREGA usar nombre «Entrega», modo PROCESS y sin herramientas especiales.

## Mensajes en cola

`persist_two_inbounds_before_processing` guarda ambos textos con IDs diferentes,
en orden temporal, sin ejecutar aún el agente. `process_first_inbound` construye
el snapshot para el primer trigger aunque el segundo ya esté persistido.
`process_second_inbound` procesa el segundo posteriormente. La evidencia de
confirmación debe corresponder a la versión efectivamente presentada al huésped,
sin autorizar retrospectivamente una edición con un mensaje futuro.

Todo adaptador debe informar FAILED, NOT_RUN o HARNESS_ERROR ante falta de
evidencia; no deducir un éxito solo porque no hubo excepción.
