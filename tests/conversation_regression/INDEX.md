# Índice de conversaciones

62 casos ligados a los 21 escenarios del contrato. Todos son obligatorios en Spring/H2.
El replay Python declara por separado los pasos y comprobaciones que necesitan backend.

| Caso | Conversación | Replay Python requerido |
| --- | --- | --- |
| [SC-001-split-quantity](fixtures/cases/SC-001-split-quantity.json) | Producto → cantidad italiana | Parcial |
| [SC-002-partial-basket](fixtures/cases/SC-002-partial-basket.json) | Artículos completos e incompletos | Parcial |
| [SC-003-spa-split-time](fixtures/cases/SC-003-spa-split-time.json) | Spa con hora en mensaje separado | Parcial |
| [SC-004-edit-location](fixtures/cases/SC-004-edit-location.json) | Cambiar lugar ya capturado | Parcial |
| [SC-005-edit-items](fixtures/cases/SC-005-edit-items.json) | Editar cantidad y quitar otro artículo | Parcial |
| [SC-006-interrupt-resume](fixtures/cases/SC-006-interrupt-resume.json) | Interrupciones conservan pedido | Parcial |
| [SC-006-partial-resume](fixtures/cases/SC-006-partial-resume.json) | Retomar pedido interrumpido pide únicamente la cantidad faltante | Parcial |
| [SC-007-language-partial](fixtures/cases/SC-007-language-partial.json) | Idioma cambia con un dato incompleto | Parcial |
| [SC-007-language-switch](fixtures/cases/SC-007-language-switch.json) | Idioma cambia; borrador no | Parcial |
| [SC-008-language-evidence](fixtures/cases/SC-008-language-evidence.json) | Los datos no piden idioma | Parcial |
| [SC-009-cancelled-draft-en](fixtures/cases/SC-009-cancelled-draft-en.json) | Cancelled draft rejects every old menu without requesting a reference (en) | Parcial |
| [SC-009-cancelled-draft-es](fixtures/cases/SC-009-cancelled-draft-es.json) | Cancelled draft rejects every old menu without requesting a reference (es) | Parcial |
| [SC-009-delivered-buttons-en](fixtures/cases/SC-009-delivered-buttons-en.json) | Botones antiguos después de entrega (en) | Parcial |
| [SC-009-delivered-buttons-es](fixtures/cases/SC-009-delivered-buttons-es.json) | Botones antiguos después de entrega (es) | Parcial |
| [SC-009-double-space](fixtures/cases/SC-009-double-space.json) | Confirmación conserva identidad con espacios dobles | Parcial |
| [SC-009-maintenance-hypothetical-en](fixtures/cases/SC-009-maintenance-hypothetical-en.json) | Hypothetical maintenance question cannot become recurrence (en) | Parcial |
| [SC-009-maintenance-hypothetical-es](fixtures/cases/SC-009-maintenance-hypothetical-es.json) | Hypothetical maintenance question cannot become recurrence (es-MX) | Parcial |
| [SC-009-maintenance-new-direct-en](fixtures/cases/SC-009-maintenance-new-direct-en.json) | New maintenance complaint with matching closed history (direct, en) | Parcial |
| [SC-009-maintenance-new-direct-es](fixtures/cases/SC-009-maintenance-new-direct-es.json) | New maintenance complaint with matching closed history (direct, es-MX) | Parcial |
| [SC-009-maintenance-new-menu-en](fixtures/cases/SC-009-maintenance-new-menu-en.json) | New maintenance complaint with matching closed history (menu, en) | Parcial |
| [SC-009-maintenance-new-menu-es](fixtures/cases/SC-009-maintenance-new-menu-es.json) | New maintenance complaint with matching closed history (menu, es-MX) | Parcial |
| [SC-009-maintenance-recurrence-en](fixtures/cases/SC-009-maintenance-recurrence-en.json) | Historical maintenance resolution asks before opening a linked recurrence | Parcial |
| [SC-009-maintenance-recurrence-es](fixtures/cases/SC-009-maintenance-recurrence-es.json) | Historical maintenance resolution asks before opening a linked recurrence | Parcial |
| [SC-009-maintenance-reference-en](fixtures/cases/SC-009-maintenance-reference-en.json) | Written recurrence: select reference, confirm original issue and prevent duplication (en) | Parcial |
| [SC-009-maintenance-reference-es](fixtures/cases/SC-009-maintenance-reference-es.json) | Written recurrence: select reference, confirm original issue and prevent duplication (es) | Parcial |
| [SC-009-maintenance-timeout](fixtures/cases/SC-009-maintenance-timeout.json) | Historical maintenance resolution asks before opening a linked recurrence | Pendiente de backend |
| [SC-009-replaced-draft-submitted](fixtures/cases/SC-009-replaced-draft-submitted.json) | Earlier draft version remains rejected after submitting its replacement | Parcial |
| [SC-009-room-lifecycle-buttons](fixtures/cases/SC-009-room-lifecycle-buttons.json) | Used confirmation stays bound through kitchen review, acceptance and delivery | Parcial |
| [SC-009-room-state-en](fixtures/cases/SC-009-room-state-en.json) | Room-service existing-order changes use persisted state (en) | Parcial |
| [SC-009-room-state-es](fixtures/cases/SC-009-room-state-es.json) | Room-service existing-order changes use persisted state (es) | Parcial |
| [SC-009-stale-confirmation](fixtures/cases/SC-009-stale-confirmation.json) | BotÃ³n de confirmaciÃ³n antiguo despuÃ©s de ediciÃ³n | Parcial |
| [SC-010-duplicate-event](fixtures/cases/SC-010-duplicate-event.json) | Evento y herramienta repetidos crean una operación | Parcial |
| [SC-010-idempotency-conflict](fixtures/cases/SC-010-idempotency-conflict.json) | Mismo ID con otros argumentos | Parcial |
| [SC-010-retry-after-commit](fixtures/cases/SC-010-retry-after-commit.json) | Reintento tras perder el resultado de un servicio ya creado | Parcial |
| [SC-010-tool-failure](fixtures/cases/SC-010-tool-failure.json) | Fallo de inicio conserva el pedido y no anuncia éxito | Parcial |
| [SC-011-spa-date](fixtures/cases/SC-011-spa-date.json) | Fecha ambigua conserva otros campos | Parcial |
| [SC-011-spa-name](fixtures/cases/SC-011-spa-name.json) | Aclarar tratamiento sin pedir fecha | Parcial |
| [SC-011-spa-treatment-alternatives](fixtures/cases/SC-011-spa-treatment-alternatives.json) | Aclarar alternativas de tratamiento conservando fecha y hora | Parcial |
| [SC-012-failed-edit](fixtures/cases/SC-012-failed-edit.json) | Fallo técnico conserva datos y bloquea confirmación | Parcial |
| [SC-012-outbox-retry](fixtures/cases/SC-012-outbox-retry.json) | Traducción fallida no repite herramientas | Parcial |
| [SC-013-route-renewal](fixtures/cases/SC-013-route-renewal.json) | Misma identidad retoma spa | Pendiente de backend |
| [SC-014-contact-reassignment](fixtures/cases/SC-014-contact-reassignment.json) | Identidad o estancia nueva no hereda datos | Pendiente de backend |
| [SC-014-new-stay](fixtures/cases/SC-014-new-stay.json) | Identidad o estancia nueva no hereda datos | Pendiente de backend |
| [SC-015-ambiguous-tasks](fixtures/cases/SC-015-ambiguous-tasks.json) | Sí con dos tareas abiertas | Parcial |
| [SC-016-forbidden-tool](fixtures/cases/SC-016-forbidden-tool.json) | Rechazar forbidden-tool | Parcial |
| [SC-016-foreign-evidence](fixtures/cases/SC-016-foreign-evidence.json) | Rechazar foreign-evidence | Parcial |
| [SC-016-missing-confirmation](fixtures/cases/SC-016-missing-confirmation.json) | Rechazar missing-confirmation | Parcial |
| [SC-016-missing-input](fixtures/cases/SC-016-missing-input.json) | Rechazar missing-input | Parcial |
| [SC-016-stale-version](fixtures/cases/SC-016-stale-version.json) | Rechazar stale-version | Parcial |
| [SC-016-unknown-offering](fixtures/cases/SC-016-unknown-offering.json) | Rechazar unknown-offering | Parcial |
| [SC-017-approved-faq](fixtures/cases/SC-017-approved-faq.json) | FAQ responde solo desde fuente aprobada | Pendiente de backend |
| [SC-017-unknown-faq](fixtures/cases/SC-017-unknown-faq.json) | FAQ desconocida deriva sin inventar | Parcial |
| [SC-018-configured-front-desk](fixtures/cases/SC-018-configured-front-desk.json) | Recepción respeta nuevos requisitos | Pendiente de backend |
| [SC-018-direct-front-desk](fixtures/cases/SC-018-direct-front-desk.json) | Recepción sin requisitos inicia directamente | Parcial |
| [SC-019-ambiguous-selection](fixtures/cases/SC-019-ambiguous-selection.json) | Dos ubicaciones no autorizan elegir | Parcial |
| [SC-019-button](fixtures/cases/SC-019-button.json) | Selección de ubicación por button | Parcial |
| [SC-019-code](fixtures/cases/SC-019-code.json) | Selección de ubicación por code | Parcial |
| [SC-019-generic-offering](fixtures/cases/SC-019-generic-offering.json) | Catálogo no depende de ROOM_SERVICE | Pendiente de backend |
| [SC-019-translation](fixtures/cases/SC-019-translation.json) | Selección de ubicación por translation | Parcial |
| [SC-020-channel-limit](fixtures/cases/SC-020-channel-limit.json) | Presentación larga conserva datos | Pendiente de backend |
| [SC-020-preserve-product](fixtures/cases/SC-020-preserve-product.json) | Traducción conserva producto y restricciones | Parcial |
| [SC-021-queued-messages](fixtures/cases/SC-021-queued-messages.json) | Mensaje futuro no confirma turno anterior | Pendiente de backend |
