# Índice de conversaciones

40 casos concretos ligados a los 21 escenarios del contrato. «Disponible» significa
que hay un guion sintético offline, no que el caso pase. Ver resultados y límites
en [README](README.md) y [reporte](reports/sandbox-baseline.json).

| Caso | Conversación | Replay offline |
| --- | --- | --- |
| [SC-001-split-quantity](fixtures/cases/SC-001-split-quantity.json) | Producto → cantidad italiana | Disponible |
| [SC-002-partial-basket](fixtures/cases/SC-002-partial-basket.json) | Artículos completos e incompletos | Pendiente de guion/adaptador |
| [SC-003-spa-split-time](fixtures/cases/SC-003-spa-split-time.json) | Spa con hora en mensaje separado | Disponible |
| [SC-004-edit-location](fixtures/cases/SC-004-edit-location.json) | Cambiar lugar ya capturado | Disponible |
| [SC-005-edit-items](fixtures/cases/SC-005-edit-items.json) | Editar cantidad y quitar otro artículo | Disponible |
| [SC-006-interrupt-resume](fixtures/cases/SC-006-interrupt-resume.json) | Interrupciones conservan pedido | Replay e integración disponibles; ver corrección 6 |
| [SC-007-language-partial](fixtures/cases/SC-007-language-partial.json) | Idioma cambia con un dato incompleto | Replay, integración y modelo real; ver sección 7 de conversation-fixes |
| [SC-007-language-switch](fixtures/cases/SC-007-language-switch.json) | Idioma cambia; borrador no | Disponible |
| [SC-008-language-evidence](fixtures/cases/SC-008-language-evidence.json) | Los datos no piden idioma | Pendiente de guion/adaptador |
| [SC-009-double-space](fixtures/cases/SC-009-double-space.json) | Confirmación conserva identidad con espacios dobles | Disponible |
| [SC-009-stale-confirmation](fixtures/cases/SC-009-stale-confirmation.json) | Botón de confirmación antiguo después de edición | Pendiente de guion/adaptador |
| [SC-010-duplicate-event](fixtures/cases/SC-010-duplicate-event.json) | Evento y herramienta repetidos crean una operación | Pendiente de guion/adaptador |
| [SC-010-idempotency-conflict](fixtures/cases/SC-010-idempotency-conflict.json) | Mismo ID con otros argumentos | Pendiente de guion/adaptador |
| [SC-010-tool-failure](fixtures/cases/SC-010-tool-failure.json) | No anunciar un servicio fallido | Pendiente de guion/adaptador |
| [SC-011-spa-date](fixtures/cases/SC-011-spa-date.json) | Fecha ambigua conserva otros campos | Pendiente de guion/adaptador |
| [SC-011-spa-name](fixtures/cases/SC-011-spa-name.json) | Aclarar tratamiento sin pedir fecha | Disponible |
| [SC-012-failed-edit](fixtures/cases/SC-012-failed-edit.json) | Fallo técnico conserva datos y bloquea confirmación | Disponible |
| [SC-012-outbox-retry](fixtures/cases/SC-012-outbox-retry.json) | Traducción fallida no repite herramientas | Pendiente de guion/adaptador |
| [SC-013-route-renewal](fixtures/cases/SC-013-route-renewal.json) | Misma identidad retoma spa | Pendiente de guion/adaptador |
| [SC-014-contact-reassignment](fixtures/cases/SC-014-contact-reassignment.json) | Identidad o estancia nueva no hereda datos | Pendiente de guion/adaptador |
| [SC-014-new-stay](fixtures/cases/SC-014-new-stay.json) | Identidad o estancia nueva no hereda datos | Pendiente de guion/adaptador |
| [SC-015-ambiguous-tasks](fixtures/cases/SC-015-ambiguous-tasks.json) | Sí con dos tareas abiertas | Pendiente de guion/adaptador |
| [SC-016-forbidden-tool](fixtures/cases/SC-016-forbidden-tool.json) | Rechazar forbidden-tool | Pendiente de guion/adaptador |
| [SC-016-foreign-evidence](fixtures/cases/SC-016-foreign-evidence.json) | Rechazar foreign-evidence | Pendiente de guion/adaptador |
| [SC-016-missing-confirmation](fixtures/cases/SC-016-missing-confirmation.json) | Rechazar missing-confirmation | Pendiente de guion/adaptador |
| [SC-016-missing-input](fixtures/cases/SC-016-missing-input.json) | Rechazar missing-input | Pendiente de guion/adaptador |
| [SC-016-stale-version](fixtures/cases/SC-016-stale-version.json) | Rechazar stale-version | Pendiente de guion/adaptador |
| [SC-016-unknown-offering](fixtures/cases/SC-016-unknown-offering.json) | Rechazar unknown-offering | Pendiente de guion/adaptador |
| [SC-017-approved-faq](fixtures/cases/SC-017-approved-faq.json) | FAQ responde solo desde fuente aprobada | Pendiente de guion/adaptador |
| [SC-017-unknown-faq](fixtures/cases/SC-017-unknown-faq.json) | FAQ desconocida deriva sin inventar | Pendiente de guion/adaptador |
| [SC-018-configured-front-desk](fixtures/cases/SC-018-configured-front-desk.json) | Recepción respeta nuevos requisitos | Pendiente de guion/adaptador |
| [SC-018-direct-front-desk](fixtures/cases/SC-018-direct-front-desk.json) | Recepción sin requisitos inicia directamente | Disponible |
| [SC-019-ambiguous-selection](fixtures/cases/SC-019-ambiguous-selection.json) | Dos ubicaciones no autorizan elegir | Pendiente de guion/adaptador |
| [SC-019-button](fixtures/cases/SC-019-button.json) | Selección de ubicación por button | Disponible |
| [SC-019-code](fixtures/cases/SC-019-code.json) | Selección de ubicación por code | Disponible |
| [SC-019-generic-offering](fixtures/cases/SC-019-generic-offering.json) | Catálogo no depende de ROOM_SERVICE | Pendiente de guion/adaptador |
| [SC-019-translation](fixtures/cases/SC-019-translation.json) | Selección de ubicación por translation | Disponible |
| [SC-020-channel-limit](fixtures/cases/SC-020-channel-limit.json) | Presentación larga conserva datos | Pendiente de guion/adaptador |
| [SC-020-preserve-product](fixtures/cases/SC-020-preserve-product.json) | Traducción conserva producto y restricciones | Pendiente de guion/adaptador |
| [SC-021-queued-messages](fixtures/cases/SC-021-queued-messages.json) | Mensaje futuro no confirma turno anterior | Pendiente de guion/adaptador |
# Caso añadido durante las correcciones

- [SC-011-spa-treatment-alternatives](fixtures/cases/SC-011-spa-treatment-alternatives.json): aclarar dos tratamientos alternativos conservando fecha y hora.
- [SC-006-partial-resume](fixtures/cases/SC-006-partial-resume.json): retomar después de una FAQ preguntando únicamente la cantidad faltante. Replay e integración disponibles.
