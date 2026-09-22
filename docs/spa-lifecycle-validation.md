# Validación de SPA — 21 de septiembre de 2026

Resultado: **PASS** para la muestra sintética autorizada y la cobertura local existente. Los archivos de código del agente y backend coinciden con las huellas de la corrida completa anterior; no se modificó código durante esta validación.

## Modelo real

- Modelo: `gpt-5.6-terra`, razonamiento `none`, configuración usada para el sandbox.
- 12 intentos autorizados y 12 llamadas realizadas, sin reintentos del SDK: 31,591 tokens.
- 10 clasificaciones de intención: cancelar, cambiar, preguntas hipotéticas, negaciones y reservas adicionales, en español e inglés.
- 2 extracciones de cambio parcial: modificar solamente la hora conserva fecha y tratamiento en ambos idiomas y presenta un resumen antes de confirmar.
- Los cuatro casos positivos de gestión se conectaron al planificador local: ninguna mutación antes de confirmar; propuesta de la acción correcta después de una confirmación nueva; rechazo cuando cambia la versión del folio.
- Los casos negativos validan la clasificación y que no se active el manejador de cambios/cancelaciones. No evalúan una respuesta completa generada por el planificador general.

Los 12 casos pasaron. La traducción completa de respuestas, WhatsApp y las transiciones remotas de BPM no se ejecutaron en esta muestra. No se enviaron mensajes ni se ejecutaron herramientas reales del hotel.

Evidencia completa del gate y del modelo: [spa-lifecycle-validation.json](../tests/conversation_regression/reports/spa-lifecycle-validation.json). El runner local de la sesión impide sobrescribir una corrida existente y limita los intentos antes de acceder a la API.

## Pruebas locales previas

El gate completo, incluido en la evidencia anterior, pasó: 479 pruebas del agente, 601 del backend y 68 conversaciones de integración con 257 turnos. Conserva las omisiones previamente declaradas; no constituye una corrida completa de todas las pruebas con modelo real.

No se hizo push ni despliegue como parte de esta validación. Después de desplegar ambos componentes en sandbox falta comprobar el recorrido por WhatsApp con el usuario desde su teléfono.
