# Protección de cambios de prompts — punto 4

Implementado en el checkout local de sandbox. No se desplegó ni se modificaron
prompts de producción como parte de este punto. La referencia inicial incorpora
las cinco correcciones documentadas en [conversation-fixes.md](conversation-fixes.md).

## Qué queda protegido

El [manifiesto](../contracts/prompts/manifest.v1.json) asigna IDs estables a 12
familias, enlaza las reglas BC del contrato con archivos de pruebas y casos de
conversación concretos. Los vínculos inexistentes, fuentes nuevas sin registrar
y capturas incompletas producen error. BC-010 (aislamiento de identidad y sesión)
pertenece al backend; no se presenta como una garantía de los prompts.

| ID | Función |
| --- | --- |
| PR-001 | Planificador V2, captura y reparación de planes inválidos |
| PR-002 | Intención, idioma, decisiones y selección de opciones |
| PR-003 | Extracción y edición de pedidos |
| PR-004 | Extracción de spa y aclaraciones |
| PR-005 | Identificar saludos sin solicitudes concretas |
| PR-006 | Seleccionar conocimiento aprobado para FAQ |
| PR-007 | Verificar la respuesta FAQ contra su fuente |
| PR-008 | Traducir la presentación conservando datos y límites del canal |
| PR-009 | Las doce capacidades del agente configurable |
| PR-010 | Constructores y consumidores de prompts del flujo anterior |
| PR-011 | Plantillas deterministas de mensajes |
| PR-012 | Política compartida de cantidades |

La [referencia versionada](../contracts/prompts/baseline/index.json) conserva:

- Prompts efectivos en conversaciones sintéticas, capturados antes de llamar al
  modelo, junto con el esquema y las opciones de respuesta. Incluye probes de
  traducción, saludo, selección/verificación FAQ y reparación del planner.
- Dos idiomas del constructor base V2, las doce capacidades y todos los
  constructores de prompts del flujo anterior con contextos sintéticos.
- Archivos completos de instrucciones, plantillas y sus dependencias locales,
  además de fixtures, contrato, pruebas vinculadas y código del verificador.

Las 14 anclas críticas se buscan en los **prompts generados**. Dejar una frase
como comentario en el código no satisface esa comprobación. Quitar una ancla del
prompt sigue fallando aunque se regenere la referencia. Cualquier otro cambio
de texto, contexto, esquema o fuente también genera un diff y bloquea el check.

Se normalizan finales de línea y los UUID generados por las pruebas; se conservan
las relaciones entre IDs y la evidencia original. Los tiempos de timeout de las
llamadas se excluyen de la captura efectiva por ser variables; su implementación
queda en la referencia de fuentes. Las credenciales y archivos `.env` no se capturan.

## Uso cotidiano

Desde la raíz del agente, con su entorno Python:

```powershell
python -m tests.prompt_contract.guard check
python -m tests.run_offline
```

El check devuelve 0 si la referencia coincide y las anclas siguen presentes;
devuelve 1 ante cambios, errores o referencias incompletas. Forma parte de la
suite habitual mediante `tests/test_prompt_contract.py`. No llama a OpenAI y
bloquea conexiones durante la captura.

Para proponer un cambio:

```powershell
python -m tests.prompt_contract.guard propose --out tests/conversation_regression/reports/prompt-review-001 --reason "Motivo concreto del cambio"
```

La carpeta debe ser nueva. Se generan `review.md` con el diff, `report.json` con
hashes antes/después, instrucciones añadidas/eliminadas y reglas/pruebas
afectadas, y `candidate/` con la posible nueva referencia. **Generar una propuesta
no la acepta**. Un cambio devuelve 1 también en este comando; el reporte permite
distinguir un cambio esperado de un error de captura. Los resultados conductuales
aparecen como `NOT_RUN`, incluso cuando existe un guion offline.

## Cómo aceptar un cambio intencional

1. Explicar por qué cambia el comportamiento y revisar `review.md`. Revisar las
   eliminaciones, el contexto y los esquemas, no únicamente las frases nuevas.
2. Mantener la garantía de cada regla afectada y agregar la regresión pertinente.
   Si una instrucción crítica cambia de redacción, modificar su ancla con
   justificación y conservar una prueba conductual. No vaciar la lista de anclas.
3. Ejecutar los casos afectados en los niveles necesarios. Una respuesta sintética
   verifica orquestación; un cambio semántico requiere evaluación con modelo real.
   Los casos sin adaptador siguen pendientes y no se contabilizan como éxitos.
4. Generar una propuesta nueva después de la última modificación. Incluir en la
   revisión las evidencias, fallos y pendientes, además del diff del manifiesto.
5. Tras revisar, sustituir deliberadamente `contracts/prompts/baseline/` por el
   contenido de `candidate/` en el mismo cambio de código. No hay comando de
   aceptación automática. Ejecutar de nuevo el check y la suite offline completa;
   una propuesta desactualizada no coincide con el código y falla.

La sustitución es una edición de archivos sujeta a revisión del repositorio; el
programa no verifica la identidad de un revisor ni demuestra aprobación humana.
El registro inicial de este punto es una referencia de sandbox, no una aprobación
de despliegue. La automatización de revisiones y el bloqueo de promoción a dev
corresponden al punto 7 y siguen pendientes.

## Alcance y límites

Este mecanismo detecta cambios accidentales. No demuestra equivalencia semántica
ni garantiza que el modelo siempre obedecerá una instrucción presente. Las
capturas efectivas cubren las rutas sintéticas ejecutadas; otras ramas quedan
protegidas por el diff conservador de sus archivos fuente. Por eso un cambio de
código en uno de esos archivos puede exigir revisión aunque no cambie el texto
de un prompt. Las referencias de conversación son candidatas a ejecutar, no una
afirmación de que la prueba ejercite cada familia en todos sus niveles.

La búsqueda de nuevos consumidores reconoce imports estáticos del cliente y
constructores en `app/prompts`; no pretende impedir un bypass deliberado mediante
Python dinámico, otro proveedor o la modificación simultánea del verificador y
su referencia. El control de revisión y CI debe proteger esos cambios.

El fingerprint de los reportes de conversación ahora incluye JSON del agente
(incluidas las plantillas) y el contrato de prompts. Los fingerprints anteriores
al punto 4 se calcularon sin esos elementos y conservan su valor histórico.

## Validación de esta implementación

- [Check del contrato](../tests/conversation_regression/reports/point4-guard.json):
  PASS; 12 familias, 14 anclas y 201 documentos de referencia.
- [Suite offline](../tests/conversation_regression/reports/point4-unit.log):
  376 pruebas, 366 aprobadas y 10 opt-in omitidas. Incluye 16 pruebas nuevas del
  verificador: eliminación de reglas, pérdida de contexto, cambios de esquema,
  plantillas, dependencias compartidas, nuevas fuentes, integridad de snapshots,
  bloqueo de red y propuestas que no aceptan ni sobrescriben la referencia.
- [Replay](../tests/conversation_regression/reports/point4-offline.json):
  20 OFFLINE_PARTIAL_PASS, 21 NOT_RUN y ningún fallo. Los pendientes permanecen
  visibles y no se cuentan como cobertura completa.
- [Demostración simulada](../tests/conversation_regression/reports/point4-removal-demonstration.json):
  quitar en memoria la instrucción de eliminación de un artículo produce BLOCKED,
  mostrando reglas y pruebas afectadas; los archivos de producción no se cambian.

Cero llamadas a OpenAI. No se repitió Spring/H2 porque este punto modifica las
herramientas locales de revisión y los fingerprints, sin cambiar el comportamiento
del agente o del backend. La integración de las correcciones 1–5 conserva su
evidencia histórica; no se presenta como una ejecución nueva del punto 4.

## Punto 7: check obligatorio preparado en sandbox

El [bloqueo de CI](conversation-ci-gate.md) conecta este contrato con las suites
del agente, backend y conversaciones Spring/H2. Incluye un registro de revisión
vinculado a la referencia exacta y protege también el código y la política del
verificador. Los workflows están preparados localmente; exigirlos en GitHub y
activar la espera de checks en Render sigue pendiente de configuración remota.
