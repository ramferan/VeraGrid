# Protocolo de Inicio
1. **LEE EL CONTEXTO GLOBAL**: Antes de nada, lee `../casandra/README.md` y `../casandra/PLANNING_CONTEXT.md` para entender el ecosistema completo.
2. **LEE LAS DECISIONES TÉCNICAS**: Lee `../casandra/TECHNICAL_DECISIONS.md` para conocer las decisiones arquitectónicas clave.
3. **LEE SIEMPRE `TASKS.md`**: Lee este fichero para entender el estado actual y tu objetivo inmediato.
4. **LEE EL CONTEXTO LOCAL**: Lee `VISION.md` y `PROJECT_OVERVIEW.md` de este repositorio.
5. **MANTÉN LA DOCUMENTACIÓN VIVA**: Si tu código cambia la realidad descrita en estos documentos, ACTUALÍZALOS.
6. **IDIOMA (HÍBRIDO)**: Todo el código debe estar en **Inglés**: variables, funciones, clases, docstrings y comentarios. Los commits también en inglés. Los archivos **Markdown (`.md`)** deben estar en **Español**.
7. **PROTOCOL GESTIÓN GIT**: NO realizar PUSH directo a `devel` o `main`. Crear siempre una rama `feat/agent-...` y esperar validación.
8. **PLAN-BEFORE-CODE**: Presentar SIEMPRE un plan detallado de pasos y esperar la aprobación del USER antes de modificar archivos o ejecutar comandos.
9. **ACTUALIZA `TASKS.md`**: Al terminar tu tarea, actualiza el estado.

# Instrucciones para Agentes de IA
**Contexto Global**: Este repositorio (`VeraGrid`) es una herramienta de simulación **LEGACY**. Se mantiene por motivos históricos y de comparación mientras se migra a `hybridcircuit` (Axon).

Eres un desarrollador experto en Python y Cálculo Numérico.

## 1. Prioridades
- **Rendimiento**: Optimización crítica (Numpy/Numba).
- **Mantenibilidad**: No añadir features nuevas de UI. Refactorizar para extraer lógica útil.

## 2. Gestión del Código y Git
- **Workflows**: Sigue estrictamente los flujos definidos en `.agent/workflows/`.
- **Commits**: Usa Conventional Commits.
- **Ramas**: Mantén el trabajo en `devel` o ramas de feature específicas.

## 3. Seguridad
- Evita incluir datos reales de la red eléctrica en el repositorio.
