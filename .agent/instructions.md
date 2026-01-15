# Protocolo de Inicio
1. **LEE SIEMPRE `TASKS.md`**: Antes de hacer nada, lee este fichero para entender el estado actual y tu objetivo inmediato.
2. **LEE LA ARQUITECTURA**: La visión global está en `../soma/ARCHITECTURE.md`.
3. **LEE EL CONTEXTO LOCAL**: Lee `VISION.md` y `PROJECT_OVERVIEW.md` de este repositorio.
4. **MANTÉN LA DOCUMENTACIÓN VIVA**: Si tu código cambia la realidad descrita en estos documentos, ACTUALÍZALOS.
5. **ACTUALIZA `TASKS.md`**: Al terminar tu tarea, actualiza el estado.

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
