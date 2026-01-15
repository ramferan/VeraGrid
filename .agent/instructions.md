# Instrucciones para Agentes de IA
**Contexto Global**: Este repositorio es el MOTOR ELÉCTRICO del ecosistema **Casandra**. Para entender la visión de negocio y el proceso de planificación completo, consulta obligatoriamente los documentos en el repositorio maestro: `casandra/.agent/`.

Eres un desarrollador experto trabajando en HybridCircuit. Sigue estrictamente estas normas:

## 1. Gestión del Código
- **Atomicidad**: Realiza cambios pequeños y verificables.
- **Ramas**: Si vas a implementar una nueva funcionalidad completa, sugiere crear una rama `feat/nombre` o `fix/nombre`.
- **Commits**: Al generar mensajes de commit o descripciones de Pull Request, usa Conventional Commits (`feat:`, `fix:`, `docs:`).

## 2. Gestión de Datos y Rutas
- **IMPORTANTE**: Jamás hardcodees rutas absolutas a tu máquina local (ej: `C:\Users\...`).
- Usa siempre `hybridcircuit.config.get_data_path()` para acceder a archivos de datos.
- Asume que los datos confidenciales no están en el repositorio, sino en carpetas externas configuradas vía `.env`.

## 3. Estilo de Python
- Usa Type Hints (`def funcion(a: int) -> str:`).
- Documenta las funciones complejas con docstrings.
- Mantén la estructura modular del proyecto.

## 4. Seguridad
- Revisa siempre no incluir credenciales, IPs internas o nombres de archivos sensibles en el código commiteado.
