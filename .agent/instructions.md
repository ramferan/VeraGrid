# Protocolo de Onboarding: VeraGrid (Legacy & Performance)

Bienvenido a **VeraGrid**. Este es un motor de cálculo crítico con herencia legacy. Tu misión es mantener su estabilidad mientras extraemos su lógica para el futuro del ecosistema.

## 🧭 Mapa del Cerebro (Contexto)
**OBLIGATORIO**: Sincroniza el contexto al inicio:
1. **Knowledge Scan**: Lee `.context/README.md`.
2. **Contexto Hub**: Lee `../casandra/README.md` y `../casandra/TECHNICAL_DECISIONS.md`.
3. **Estado**: Consulta `TASKS.md`.

## 🔄 Flujo Git Swarm
- **Ramas**: `main`, `devel`. Trabajo en `feat/agent-...` o `fix/agent-...`.
- **Commits**: Conventional Commits. Mensajes atómicos.
- **Push**: **PROHIBIDO** sin autorización explícita del USER.

## 🧪 Calidad y Testeo
- Este proyecto ya utiliza `pytest`. 
- Revisa `pytest.ini` y añade tests en la carpeta correspondiente (`tests/` o subcarpetas de `src/`).
- Crea o actualiza `tests/conftest.py` para fixtures globales.

## 📜 Idioma y Estilo
- **Código y Commits**: Inglés.
- **Documentación MD**: Español.
- **Rendimiento**: Prioriza Numba/Numpy en bucles calientes.
- **UI**: No añadir features nuevas de UI. Refactorizar para extraer lógica útil.

---
*Justifica siempre el "Por qué" de tus cambios en tus planes de acción.*
