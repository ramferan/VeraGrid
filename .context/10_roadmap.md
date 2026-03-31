# 10. Roadmap Técnico y Tareas: VeraGrid (Vendor Support)

VeraGrid se mantiene en el ecosistema como un componente de **soporte para funcionalidades heredadas** y motores de cálculo específicos ya desarrollados. No se prevé una evolución del núcleo del motor, sino su integración mediante interfaces modernas.

## 🚨 Fase 0: Operatividad con AXON (Prioridad Máxima)
- [ ] **Integración de Motores**: Asegurar que los motores de cálculo específicos requeridos por REE son accesibles y ejecutables desde AXON sin errores de entorno.
- [ ] **Validación de Salidas**: Garantizar que los formatos de salida actuales son correctamente interpretados por la capa de guardado (Pinakes).

## 🚀 Fase 1: Interoperabilidad (Fase de Evolución)
- [ ] **Resultados en Parquet**: Implementar la persistencia en formato Parquet solo si mejora el Hito 0 o como parte de la migración final.
- [ ] **Exposición vía MCP**: Crear wrappers MCP para facilitar el uso de los algoritmos de VeraGrid por agentes de IA.

## ⚙️ Fase 2: Estabilidad (Continuo)
- [ ] **Compatibilidad**: Asegurar que las actualizaciones del entorno Python de Casandra no rompan las dependencias específicas del vendor.

---

## 🛡️ Reglas Específicas del Repo
- **Legacy Protection**: No modificar el núcleo del vendor a menos que sea estrictamente necesario para corregir bugs críticos.
- **Exposure First**: Priorizar la creación de herramientas MCP sobre cualquier otra mejora funcional.

---

## 🔝 Estrategia de Desarrollo (MANDATORIO)
1. **Branch Management**: Toda nueva funcionalidad debe desarrollarse en una **rama dedicada** (`feat/nombre-funcionalidad`). Nunca trabajar directamente en `devel` o `main`.
2. **Testing First**: Cada feature debe contar con tests que garanticen su funcionalidad antes de ser integrada.
3. **Full Registry Check**: Antes de dar por finalizada una tarea, se deben ejecutar y pasar el **100% de los tests** del repositorio.
4. **Context Sync**: Mantener actualizados los archivos de `.context/` tras cambios significativos.
