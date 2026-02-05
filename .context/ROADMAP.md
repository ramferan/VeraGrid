# Roadmap Particularizado: VeraGrid (Vendor Support)

VeraGrid se mantiene en el ecosistema como un componente de **soporte para funcionalidades heredadas** y motores de cálculo específicos ya desarrollados. No se prevé una evolución del núcleo del motor, sino su integración mediante interfaces modernas.

## 📅 Hitos de Mantenimiento e Integración

### 0. Operatividad con AXON (Prioridad Máxima 🚨)
*   **Integración de Motores:** Asegurar que los motores de cálculo específicos requeridos por REE son accesibles y ejecutables desde AXON sin errores de entorno.
*   **Validación de Salidas:** Garantizar que los formatos de salida actuales son correctamente interpretados por la capa de guardado (Pinakes).

### 1. Interoperabilidad (Fase de Evolución)
*   **Resultados en Parquet:** Implementar la persistencia en formato Parquet solo si mejora el Hito 0 o como parte de la migración final.
*   **Exposición vía MCP:** Crear wrappers MCP para facilitar el uso de los algoritmos de VeraGrid por agentes de IA.

### 2. Estabilidad (Continuo)
*   **Compatibilidad:** Asegurar que las actualizaciones del entorno Python de Casandra no rompan las dependencias específicas del vendor.

---

## 🛡️ Reglas Específicas del Repo
- **Legacy Protection:** No modificar el núcleo del vendor a menos que sea estrictamente necesario para corregir bugs críticos.
- **Exposure First:** Priorizar la creación de herramientas MCP sobre cualquier otra mejora funcional.

