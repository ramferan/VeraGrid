# Estado del Proyecto: Mantenimiento / Evaluación Estratégica 📉

## 🎯 Objetivo Principal
Herramienta heredada de simulación eléctrica. El objetivo actual es evaluar su futuro frente a `hybridcircuit` (Axon) y resolver problemas de rendimiento crítico y usabilidad de la GUI.

## 📋 Próximos Pasos (To-Do)
- [ ] **Estrategia**: Definir si se mantiene el fork o se migra la lógica útil a `hybridcircuit` y se archiva este repo.
- [ ] **Rendimiento**:
    - [ ] Optimizar algoritmos de análisis lineal (DC, MLODF, contingencias N-1).
    - [ ] Investigar vectorización (Numpy) y JIT (Numba) para alcanzar rendimiento tipo C.
- [ ] **GUI**:
    - [ ] La interfaz actual es pesada y difícil de mantener. No priorizar nuevas features de UI aquí.

## ℹ️ Contexto
- **Problemas**: Rendimiento muy pobre en cálculos masivos; GUI obsoleta.
- **Relación**: `axon` (HybridCircuit) es el potencial sucesor para el motor de cálculo.
