# 01. Gestión de Datos: Computación Científica Masiva (VERAGRID)

## 🎯 Alcance Local (Engine)

### 1. Ingesta Zero-Copy (Arrow)
- VeraGrid debe consumir *escenarios de red* directamente desde **Arrow Buffers** recibidos del Orquestador, evitando parseos de texto adicionales.
- Uso de `pyarrow` para acceder a arrays en un `SharedMemory` buffer durante el cómputo distribuido.

### 2. Salida: Parquet
- Las miles de simulaciones N-k se escriben concurrentemente a **Parquet**.
- Escritura distribuida con metadatos de partición que permitan consultas analíticas rápidas (ej: "Dime qué contingencias superaron el 110% de carga").

## 🛤️ Pasos de Implementación
1. [ ] Implementar un `ArrowScenarioReader` para la carga rápida de contingencias.
2. [ ] Configurar salida compatible con **Amazon S3 Tables / Glue**.
