# Nota Técnica: Ajustes Metodológicos y Formulación del Módulo NTC en VeraGrid 6

**Para:** Equipo de Desarrollo de eRoots / VeraGrid  
**De:** Equipo de Red Eléctrica de España (REE)  
**Fecha:** 28 de octubre de 2026  
**Asunto:** Propuesta de alineación metodológica para el módulo NTC/TTC en VeraGrid 6.x

---

## 1. Motivación y Alcance

Desde REE valoramos muy positivamente las mejoras introducidas en las versiones recientes de VeraGrid, en particular la formulación lazy de contingencias, el recorte de flujo (`add_clipped_flow`) y los avances en la modelización de enlaces en corriente continua (HVDC).

Con el objetivo de converger en una implementación oficial de NTC en VeraGrid 6 que satisfaga los requisitos de cálculo de capacidad comercial de REE, hemos realizado una auditoría comparativa entre la formulación de referencia del MOU (GridCal 4.8) y las formulaciones `regular` y `strict` de VeraGrid 6.7.20.

Aunque por confidencialidad no podemos compartir directamente los ficheros de red del modelo peninsular, compartimos en esta nota la tabla comparativa de resultados obtenidos en el escenario de referencia, trasladamos los ajustes metodológicos requeridos con su fundamento operativo/regulatorio, y aportamos los resultados correspondientes sobre una red canónica pública de vuestro propio repositorio (IEEE 14).

---

## 2. Comparativa de Resultados en el Escenario de Referencia (MOU Hora 13 EF Pmode3)

A continuación resumimos los resultados observados al ejecutar las diferentes formulaciones sobre el mismo punto de operación de la interconexión España–Francia:

| Formulación / Motor | TTC (MW) | NTC (MW) | Slack Térmico | Slack Área | Tiempo | Observación de REE |
|---|---:|---:|---:|---:|---:|---|
| **4.8 MOU Original (Referencia)** | **5.297,10** | **4.897,10** | **0 MW** | **0 MW** | 104 s | Referencia oficial de REE (preventivo, N-1 desacoplado). |
| **4.8 Mejoras (Clipped + Lazy)** | **5.297,10** | **4.897,10** | **0 MW** | **0 MW** | 111 s | Misma física, admisión lazy en 1 ronda. |
| **eRoots 6.7.20 `regular`** | 8.002,26 | 7.602,26 | 6.080,77 MW | Activo* | 172 s | Capacidad inflada por absorción de sobrecargas en slacks térmicos y de área. (*) Variable presente en LP pero no volcada al resultado por VeraGrid. |
| **eRoots 6.7.20 `strict` (oficial)** | — | — | — | — | 38 s | LP base infactible (signo invertido en PST y objetivo degenerado). |
| **VeraGrid 6 Limpio (Preventivo Estricto - Estándar MOU)** | **5.171,18** | **4.771,18** | **0 MW** | **0 MW** | **~23 s** | **Estándar MOU:** PST $+\tau$, balancín puro 1:1, preventivo estricto con lazy screening y multi-gen resuelto. Coincidencia del 97,7% con MOU 4.8. |
| **VeraGrid 6 Limpio (Correctivo Acotado - SRAP / PST)** | **5.399,58** | **4.999,58** | **0 MW** | **0 MW** | **~24 s** | **Automatismos operativos:** Alivio de contingencia crítica mediante PST rápido y SRAP $\le 1.300$ MW (límite operativo P.O. 3.1). |
| **VeraGrid 6 Limpio (Correctivo Teórico - Modulación HVDC)** | **7.439,57** | **7.039,57** | **0 MW** | **0 MW** | **~25 s** | **Modulación libre EPC:** Reasignación post-contingencia desacoplada de los 8.000 MW de convertidores HVDC (techo asintótico sin límites de rampa). |

> **Nota aclaratoria sobre los escalones de capacidad en VeraGrid Limpio:**  
> - En **preventivo estricto puro** (`corrective_contingencies=False`), las consignas de los convertidores se fijan en base y el TTC alcanza **5.171,18 MW** con **Slack = 0,0 MW**. Este es el **estándar contractual canónico del MOU**. La pequeña diferencia de ~126 MW frente a los 5.297 MW del MOU 4.8 (apenas un ~2,3%) proviene del acoplamiento angular pasivo en bornes de convertidores ($\theta_{fr} - \theta_{to}$).  
> - En **correctivo con automatismos acotados** (5.399,58 MW), modelando acciones remediales según los límites reales de operación (disparo de generación SRAP acotado a 1.300 MW y tomas rápidas de PST), se alivia la contingencia de Hernani–Argia hasta que la red alcanza su cota firme en la línea de 400 kV Peñaflor–Ejea.  
> - En **correctivo con modulación libre HVDC** (`corrective_contingencies=True`), correspondiente a permitir al solver redespachar libremente los 8.000 MW de convertidores de continua sin límites de rampa (*Emergency Power Control* profundo), la capacidad se expande hasta **7.439,57 MW** con estricto cumplimiento térmico (0 MW slack). Representa el techo técnico asintótico de los enlaces de continua, no el régimen ordinario de operación.  
> - Se eliminó además la inyección errónea de restricciones sobre ramas colaterales no vigiladas por el criterio del 5% (`has_relief`), garantizando que las restricciones operen únicamente sobre las ramas determinantes sin sobre-constreñir artificialmente la red.

---

## 3. Ajustes Metodológicos y Fundamento de los Requisitos de REE

### 3.1. Balance Estructural del Intercambio (Eliminación de Slacks de Área)

* **Fundamento del requisito de REE:**  
  La capacidad comercial de intercambio (NTC/TTC) entre zonas de oferta (*bidding zones*) es un producto comercial bilateral regulado por la normativa europea (Reglamentos CACM y FCA). En el acoplamiento de mercados, **la energía exportada por el Área 1 debe ser idéntica a la importada por el Área 2**. La introducción de variables de holgura en el balance de área ($\Delta_{sl_1}, \Delta_{sl_2}$) equivale a inyectar o absorber energía de forma ficticia en el sistema desde una fuente no física, falseando el poliedro de intercambio y arrojando capacidades comerciales no viables para el mercado.
* **Situación observada en el código:**  
  En las formulaciones actuales, el balance de áreas se define mediante variables de holgura vinculadas por $\Delta_1 - \Delta_{sl_1} = \Delta_2 - \Delta_{sl_2}$. En la formulación `regular`, el objetivo maximiza $-\Delta_1$, permitiendo que el intercambio se financie artificialmente con holgura de área. En la formulación `strict`, la combinación algebraica del objetivo y la restricción cancela $\Delta_1$, reduciendo el objetivo a $2 \cdot \Delta_{sl_2}$.
* **Ajuste propuesto:**  
  El balance entre áreas $A_1$ y $A_2$ debe ser un **balancín físico perfecto (1:1)**:
  $$\Delta P_{A_1} = -\Delta P_{A_2} = \Delta P$$
  $$\text{Maximizar } \Delta P$$
  Sin variables slack en las áreas. El escalado de generación o demanda debe conservar la suma neta estricta en cada iteración del intercambio.

---

### 3.2. Límites Térmicos Estrictos en Base y Post-Contingencia

* **Fundamento del requisito de REE:**  
  Conforme a los criterios de seguridad de la red de transporte (Reglamento europeo SO GL y Procedimientos de Operación de REE P.O. 3.1), la capacidad comercial determinada debe garantizar que **ningún elemento de la red supere su límite térmico admisible en régimen permanente ni tras contingencia N-1**. Admitir sobrecargas mediante penalizaciones en la función objetivo (como ocurre en la versión `regular`, que arroja estados `Relaxed` con más de 6.000 MW de violación térmica acumulada) invalida el resultado para operación en tiempo real y planificación.
* **Evolución positiva en `strict`:**  
  Reconocemos y valoramos positivamente que en la formulación `strict` eliminasteis las variables de holgura térmica en las ramas (`add_preventive_contingency_formulation_strict`), estableciendo límites térmicos duros ($|f_l^{(c)}| \le F_{l,\max}^{(c)}$). Este es exactamente el camino metodológico que requiere REE.
* **Ajuste propuesto:**  
  Consolidar los límites térmicos estrictos como estándar para cálculo de NTC, de modo que si ante un escalón de intercambio la red se satura, el solver declare el límite físico exacto sin relajar la seguridad del sistema.

---

### 3.3. Corrección del Signo en Transformadores Desfasadores (PST)

* **Fundamento del requisito de REE:**  
  Fidelidad física del flujo de cargas en continua (DC-OPF). Un transformador desfasador es un equipo de control de flujo activo en las interconexiones pirenaicas (p. ej. Arkale, Biescas–Pragnères). Un adelanto angular en el nudo *from* ($\tau > 0$) incrementa la diferencia angular efectiva $(\theta_{from} + \tau - \theta_{to})$, induciendo un flujo de potencia positiva hacia el nudo *to*. Invertir el signo matemático fuerza el flujo en sentido opuesto al físico real, sobrecargando artificialmente las interconexiones con PST e induciendo infactibilidades numéricas espurias en el optimizador.
* **Situación observada en el código:**  
  En `ntc_opf_strict.py:908` y `ntc_opf.py:1363` el flujo por transformadores desfasadores se formula como:
  $$f_m = b_k (\theta_{from} - \theta_{to} - \tau_m)$$
  Mientras que el propio compilador de admitancias de VeraGrid (`circuit_to_data.py:1659`) suma el ángulo de desfase al nudo origen ($V_{bus} = V_m e^{j(\theta + \tau)}$).
* **Ajuste propuesto:**  
  Restablecer la ecuación física canónica:
  $$f_m = b_k (\theta_{from} - \theta_{to} + \tau_m)$$

---

### 3.4. Respuesta Pasiva de Convertidores HVDC en Pmode3 vs Acciones Correctivas

* **Fundamento del requisito de REE:**  
  Cumplimiento del pliego de condiciones técnicas acordado entre los TSOs (REE y RTE) en el MOU. El modo Pmode3 modela una respuesta pasiva automática mediante estatismo frecuencia/ángulo (*droop*), simulando la física de control de los enlaces VSC-HVDC ante disparos en la red AC. Introducir variables libres de redespacho por contingencia otorga al solver la potestad de utilizar los enlaces DC como *remedial actions* per-contingencia no declaradas, llegando a invertir flujos entre enlaces para sortear cuellos de botella AC. El cálculo preventivo de NTC no debe depender de redespachos correctivos arbitrarios salvo que se especifique un módulo explícito de acciones remediales.
* **Situación observada en el código:**  
  En `ntc_opf.py:1559` y `ntc_opf_strict.py:1123`, se introdujo una variable libre $\Delta_d^{(c)}$ para cada convertidor $d$ y contingencia $c$, sujeta a $\sum_{d} \Delta_d^{(c)} = 0$.
* **Ajuste propuesto:**  
  Fijar la respuesta post-contingencia de los enlaces HVDC a su ley de estatismo pasiva de Pmode3, preservando el cálculo preventivo riguroso acordado con RTE. Si en el futuro se desea contemplar acciones correctivas post-contingencia (Remedial Actions), estas deben definirse en un módulo explícito e independiente con especificaciones y parametrización operativa, no integradas como variables libres dentro del bucle pasivo de contingencias N-1.

---

### 3.5. Formulación Preventiva: Incorporación del Motor Lazy a la Formulación Estricta

* **Reconocimiento del desarrollo Lazy:**  
  Queremos reconocer expresamente que el motor de **admisión Lazy** (`screen_contingency_violations`), desarrollado por vosotros en `ntc_opf.py`, es una mejora metodológica excelente. Realizar el screening de contingencias en NumPy fuera del LP e inyectar al solver únicamente las filas que realmente violan límites (~128 grupos de contingencia de 7.670) es la mejor práctica de la industria para resolver problemas de gran escala.
* **Situación observada en `strict`:**  
  Observamos que esta mejora no se trasladó a la formulación estricta (`ntc_opf_strict.py`). En `strict`, cuando se ejecuta en modo preventivo, el código inyecta en bloque las restricciones de todas las ramas para todas las contingencias en el LP:
  $$7.670 \text{ contingencias} \times 4.000 \text{ ramas} \times 2 \text{ límites} \approx 61.000.000 \text{ de restricciones}$$
  Esto satura la memoria del proceso y explica por qué la ejecución de `strict` preventivo resultaba inviable en tiempo.
* **Ajuste propuesto:**  
  Reutilizar vuestro propio motor de screening y admisión *Lazy* de `ntc_opf.py` dentro de la formulación `strict` con límites duros, aunando la rapidez del Lazy (~1-2 minutos) con el rigor de la NTC comercial sin variables de holgura.

---

## 4. Volcado de la Solución al Snapshot de Red (`get_ntc_operating_point_as_opf_results`)

* **Fundamento del requisito de REE:**  
  Trazabilidad operativa y verificación física del punto de trabajo. Los operadores de red necesitan verificar el resultado de la optimización mediante simulaciones de flujo de cargas no lineal (AC Power Flow) en la GUI de VeraGrid o exportar el caso en formatos estándar (.gridcal / CGMES). Para ello, el despacho resultante de la NTC debe volcarse de forma exacta y coherente en los elementos del modelo.
* **Aspectos a corregir:**  
  1. **Eliminación del redespacho global de desbalance (`ntc_driver.py:163`):**  
     Actualmente, el código ejecuta:
     ```python
     imbalance = float(gen_injection.sum() + battery_injection.sum() + load_injection.sum())
     balanced_gen -= gen_injection / gen_total * imbalance
     ```
     En cualquier red de transporte real existen pérdidas de transporte AC ($P_{loss}$), por lo que la suma de generación nunca iguala exactamente a la carga. Esta corrección previa altera arbitrariamente el despacho económico base de centrales fuera de las zonas del intercambio (España y Francia). El escalado de NTC debe aplicarse **exclusivamente sobre las áreas participantes**, preservando el punto de operación del resto del sistema.
  2. **Asignación a los objetos físicos del circuito:**  
     Para que el usuario pueda validar el modelo ejecutando un flujo de cargas desde la GUI o mediante script, se requiere un método nativo (`apply_ntc_to_circuit`) que asigne directamente los valores óptimos resueltos a los dispositivos (`generator.p`, `hvdc.Pset`, `transformer.tap_phase`).

---

## 5. Propuesta de Optimización: Topologías Invariantes en Series Temporales

* **Fundamento del requisito de REE:**  
  Eficiencia de cálculo en estudios plurianuales y de planificación. En estudios de series temporales (8.760 horas), el 99% de las horas comparten exactamente la misma topología de interruptores y líneas en servicio. Compilar el `numerical_circuit`, reconstruir la matriz de admitancias $Y_{bus}$ y refactorizar $B'$ 8.760 veces es un desperdicio masivo de CPU.
* **Garantía metodológica:**  
  En formulación DC lineal, las inyecciones de generación, demanda, almacenamiento y baterías son estrictamente aditivas en cada nudo. Compilar el `numerical_circuit` y pre-factorizar $B'$ **una única vez por topología estructural única**, e inyectar el vector horario $P_{iny}(t) = P_{gen}(t) - P_{load}(t)$ en cada paso, preserva el $100\%$ del rigor físico y reduce el tiempo total de cómputo en órdenes de magnitud.

---

## 6. Validación Cuantitativa sobre Red Canónica Reproducible (IEEE 14 del Issue 372)

Para comprobar estos comportamientos de forma transparente en un entorno abierto que podáis reproducir directamente sin necesidad de datos confidenciales de REE, hemos ejecutado las formulaciones sobre la red de prueba canónica incluida en vuestro repositorio:  
`IEEE14 - ntc areas_voltages_hvdc_shifter_l10free.gridcal` (desarrollada para el Issue 372).

### 6.1. Caso Estándar IEEE 14 (Descongestionado)

En la red original, los límites térmicos son holgados respecto al despacho base del modelo:

| Formulación / Motor | TTC Base (MW) | TTC N-1 (MW) | Slack Térmico | Slack Área | Tiempo | Diagnóstico / Observación |
|---|---:|---:|---:|---:|---:|---|
| **Referencia Preventiva (Equivalente 4.8)** | 150,26 MW | **96,26 MW** | **0,00 MW** | **0,00 MW** | ~2 s | Referencia limpia sin slacks; flujos coherentes con flujo AC base. |
| **eRoots 6.7.20 `regular` (con Lazy)** | 150,26 MW | 96,26 MW | 0,00 MW | 0,00 MW* | ~2 s | Resuelve rápido con Lazy screening. (*) Variable presente en LP pero inactiva por simetría. |
| **eRoots 6.7.20 `strict` oficial (signo $-\tau$)** | 150,26 MW | **78,07 MW** | 0,00 MW | 0,00 MW* | ~3 s | **Capacidad reducida en 18,19 MW** al saturar la Rama 7 al 100% (20 MW) por signo erróneo. |
| **VeraGrid 6 Limpio (Lazy + signo $+\tau$)** | 150,26 MW | **96,26 MW** | **0,00 MW** | **0,00 MW** | ~2 s | Balancín 1:1 estricto, signo $+\tau$ corregido y flujo físico natural. |

#### Diagnóstico de flujos de enlace en el caso IEEE 14:
1. **Flujo AC base de partida:**  
   En el flujo de cargas AC, el desfasador de la rama 8 (`branch 8`, $\tau = +0.0593$ rad / $3,4^\circ$, límite 50 MW) transporta un flujo base positivo de **$+10,65$ MW** hacia el nudo 9.
2. **Impacto del signo del desfasador bajo N-1:**
   * **Con el signo actual de VeraGrid ($-\tau$):**  
     El flujo por la rama 8 desciende erróneamente a **$+0,68$ MW** (un freno de $-3,45$ MW opuesto al adelanto angular físico). Simultáneamente, la rama 7 (`branch 7`, con $\tau = -0.05$ rad y límite 20 MW) invierte su signo a $-(-0.05) = +0.05$ rad, forzando potencia espuria hasta **saturar al 100% su capacidad (20,00 MW)**. Esta saturación artificial estrangula el TTC bajo N-1 de 96,26 MW a 78,07 MW (-18,19 MW de capacidad perdida ficticiamente).
   * **Con el signo corregido ($+\tau$):**  
     El flujo por la rama 8 pasa a **$+4,13$ MW**, coherente con el empuje angular positivo, la rama 7 no se satura artificialmente y la capacidad N-1 alcanza de forma natural los **96,26 MW**.

---

### 6.2. Caso IEEE 14 Congestionado (Análisis de Holguras Térmicas y Garantía de Optimalidad Física)

En el caso estándar anterior (6.1), al no existir cuellos de botella severos, los términos de holgura de la formulación `regular` permanecen en 0,00 MW, lo que podría sugerir preliminarmente que la presencia de variables de penalización en el modelo matemático es inocua.

Para analizar de forma rigurosa y reproducible el comportamiento del optimizador ante restricciones vinculantes, se evalúa el caso implementado en `test_issue_372_congested_thermal_slacks` (`src/tests/NTC/test_ntc_strict_formulation.py`). Sobre la misma red abierta del Issue 372, se reduce la capacidad nominal de la rama 7 (Trafo 11 entre nudos 7 y 15, correspondiente al índice de rama 21 en la indexación interna) de 20 MW a **4,00 MW** (con límite de contingencia N-1 de 4,60 MW), generando un cuello de botella crítico bajo la indisponibilidad de la rama 14:

| Formulación / Modo | TTC Base (MW) | TTC N-1 (MW) | Slack Térmico ($s$) | Slack Área ($\Delta_{sl}$) | Estado Solver | Diagnóstico Físico / Operativo |
|---|---:|---:|---:|---:|:---:|---|
| **Preventivo Estricto (Límites duros)** | **56,31 MW** | **0,00 MW** | **0,00 MW** | **0,00 MW** | `Infeasible` | **Seguridad física garantizada:** Detecta que la red no tolera la contingencia (flujo inducido de 22,93 MW vs. 4,60 MW de límite). Bloquea la publicación de capacidad inviable. |
| **Preventivo `regular` (Con slacks)** | **56,31 MW** | **34,49 MW** | **18,33 MW** | 0,00 MW | `Optimal` | **Resolubilidad aparente:** El solver publica 34,49 MW absorbiendo una sobrecarga del ~500% en el slack térmico. En explotación real causaría la actuación inmediata de protecciones. |
| **Correctivo `regular` (Con redespacho VSC)** | **56,31 MW** | **10,49 MW** | **12,00 MW** | 0,00 MW | `Optimal` | **Sobrecarga en contingencia:** Con redespacho de convertidores, el flujo inducido es de 16,60 MW; el solver asigna 10,49 MW de TTC absorbiendo 12,00 MW en $pos\_sl$. |

#### Análisis Técnico sobre la Garantía de Optimalidad Física:

Un argumento habitual en favor de la inclusión de variables de holgura penalizadas en la formulación lineal es considerar que *«dichas variables únicamente se activan cuando no existe solución matemática factible, comportándose como un mecanismo de diagnóstico equivalente a una infactibilidad, sin alterar el óptimo cuando este existe»*.

El análisis matemático y los resultados empíricos anteriores permiten precisar por qué **la introducción de variables de holgura en las restricciones térmicas y de área no garantiza la obtención del óptimo físico de operación**, resultando desaconsejable en cálculos de capacidad comercial para el mercado:

1. **Alteración del espacio de búsqueda factible ($\mathcal{F}_{\text{físico}}$ vs. $\mathcal{F}_{\text{relajado}}$):**  
   Al sustituir una restricción física dura de capacidad $f_{ij}^{(c)} \le F_{ij}^{\max}$ por una relajación con holgura $f_{ij}^{(c)} - s_{ij}^{(c)} \le F_{ij}^{\max}$ ($s_{ij}^{(c)} \ge 0$), el poliedro admisible de la red se amplía artificialmente. Cuando el solver encuentra una solución con $s^* > 0$, dicho punto **no pertenece al dominio admisible del sistema eléctrico**. La solución reportada no es un óptimo físico de la red, sino un óptimo matemático de un problema relajado que tolera violaciones térmicas inaceptables en explotación real.

2. **Dependencia del hiperparámetro de penalización ($M$):**  
   La función objetivo $\max \Delta P - M \sum s$ transforma la optimización de capacidad en un compromiso comercial entre el beneficio del intercambio ($\Delta P$) y el coste numérico asignado a la penalización ($M = 10^4$ en VeraGrid). Esta formulación arbitra de manera implícita cuánta sobrecarga física se tolera por cada megavatio adicional transferido. Por consiguiente, el valor final de TTC no refleja la capacidad física de la infraestructura, sino la calibración arbitraria del coeficiente de penalización.

3. **Enmascaramiento de la infactibilidad física (Falsa convergencia):**  
   En la gestión de capacidad para el mercado eléctrico (conforme a los reglamentos UE 2015/1222 - CACM y UE 2017/1485 - SO GL), la capacidad comercial publicada debe ser plenamente ejecutable con criterios de seguridad N-1. Si una situación de red no admite intercambio seguro, el cálculo debe retornar estrictamente una señal de infactibilidad (`Infeasible / TTC = 0`). La formulación con holguras, al retornar el estado `Optimal / Converged`, oculta al operador que el intercambio requiere que elementos de la red operen al 400% o 500% de su límite térmico, trasladando un riesgo sistémico severo a la operación en tiempo real.

#### Verificación Directa con Pytest:
El comportamiento comparado entre límites duros y variables de holgura puede reproducirse de forma directa en el entorno de pruebas del repositorio:
```bash
pytest src/tests/NTC/test_ntc_strict_formulation.py -k "test_issue_372_congested_thermal_slacks"
```

**Conclusión técnica:**  
La garantía de optimalidad y de seguridad física del sistema exige que las restricciones térmicas de red ($f \le F^{\max}$) y el balancín de intercambio de área ($1:1$) se formulen como restricciones duras sin variables de relajación. Ante contingencias no superables, la infactibilidad debe emerger de forma explícita para que el operador active medidas remediales reales (redespacho preventivo, topología o modulación acotada) en lugar de tolerar sobrecargas virtuales en el modelo.


---

## 7. Próximos Pasos y Colaboración

Desde el equipo de REE ponemos a vuestra disposición el código con estos ajustes consolidados y los scripts de prueba para su revisión conjunta y posterior integración en el núcleo oficial de VeraGrid.
