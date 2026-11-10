# Nota Técnica: Correcciones y Ajustes en el Módulo NTC de VeraGrid 6

**Para:** Equipo de Desarrollo de eRoots / VeraGrid  
**De:** Equipo de Red Eléctrica de España (REE)  
**Fecha:** 5 de noviembre de 2026  
**Asunto:** Correcciones necesarias en la formulación y parser del módulo NTC en la rama oficial de VeraGrid 6.x

---

## 1. Motivación y Alcance

Durante la evaluación del módulo de cálculo de NTC/TTC en la versión oficial de VeraGrid 6.x, hemos realizado un análisis técnico en profundidad de las dos formulaciones disponibles (`regular` y `strict`).

El objetivo de esta nota técnica es trasladar los hallazgos técnicos detectados y detallar las **correcciones puntuales necesarias en el código** para que la formulación estricta (`strict`) sea plenamente operativa, resuelva de forma factible y rápida, y cumpla con la normativa técnica europea (códigos de red CACM y SO GL), que no permite holguras en capacidad comercial ni sobrecargas en contingencia.

---

## 2. Diagnóstico del Estado Actual en VeraGrid 6.x

Al ejecutar el módulo de NTC sobre el caso de estudio de la interconexión (`snapshots/MOU_pmode3_h13.gridcal`, dirección Exportación España → Francia, Pmode3, TRM = 400 MW), se observan los siguientes comportamientos:

| Modo en VeraGrid 6.x | TTC (MW) | Slack Térmico | Slack Área | Estado Solver | Tiempo | Diagnóstico Técnico |
|---|---:|---:|---:|:---:|---:|---|
| **VeraGrid 6 `regular`** | 8.002,26 | 6.080,77 MW | Activo | `Relaxed` | 172 s | Capacidad inflada por absorción de sobrecargas en variables slack. |
| **VeraGrid 6 `strict` (Upstream)** | — | — | — | `Infeasible` | 38 s | Infactible por signo invertido en PST ($-\tau$) y cota $\Delta \ge 0$. |
| **VeraGrid 6 `strict` (Con correcciones)** | **5.298,61** | **0,00 MW** | **0,00 MW** | **`Optimal`** | **27,9 s** | **Resuelve con límites duros, slack cero, GSK activo ($P_{\max}$) y screening Lazy.** |

* **En el modo `regular`:** Al incorporar variables de holgura térmica ($s \ge 0$) penalizadas en la función objetivo, el solver absorbe más de 6.000 MW de sobrecargas en lugar de modular a la baja el escalón de intercambio comercial $\Delta P$. Esto arroja un resultado relajado (8.002 MW) que supera la capacidad física admisible del sistema.
* **En el modo `strict`:** La formulación no llega a resolver porque adolece de tres fallos concretos de implementación (signo del PST, ausencia de screening Lazy y bloqueo de cota inferior en $\Delta$) que provocan infactibilidad matemática o colapso por memoria.

Una vez aplicadas las correcciones descritas a continuación, el modo estricto resuelve de forma natural en **23 segundos** con **Slack = 0 MW**.

---

## 3. Correcciones Necesarias en el Código

### 3.1. Corrección del Signo en Transformadores Desfasadores (PST)
* **Ubicación:** `ntc_opf_strict.py:908` y `ntc_opf.py:1363`.
* **Problema:** El flujo se formula como $f = b(\theta_f - \theta_t - \tau)$.  
  Físicamente, un adelanto angular $\tau > 0$ en el nudo *from* empuja flujo hacia el nudo *to* ($V_m e^{j(\theta + \tau)}$). El signo negativo frena o invierte el flujo, saturando artificialmente las interconexiones pirenaicas y provocando infactibilidades en el solver.
* **Corrección requerida:** Reemplazar por la formulación canónica:
  $$f_m = b_k (\theta_{\text{from}} - \theta_{\text{to}} + \tau_m)$$

### 3.2. Integración del Screening Lazy en el Modo Estricto
* **Ubicación:** `ntc_opf_strict.py` (función de construcción de contingencias).
* **Problema:** En el modo `strict`, el código intentaba inyectar en el LP las restricciones térmicas de todas las ramas para todas las contingencias mediante bucles directos ($7.670 \times 4.000 \times 2 \approx \mathbf{61\text{ millones de restricciones}}$), colapsando la memoria RAM y bloqueando el solver durante horas.
* **Corrección requerida:** Reutilizar el motor de screening matricial *Lazy* (`screen_contingency_violations`) que ya tenéis implementado en `ntc_opf.py`. Esto filtra en NumPy las contingencias y añade al LP únicamente las restricciones activas (~398 filas), reduciendo el tiempo de resolución a **23 segundos**.

### 3.3. Liberación de la Cota Inferior del Intercambio en Rondas Lazy
* **Ubicación:** `ntc_opf_strict.py:2005` y `ntc_opf.py:2080`.
* **Problema:** En cada iteración del lazo lazy se re-fijaba `lb=0` a la variable de intercambio $\Delta$. En escenarios donde el flujo óptimo implica importación o un delta negativo, el problema se volvía infactible por construcción.
* **Corrección requerida:** Declarar `transfer_lb = -INFINITY` en las llamadas de reoptimización del lazo lazy.

### 3.4. Eliminación de Slacks de Área (Balance Comercial 1:1)
* **Ubicación:** Formulación del balance inter-área en `ntc_opf_strict.py`.
* **Problema:** La presencia de variables de holgura en el balance de área ($\Delta_{sl_1}, \Delta_{sl_2}$) permite inyectar o absorber potencia ficticia fuera de la red. Además, en la versión `strict` original, la vinculación algebraica cancelaba el término $\Delta_1$ en el objetivo.
* **Corrección requerida:** Establecer un balancín comercial bilateral estricto:
  $$\Delta P_{A_1} = -\Delta P_{A_2} = \Delta P, \quad \text{Maximizar } \Delta P$$
  Sin variables de holgura en las áreas comercializadoras.

### 3.5. Corrección en el Parser de Carga de Red (`pack_unpack.py`)
* **Ubicación:** `pack_unpack.py` al parsear transformadores y desfasadores de ficheros `.gridcal`.
* **Problema:** El perfil de control temporal `tap_phase_control_mode_prof` se inicializaba por defecto en `Fixed`, pisando el atributo escalar del dispositivo (`Pt`). Esto provocaba que los desfasadores quedaran inmóviles durante la optimización.
* **Corrección requerida:** Asegurar que `prof[0] = Pt` en la carga para que los PST mantengan su capacidad de modulación.

### 3.6. Robustez de Tipos en el Enum `AvailableTransferMode`
* **Ubicación:** `available_transfer_capacity_driver.py` (`compute_dP`).
* **Problema:** En VeraGrid 6, `.value` del Enum devuelve una cadena de texto (`"InstalledPower"`), mientras que `compute_dP` discrimina mediante enteros (`if mode == 1`). Si una llamada pasa directamente el enum o su valor, cae silenciosamente en el bloque `else` (reparto sobre potencia base $P_0$ en vez de potencia instalada).
* **Corrección requerida:** Modificar la firma y comprobaciones de `compute_dP` para aceptar y resolver directamente las instancias de `AvailableTransferMode`.

### 3.7. Filtro de Contingencias Habilitadas (`contingency_enabled`)
* **Ubicación:** `pack_unpack.py:1336`.
* **Problema:** El lector procesa la columna de contingencias como señal de instanciación pero omite comprobar si el flag `contingency_enabled` está a 0, creando contingencias que el usuario había desactivado.
* **Corrección requerida:** Comprobar `enabled != 0` antes de añadir la contingencia al circuito.

### 3.8. Modelado de Contingencias HVDC (Declaración en Modelo vs Hardcodeo)
* **Ubicación:** `linear_analysis.py` (`ContingencyIndices`) y formulación de contingencias en el LP.
* **Problema y objetivo metodológico:**  
  Históricamente, algunos scripts aplicaban la regla del disparo de polo mediante parámetros cableados a mano en el código (asumiendo siempre 2 polos para todo enlace). Nuestro objetivo es **eliminar cualquier hardcodeo del código y trasladar la definición física de la contingencia formalmente al modelo de red**.
  
  En un enlace bipolar de gran capacidad (ej. 2.000 MW), la contingencia más representativa es la pérdida de un solo polo (reducción del 50% de la capacidad de transporte). Cuando esto sucede:
  * El polo superviviente puede transportar hasta su capacidad nominal ($1.000\text{ MW}$).
  * Si el flujo previo era inferior a $1.000\text{ MW}$, el enlace no pierde potencia hacia la red AC ($\Delta f_{\text{spill}} = 0$).
  * Si el flujo previo excedía la capacidad del polo superviviente ($|f_{\text{hvdc}}| > 1.000\text{ MW}$), el polo superviviente satura a $1.000\text{ MW}$ y únicamente el exceso se derrama hacia la red AC:
    $$\text{Derrame} = \max\left(0, \ |f_{\text{hvdc}}| - R_{\text{surv}}\right) \times \text{signo}(f_{\text{hvdc}})$$
  * Ese derrame se redistribuye en cada rama AC $m$ según los factores de distribución del enlace:
    $$f_m^{(c)} = f_m^0 + \text{HvdcODF}[m, \text{hvdc}] \cdot \text{Derrame}$$

* **Solución limpia propuesta (sin tocar la clase `HvdcLine` ni la GUI):**  
  Para que no sea necesario añadir atributos ni modificar la GUI de `HvdcLine`, proponemos reutilizar el tipo de operación de contingencia que VeraGrid ya tiene implementado en su enumeración oficial:
  ```python
  ContingencyOperationTypes.PowerPercentage  # símbolo '%'
  ```
  1. **En el modelo de datos (`contingency.csv`):** El usuario declara explícitamente en el catálogo de contingencias el evento sobre el `HvdcLine` deseado:
     - Para un disparo de 1 polo en un bipolo: `prop = '%'`, `value = 50.0` (o el porcentaje correspondiente).
     - Para un disparo completo del enlace: `prop = 'active'`, `value = 0.0` (o `prop = '%'`, `value = 100.0`).
  2. **En el compilador (`linear_analysis.py`):** Si `cnt.tpe == DeviceType.HVDCLineDevice` y `cnt.prop == ContingencyOperationTypes.PowerPercentage`, el factor de indisponibilidad se extrae como $\text{frac} = \text{value} / 100$, definiendo la capacidad superviviente $R_{\text{surv}} = \text{Rate} \times (1 - \text{frac})$.
  3. **En el LP:** El término de derrame se evalúa con la capacidad superviviente $R_{\text{surv}}$, desacoplando la regla del código y otorgando al modelo el control pleno de si un enlace dispara un polo o el enlace completo.

### 3.9. Filtrado Paritario de Sensibilidad $(m, c)$ e Impacto Topológico en el Screening Lazy
* **Ubicación:** `ntc_opf.py` (`screen_contingency_violations`, `get_contingency_monitorable_branches`) y `ntc_opf_strict.py`.
* **Problema:** En el código oficial de VeraGrid 6.x concurren dos factores en el cribado que distorsionan el problema de optimización:
  1. `get_contingency_monitorable_branches` calculaba un vector booleano global a partir del máximo sobre *todas* las contingencias del sistema:
     $$\alpha_{\max} = \max_{c} |\alpha_{N-1}[m, c]|$$
     Si una rama interna remota superaba el umbral (5%) ante alguna contingencia cualquiera del sistema, quedaba marcada como monitorizable para el screening de **todas** las contingencias. Físicamente, esto inyectaba sobrecargas internas desacopladas (ej. la red local francesa tras Pragnères ante el disparo de un autotransformador de 400/225 kV) como cotas duras al LP, forzando al solver a invertir o ahogar el intercambio comercial.
  2. El screening evaluaba ramas sin exigir un impacto físico real de la contingencia en la rama ($\operatorname{MLODF}_{m, c} \approx 0$) y admitía ramas preventivas al 90% de carga (`at_risk_fraction = 0.9`), lo que en un esquema de límites estrictos (sin variables de holgura) sobrecarga innecesariamente el problema con cotas que no están violadas.
* **Corrección requerida:** Conforme al Reglamento UE 2015/1222 (CACM, Art. 29), una rama $m$ ante una contingencia $c$ solo debe incorporarse al LP si:
  - Presenta impacto físico de la contingencia ($|\operatorname{MLODF}_{m, k}| > 0{,}05$ para algún elemento $k \in c$).
  - Es sensible comercialmente al intercambio ($|\alpha_m^N| > 0{,}05 \lor \max_{k \in c} |\alpha_{N-1}[m, k]| > 0{,}05$).
  - Incurre en una sobrecarga térmica efectiva ($|f_m^{(c)}| > \text{Rate}_m^{(c)}$).

### 3.10. Resolución Directa de HiGHS vía Fichero LP en el Wrapper de PuLP
* **Ubicación:** `src/VeraGridEngine/Utils/MIP/pulp_interface.py` (`PulpLpModel.solve`).
* **Problema:** El backend in-memory de PuLP para HiGHS (`pulp.apis.highs.HiGHS` vía `highspy`) presenta un comportamiento errático cuando el modelo incluye variables binarias (como los indicadores de saturación en Pmode3):
  1. En `buildSolverModel`, PuLP solo marca como enteros los elementos con `var.cat == constants.LpInteger`, omitiendo `constants.LpBinary`, lo que puede provocar que variables binarias queden relajadas en memoria.
  2. El solver en memoria `highspy.Highs().run()` sobre la estructura construida por ctypes puede sufrir fallos de presolve en vértices degenerados, quedando atrapado en mínimos locales espurios y reportando erróneamente `kOptimal` sobre puntos no óptimos (por ejemplo, resolviendo a $0,0645\text{ p.u.}$ en lugar de $-107,38\text{ p.u.}$, arrojando un TTC aparente de $-6,45\text{ MW}$ en la Hora 8280).
* **Corrección requerida:** En `PulpLpModel.solve`, cuando el solver seleccionado es `MIPSolvers.HIGHS`, resolver directamente exportando el modelo al formato canónico `.lp` (`model.writeLP`) y ejecutando `highspy.Highs().readModel(lp_path)`. El parser nativo en C++ de HiGHS preserva al 100% la integrality y la topología canónica, convergiendo sin degradación al óptimo global verdadero en todos los escenarios (8.899,80 MW en H8280 y 8.181,73 MW en H98) en tiempos idénticos (~15-20 s por resolución LP).

### 3.11. Propagación del Acoplamiento Droop de Enlaces HVDC en Pmode3 en la Formulación Estricta
* **Ubicación:** `ntc_opf_strict.py` (`add_linear_branches_contingencies_formulation`).
* **Problema:** En la rama oficial, la formulación estricta (`strict`) omite el término aditivo de respuesta pasiva droop de los enlaces HVDC ante el disparo de líneas de alterna (`droop_addons`). Al producirse una contingencia AC, los enlaces HVDC en Pmode3 modifican su flujo pasivamente por variación del ángulo en bornas $\Delta\theta$; sin embargo, en `ntc_opf_strict.py` este efecto no se transfería a las llamadas de `contingency.get_lp_contingency_flows`, dejando a la red de alterna sin la compensación o sobrecarga derivada del estatismo de continua en las interconexiones.
* **Corrección requerida:** Calcular el término `droop_addon` evaluado en el punto de operación para cada contingencia admitida y propagarlo en la llamada `contingency.get_lp_contingency_flows(..., droop_addon=addon_c)`. Asimismo, restringir las cotas del LP a las ramas efectivamente admitidas en la ronda lazy (`rows_to_limit`), garantizando consistencia física plena.

### 3.12. Supresión de la Segunda Etapa ("Least Control Effort") y Maximización Directa del Flujo de Interconexión
* **Ubicación:** `ntc_opf_strict.py` (`run_linear_ntc_opf_strict`, llamadas a `solve_least_control_effort`).
* **Problema:** Tras converger la resolución del LP con cribado lazy, se ejecutaba una segunda etapa (`solve_least_control_effort`) cuyo objetivo fijaba el intercambio comercial ($\Delta_1$) y minimizaba el esfuerzo de control sumando penalizaciones por desplazamiento angular de desfasadores ($\sum \frac{1}{X} |\Delta \tau|$) y potencia en enlaces de continua ($\sum |P_{\text{hvdc}}|$). Esta segunda etapa distorsionaba el despacho óptimo:
  1. Penalizaba el flujo por los enlaces HVDC, reduciendo su evacuación forzada hacia Francia y vertiendo ese flujo sobre la red AC de interconexión (saturando prematuramente ramas críticas como Vic–Baixas y Hernani–Argia).
  2. Forzaba a los transformadores desfasadores (PST) a replegar sus tomas hacia el neutro, desoptimizando el reparto de cargas preventivo entre base y contingencias.
  3. Modificaba la función objetivo de maximización de intercambio comercial puro, sustituyéndola por una ponderación arbitraria de control.
* **Corrección requerida:** Suprimir íntegramente la llamada a `solve_least_control_effort` en la formulación estricta. El cálculo de NTC es una optimización matemática preventiva en una sola etapa donde las tomas de los PST y los flujos HVDC en Pmode3 se co-optimizan simultáneamente para el estado base y todas las contingencias admitidas. Asimismo, la función objetivo se alinea estrictamente con la formulación de interconexión:
  $$\max \left( \sum_{k \in \text{AC inter}} \text{sign}_k \cdot P_k + \sum_{h \in \text{HVDC inter}} \text{sign}_h \cdot P_h \right)$$
  donde $\text{sign}_k \in \{+1.0, -1.0\}$ se obtiene mediante `get_inter_areas()` a partir de los nudos *from* y *to* de la línea respecto a las áreas de origen y destino. Esta formulación garantiza que todo flujo hacia Francia sume en positivo y cancela algebraicamente cualquier bucle de recirculación entre fronteras, impidiendo que recirculaciones espurias inflen la capacidad calculada.

### 3.13. Clave de Reparto GSK en `AvailableTransferMode.InstalledPower`: Potencia Activa $P_{\max}$ vs Aparente $S_{\text{nom}}$
* **Ubicación:** `ntc_opf_strict.py:161` y `ntc_opf.py:157` (`get_transfer_power_scaling_per_bus`).
* **Problema:** Para el modo `InstalledPower`, el código asignaba:
  ```python
  pinst_per_bus = gen_data_t.get_installed_power_per_bus() / Sbase
  p_ref = pinst_per_bus
  ```
  donde `installed_p = elm.Snom` (potencia aparente nominal en MVA). En cálculos de transferencia y capacidad comercial según normativa CACM / ENTSO-E, la clave de reparto de generación (*Generation Shift Key*, GSK) proporcional a la potencia instalada debe formularse sobre la **capacidad activa máxima disponible** ($P_{\max}$ en MW).
  En el modelo peninsular e interconexiones europeas coexisten generadores (térmicos, hidroeléctricos, parques eólicos y solares) donde el alternador o convertidor está sobredimensionado ($S_{\text{nom}} \gg P_{\max}$). En el modelo del MOU, **1.976 generadores presentan $S_{\text{nom}} \ne P_{\max}$**. Asignar la proporción sobre $S_{\text{nom}}$ distorsiona el reparto nodal de inyecciones comerciales $\Delta P$, sobrecargando prematuramente líneas de interconexión (rama 2576 Argia–Arkale) a un TTC inferior (5.180 MW).
* **Corrección requerida:** Asignar la referencia de potencia instalada a partir de la potencia activa máxima por nudo:
  ```python
  p_ref = gen_data_t.get_pmax_per_bus() / Sbase
  ```
  Con esta corrección, el vector de inyecciones del intercambio se distribuye de forma estrictamente proporcional a la potencia activa instalada y reproducible, alcanzando paridad numérica exacta con la referencia histórica del MOU: **5.298,61 MW** frente a **5.297,10 MW** (diferencia de +1,51 MW, un 0,028%).

### 3.14. Erradicación de la Susceptancia Parásita en Factores Lineales (`linear_analysis.py`)
* **Ubicación:** `src/VeraGridEngine/Simulations/LinearFactors/linear_analysis.py:411, 430`.
* **Problema:** En el ensamblado de la matriz de susceptancia nodal para cálculo lineal DC, se introdujo una susceptancia artificial parásita de acoplamiento de $y_s = 0,01\text{ p.u.}$ en convertidores VSC sin estatismo activo y en líneas HVDC con consigna fija de potencia $P_{\text{set}}$.
  Físicamente, un enlace de corriente continua en control de potencia fija o sin estatismo no constituye un camino pasivo de alterna regido por la ley de Ohm en continua ($I \ne b \Delta\theta$). Introducir un valor artificial $y_s = 0,01$:
  1. Crea lazos ficticios de circulación AC a través de los enlaces DC en la matriz $B_{\text{bus}}$, violando las leyes de Kirchhoff en corriente alterna.
  2. Altera la matriz de sensibilidades PTDF ($\partial f / \partial P$), desvirtuando los factores de distribución transfronterizos hasta en varios puntos porcentuales.
* **Corrección requerida:** Fijar estrictamente $y_s = 0,0$ para todos los dispositivos DC que no operen en modo estatismo angular puro. Esto asegura que la matriz de susceptancias refleje exclusivamente la topología física de alterna y garantiza la coherencia rigurosa de las sensibilidades PTDF.

### 3.15. Unificación Metodológica: Adopción Exclusiva de la Formulación Estricta sin Holguras
* **Ubicación:** Integración global del módulo NTC (`ntc_driver.py`).
* **Problema:** La coexistencia de dos formulaciones (`regular` con penalizaciones de holgura $s \ge 0$, y `strict` con límites duros) genera confusión operativa y resultados contradictorios. La formulación `regular` tolera sobrecargas masivas post-contingencia (absorbidas en variables slack de holgura térmica), lo cual infringe frontalmente el mandato del Reglamento (UE) 2015/1222 (CACM) y el P.O. 3.1 de REE, que exigen capacidades firmes y seguras en N y N-1.
* **Corrección requerida:** Estandarizar la formulación estricta (`strict_formulation = True`) como la única operativa en VeraGrid 6.x. Con las correcciones de screening Lazy, signo PST y GSK aquí detalladas, el modo estricto resuelve de manera factible, óptima y en tiempos de cómputo excelentes (~25 segundos), eliminando la necesidad de recurrir a relajaciones artificiales con holguras térmicas.

---

## 4. Corrección en el Volcado de Resultados al Snapshot (`ntc_driver.py`)

* **Ubicación:** `ntc_driver.py:163` (`get_ntc_operating_point_as_opf_results`).
* **Problema:** La función calcula un supuesto desbalance de red (`imbalance = gen + batt - load`) y redistribuye las pérdidas alterando la potencia base de todos los generadores del modelo antes de aplicar el intercambio comercial:
  ```python
  imbalance = float(gen_injection.sum() + battery_injection.sum() + load_injection.sum())
  balanced_gen -= gen_injection / gen_total * imbalance
  ```
  Esto desvirtúa el despacho base original de los generadores ajenos al intercambio.
* **Corrección requerida:** Eliminar este rebalanceo global. Los incrementos de generación $\Delta P$ resueltos por el LP deben aplicarse **exclusivamente sobre los generadores de las áreas participantes en el intercambio**.

---

## 5. Demostración en Red Canónica Pública (IEEE 14 del Issue 372)

Para verificar estas correcciones en un entorno abierto directamente reproducible en vuestro repositorio (`IEEE14 - ntc areas_voltages_hvdc_shifter_l10free.gridcal`):

1. **Efecto del signo del desfasador (PST):**
   * Con signo upstream ($-\tau$): La rama 7 se satura erróneamente al 100% (20 MW), reduciendo artificialmente el TTC N-1 a **78,07 MW**.
   * Con signo corregido ($+\tau$): Flujo físico coherente y TTC N-1 de **96,26 MW**.
2. **Efecto de límites duros frente a holguras (Rama crítica limitada a 4 MW):**
   * Formulación `regular` (holguras): TTC inflado a **17,05 MW** con 12 MW de violación térmica post-falta (flujo al 400% de la capacidad).
   * Formulación estricta corregida: TTC modulado de forma segura a **10,49 MW** con **Slack = 0,00 MW** y flujo en 4,00 MW (100%).

---

## 6. Conclusión

Estas correcciones permiten que la formulación estricta de VeraGrid 6.x sea 100% factible, numéricamente robusta, rápida (23 s gracias al screening Lazy) y conforme a las exigencias de operación segura de la red.

Quedamos a vuestra disposición para facilitar los parches y colaborar en la integración de estas correcciones en la rama oficial de VeraGrid.
