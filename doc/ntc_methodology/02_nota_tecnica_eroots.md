# Nota Técnica: Correcciones y Cierre del Módulo NTC en VeraGrid 6

**Para:** Equipo de Desarrollo de eRoots / VeraGrid  
**De:** Equipo de Red Eléctrica de España (REE)  
**Fecha:** 19 de noviembre de 2026  
**Asunto:** Propuesta consolidada de correcciones, alineación metodológica y validación final del módulo NTC/TTC en VeraGrid 6.x (`ntc-finalist-6x`)

---

## 1. Motivación y Alcance

Desde REE valoramos muy positivamente los avances arquitectónicos y metodológicos incorporados en VeraGrid 6.x, en particular el modelado físico de la respuesta pasiva de enlaces HVDC/VSC en modo *angle droop* (Pmode 3), el cribado *Lazy* vectorizado fuera del LP, la formulación compacta de contingencias (`CompactContingencies`), el recorte exacto de flujo (`add_clipped_flow`) y la infraestructura multiterminal para convertidores VSC.

Durante las pruebas de aceptación sobre el modelo de referencia de la interconexión España–Francia (MOU 2023, Pmode 3 y Pmode 1, con un margen $\text{TRM} = 400\text{ MW}$ y sentido exportador $\text{ES} \to \text{FR}$) y sobre redes públicas del repositorio, identificamos un conjunto acotado de defectos e inconsistencias entre las dos formulaciones existentes (`regular` y `strict`), la interfaz gráfica (GUI) y los factores lineales de continua.

El propósito de esta nota técnica es trasladar de forma clara y reproducible **las correcciones técnicas ya implementadas y validadas en la rama finalista `ntc-finalist-6x`**, así como los resultados obtenidos sobre las **12 horas canónicas del MOU** (tanto con enlaces `HvdcLine` como con convertidores `VSC`) y sobre la red pública IEEE 14, para facilitar su revisión e integración en la rama oficial de VeraGrid.

El criterio regulatorio que rige toda la propuesta se deriva directamente de los Reglamentos (UE) 2015/1222 (CACM), (UE) 2017/1485 (SO GL) y de los Procedimientos de Operación de REE (P.O. 3.1): **balance comercial bilateral 1:1 sin holguras de área y seguridad térmica estricta en estado base $N$ y post-contingencia $N-1$ sin variables *slack* térmicas**.

---

## 2. Resumen de Resultados Validados

### 2.1. Diagnóstico y Paridad en la Hora Canónica de Referencia (Hora 13 EF, Pmode 3)

| Motor / Formulación en VeraGrid 6.x | TTC (MW) | NTC (MW) | Slack Térmico | Slack Área | Carga Máx. | Estado y Diagnóstico Técnico |
|---|---:|---:|---:|---:|---:|---|
| **VeraGrid 6.7.20 `regular` (Upstream)** | 8.002,26 | 7.602,26 | 6.080,77 MW | Activo | > 100 % | Inválido: absorbe sobrecargas masivas post-falta en variables *slack*. |
| **VeraGrid 6.7.20 `strict` (Upstream)** | — | — | — | — | — | Infactible por signo invertido en PST ($-\tau$), cota $\Delta \ge 0$ y falta de *lazy screening*. |
| **VeraGrid 6.x `strict` sin droop N-1 (`HvdcLine`)** | 5.298,61 | 4.898,61 | 0,00 MW | 0,00 MW | 193,27 %* | Sin realimentación pasiva Pmode 3 en N-1 (*). |
| **VeraGrid 6.x `ntc-finalist-6x` (`HvdcLine`)** | **5.255,51** | **4.855,51** | **0,00 MW** | **0,00 MW** | **100,00 %** | **Validado:** droop N-1 en lazo cerrado, polo `%` y límites duros. |
| **VeraGrid 6.x `ntc-finalist-6x` (`VSC` + bus DC)** | **5.255,51** | **4.855,51** | **0,00 MW** | **0,00 MW** | **100,00 %** | **Paridad exacta 0,00 MW** respecto a `HvdcLine`. |

> \* *Nota:* Sin el acoplamiento de estatismo pasivo N-1 en la formulación estricta, el punto resuelto para Hora 13 presenta `max_loading_pct = 193,27 %` al reevaluarse posteriormente la respuesta pasiva real del enlace frente a la contingencia `13011_BESCANO_400_13060_SENTMENA_400`. Al acoplar el estatismo pasivo N-1 en lazo cerrado dentro del propio LP, la carga máxima converge exactamente al **100,00 %** en todas las horas y tipologías.

---

### 2.2. Validación Completa en 12 Horas Canónicas (`ntc-finalist-6x`: `HvdcLine` vs `VSC`)

Ejecución estricta (`strict_formulation=True`, `consider_contingencies=True`, `corrective_contingencies=False`, `include_hvdc_exchange_sensitivity=False`, `skip_generation_limits=True`, $\text{TRM}=400\text{ MW}$) sobre las 12 horas canónicas del MOU (`[13, 22, 29, 66, 98, 144, 641, 1813, 3684, 3967, 8056, 8280]`), tanto en **Pmode 3** (estatismo angular pasivo) como en **Pmode 1** (consigna modulable), con modelos equivalentes `HvdcLine` (`MOU_pmode3_12h_hvdc_poles.gridcal` / `MOU_pmode1_12h.gridcal`) y convertidores `VSC` (`MOU_pmode3_12h_vsc.gridcal` / `MOU_pmode1_12h_vsc.gridcal`):

| Hora | Pmode 3 Ref. MOU (sin droop N-1) | Pmode 3 `HvdcLine` (`ntc-finalist-6x`) | Pmode 3 `VSC` (`ntc-finalist-6x`) | Dif. Pmode 3 VSC − HVDC | Pmode 1 Ref. MOU | Pmode 1 `HvdcLine` (`ntc-finalist-6x`) | Pmode 1 `VSC` (`ntc-finalist-6x`) | Dif. Pmode 1 VSC − HVDC | Slack / Carga Máx. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **13** | 5.297,0 | **5.255,51** | **5.255,51** | **0,00 MW** | 6.034,0 | **6.048,16** | **6.048,16** | **0,00 MW** | 0,00 MW / 100,00 % |
| **22** | 6.515,0 | **6.295,97** | **6.295,97** | **0,00 MW** | 7.562,0 | **7.562,54** | **7.562,54** | **0,00 MW** | 0,00 MW / 100,00 % |
| **29** | 8.853,0 | **8.873,70** | **8.873,70** | **0,00 MW** | 9.576,0 | **9.733,09** | **9.733,09** | **0,00 MW** | 0,00 MW / 100,00 % |
| **66** | 6.699,0 | **6.722,28** | **6.722,28** | **0,00 MW** | 7.676,0 | **7.737,88** | **7.737,88** | **0,00 MW** | 0,00 MW / 100,00 % |
| **98** | 8.624,0 | **8.645,57** | **8.645,57** | **0,00 MW** | 9.524,0 | **10.112,05** | **10.112,05** | **0,00 MW** | 0,00 MW / 100,00 % |
| **144** | 9.010,0 | **9.028,43** | **9.028,43** | **0,00 MW** | 9.369,0 | **9.396,96** | **9.396,96** | **0,00 MW** | 0,00 MW / 100,00 % |
| **641** | 5.199,0 | **5.212,66** | **5.212,66** | **0,00 MW** | 6.170,0 | **6.228,79** | **6.228,79** | **0,00 MW** | 0,00 MW / 100,00 % |
| **1813** | 3.759,0 | **3.777,63** | **3.777,63** | **0,00 MW** | 5.106,0 | **5.115,62** | **5.115,62** | **0,00 MW** | 0,00 MW / 100,00 % |
| **3684** | −3.055,0 | **−3.046,98** | **−3.046,98** | **0,00 MW** | −2.102,0 | **−2.102,04** | **−2.102,04** | **0,00 MW** | 0,00 MW / 100,00 % |
| **3967** | 3.433,0 | **3.436,84** | **3.436,84** | **0,00 MW** | 4.964,0 | **4.964,70** | **4.964,70** | **0,00 MW** | 0,00 MW / 100,00 % |
| **8056** | 9.033,0 | **9.057,15** | **9.057,15** | **0,00 MW** | 9.391,0 | **9.403,61** | **9.403,61** | **0,00 MW** | 0,00 MW / 100,00 % |
| **8280** | 8.994,0 | **9.015,92** | **9.015,92** | **0,00 MW** | 9.882,0 | **9.943,03** | **9.943,03** | **0,00 MW** | 0,00 MW / 100,00 % |

* **Convergencia y factibilidad:** 12/12 horas en estado `Optimal` en todas las variantes (`HvdcLine` Pmode 3, `VSC` Pmode 3, `HvdcLine` Pmode 1 y `VSC` Pmode 1), con **`0,00 MW` de slack** y carga máxima exactamente igual al **`100,00 %`**.
* **Equivalencia exacta `HvdcLine` ↔ `VSC`:** desviación máxima inferior a **`0,000001 MW`** en las 24 ejecuciones horarias (`max|ΔPmode3| = 0,000000 MW`, `max|ΔPmode1| = 0,000001 MW`).

---

## 3. Correcciones Técnicas Implementadas en `ntc-finalist-6x`

Todas las modificaciones enumeradas a continuación están consolidadas en código y verificadas sin degradar las pruebas de regresión estrictas.

### 3.1. Signo Físico en Transformadores Desfasadores (PST)
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`add_linear_branches_formulation`) y `ntc_opf.py`.
* **Problema:** El flujo de ramas con desfasador se formulaba como $f_m = b_k(\theta_{\text{from}} - \theta_{\text{to}} - \tau_m)$. Físicamente, un adelanto angular $\tau_m > 0$ en el nudo *from* impulsa potencia activa hacia *to* ($V_{\text{from}} e^{j(\theta_{\text{from}}+\tau_m)}$). El signo invertido saturaba las interconexiones pirenaicas y producía infactibilidad espuria.
* **Corrección aplicada:** Formulación física unificada con signo positivo:
  $$f_m = b_k (\theta_{\text{from}} - \theta_{\text{to}} + \tau_m)$$

### 3.2. Integración del Screening *Lazy* Matricial en la Formulación Estricta
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`run_linear_ntc_opf_strict`) y `ntc_opf.py` (`screen_contingency_violations`).
* **Problema:** En `strict`, se intentaban añadir todas las restricciones rama × contingencia de golpe ($7.670 \times 4.000 \times 2 \approx 61{,}4\text{ millones de filas}$), provocando agotamiento de memoria y bloqueos de horas.
* **Corrección aplicada:** Uso del screening *Lazy* vectorizado en NumPy (`screen_contingency_violations`), incorporando al LP únicamente las restricciones realmente activas en cada ronda de corte hasta certificar violación nula en todo el universo de contingencias.

### 3.3. Liberación de la Cota Inferior del Intercambio en Rondas *Lazy*
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` y `ntc_opf.py` (`solve_least_control_effort`).
* **Problema:** En el lazo iterativo *lazy* se re-imponía la cota `lb = 0` sobre la variable de intercambio $\Delta P$. En horas importadoras o con flujo base sobrecargado (como Hora 3684, $\text{TTC} = -3.046{,}98\text{ MW}$), el problema resultaba infactible por construcción.
* **Corrección aplicada:** Se fija `transfer_lb = -INFINITY`, permitiendo que el dominio comercial admita libremente exportación positiva o reducción/importación negativa según los límites térmicos.

### 3.4. Balance Comercial Bilateral Estricto 1:1 sin Holguras de Área ni Térmicas
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`structural_area_balance=True`).
* **Problema:** La inclusión de variables de holgura térmica ($s \ge 0$) o de área permitía absorber sobrecargas ficticias (caso `regular`: $6.080{,}77\text{ MW}$ de slack), y la vinculación algebraica previa de slacks de área cancelaba el término $\Delta_1$ del objetivo.
* **Corrección aplicada:** Desplazamiento bilateral simétrico puro ($\Delta P_{A_1} = -\Delta P_{A_2} = \Delta P$) sin holguras térmicas ni de área en todo el bloque estricto.

### 3.5. Parser de Carga de Red (`pack_unpack.py`)
* **Ubicación:** `src/VeraGridEngine/IO/veragrid/pack_unpack.py`.
* **Problema:** Al importar `.gridcal`: (a) `tap_phase_control_mode_prof` se inicializaba a `Fixed`, anulando el modo `Pt` del transformador e inmovilizando los PST; y (b) el parser de columnas `contingency_enabled` instanciaba contingencias aunque el flag estuviese a `0`.
* **Corrección aplicada:** Preservación de `prof[0] = Pt` en el perfil de control de fase durante la carga y filtrado explícito `enabled != 0` al generar contingencias heredadas.

### 3.6. Robustez de Tipos en el Enum `AvailableTransferMode`
* **Ubicación:** `src/VeraGridEngine/Simulations/ATC/available_transfer_capacity_driver.py` (`compute_dP`).
* **Problema:** `compute_dP` comparaba el modo mediante enteros (`mode == 1`), mientras que VeraGrid 6 utiliza enumerados con valor textual (`AvailableTransferMode.InstalledPower`). Las llamadas con el enumerado caían silenciosamente en la rama por defecto ($P_0$).
* **Corrección aplicada:** Soporte directo y uniforme de las instancias `AvailableTransferMode` dentro de `compute_dP`.

### 3.7. Contingencias Parciales de Polo en `HvdcLine` y `VSC` (`PowerPercentage` `'%'` y Derrame con Signo)
* **Ubicación:** `src/VeraGridEngine/Simulations/LinearFactors/linear_analysis.py` (`LinearMultiContingency.pole_spill`, `LinearMultiContingencies.compute`), `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`compute_pole_spill_expressions`), y `src/VeraGridEngine/Simulations/NTC/ntc_opf.py` (`fill_worst_contingency_per_branch`).
* **Contexto y problema:**
  1. Para representar el disparo de un polo en un enlace bipolar sin introducir atributos propietarios ni cablear constantes en el motor (`n_poles = 2`), basta reutilizar el tipo oficial `ContingencyOperationTypes.PowerPercentage` (`prop = '%'`, por ejemplo `value = 50.0` para perder el 50 % de la capacidad nominal). El polo superviviente conserva $R_{\text{surv}} = R \cdot (1 - \text{frac})$ y sólo el excedente se derrama hacia la red AC.
  2. En el cálculo algebraico previo (`flow - r_surv`), el término omitía el signo del flujo: con $f = -1.622\text{ MW}$ y $R_{\text{surv}} = 1.000\text{ MW}$, declaraba $-2.622\text{ MW}$ de derrame en lugar de $-622\text{ MW}$.
  3. En `fill_worst_contingency_per_branch` (`ntc_opf.py`), `hvdc_surviving_rates` y `vsc_surviving_rates` no estaban escalados por `sbase_val` al compararse con flujos en p.u.
* **Corrección aplicada:**
  1. Soporte nativo de `PowerPercentage` (`frac = value / 100.0`) tanto para `HvdcLine` como para `VSC`.
  2. Expresión canónica compartida `pole_spill`:
     $$\Delta f_{\text{spill}} = \operatorname{sign}(f) \cdot \max\left(0,\; |f| - R_{\text{surv}}\right)$$
     formulada exactamente en el LP (`compute_pole_spill_expressions`) reutilizando `add_clipped_flow` (`spill = flow - carried`, con `carried` acotado a $[-R_{\text{surv}}, +R_{\text{surv}}]$) y las cotas exactas del propio enlace, evitando relajaciones que permitirían aliviar restricciones en ramas con ODF de signo opuesto.
  3. Escalado uniforme de `hvdc_surviving_rates` y `vsc_surviving_rates` en `sbase_val` dentro del reporte de peores contingencias (`f34347d23`).

### 3.8. Filtrado Paritario $(m, c)$ en Screening *Lazy* y Re-evaluación de Saturación N-1
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf.py` (`screen_contingency_violations`, `get_contingency_monitorable_branches`) y `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py`.
* **Problema:**
  1. `get_contingency_monitorable_branches` aplicaba $\alpha_{\max} = \max_c |\alpha_{N-1}[m, c]|$ de forma global, arrastrando ramas internas remotas con sobrecargas base o locales no sensibles a esa contingencia concreta (caso Marsip 5050 en Hora 98).
  2. Al cambiar la saturación de los enlaces HVDC/VSC entre rondas *lazy*, pares $(m, c)$ ya añadidos al LP con otro régimen de saturación podían quedar desactualizados si no se re-evaluaban con nombre de restricción único por ronda.
* **Corrección aplicada:**
  1. Criterio paritario por par rama-contingencia $(m, c)$ acorde con CACM Art. 29: una restricción entra únicamente si presenta impacto físico de la contingencia ($|\operatorname{MLODF}_{m,k}| > 0{,}05$ para algún $k \in c$), sensibilidad comercial ($|\alpha_m^N| > 0{,}05$ o $\max_{k \in c} |\alpha_{N-1}[m,k]| > 0{,}05$) y sobrecarga térmica efectiva ($|f_m^{(c)}| > \text{Rate}_m^{(c)}$).
  2. Re-evaluación automática de pares ya añadidos (`recheck_already`) cuando el régimen de saturación del estatismo cambia entre rondas *lazy*, asignando identificadores de restricción únicos por ronda (`_r{lazy_round}`) (`f34347d23`).

### 3.9. Resolución Directa de HiGHS vía Fichero LP con `presolve="off"`
* **Ubicación:** `src/VeraGridEngine/Utils/MIP/pulp_interface.py` (`PulpLpModel.solve`).
* **Problema:** El puente in-memory de PuLP hacia HiGHS omitía marcas de integralidad en variables `LpBinary` y sufría fallos de *presolve* y *postsolve* sobre vértices degenerados, reportando falsos óptimos o falsos infactibles (por ejemplo, Hora 8280 caía a $-6{,}45\text{ MW}$).
* **Corrección aplicada:** Exportación directa del LP mediante `model.writeLP(lp_path)` y resolución a través del lector nativo C++ `highspy.Highs().readModel(lp_path)` con `presolve="off"` y reintento robusto, preservando el 100 % de las variables binarias y convergiendo al óptimo global en las 12 horas.

### 3.10. Estatismo Pasivo N-1 en Lazo Cerrado Unificado para `HvdcLine` y `VSC` (Pmode 3 y Pmode 1)
* **Ubicación:** `src/VeraGridEngine/Simulations/LinearFactors/linear_analysis.py` (`LinearAnalysis`, `LinearMultiContingency.compute_hvdc_droop_addon`, `compute_hvdc_droop_lp_addon`, `LinearMultiContingencies.compute`), `ntc_opf_strict.py` (`add_linear_branches_contingencies_formulation`), y `ntc_opf.py` (`screen_contingency_violations`, `fill_worst_contingency_per_branch`, `run_linear_ntc_opf`).
* **Problema:**
  1. En `strict`, la respuesta pasiva de estatismo angular ante contingencias AC (`droop_addon`) no se calculaba ni se propagaba dentro del LP. Además, un enfoque en lazo abierto sobreestima el alivio post-falta al ignorar que la propia inyección DC descarga la red AC y frena la apertura angular.
  2. El soporte en Pmode 1 y en Pmode 3 presentaba carencias para convertidores `VSC`: uso de `grid.vsc_converters` en lugar de `grid.vsc_devices`, ausencia de `vsc_vars.inter_space_vsc`, falta de factores de estatismo pasivo N-1 para `VSC` en `Pdc_angle_droop` y singularidad en la factorización de $B_{\text{bus}}$ al incluir buses DC aislados.
* **Corrección aplicada:**
  1. Formulación acoplada en **lazo cerrado** mediante la impedancia de Thévenin post-contingencia $Z_{\text{th}}^{(c)}$ entre terminales DC, propagada como expresión lineal exacta tanto en el screening como en el LP estricto (`compute_hvdc_droop_lp_addon`).
  2. Extensión completa del mecanismo a convertidores `VSC` en `Pdc_angle_droop` (`39e8713f9`):
     - Ensamblado de $B_{\text{bus}}$ exclusivamente sobre ramas AC pasivas activas con un nudo de referencia por componente conexa AC (evitando singularidad en subredes DC).
     - Retorno de corriente del corredor DC sobre los convertidores reguladores `Vm_dc` de cada componente conexa DC (`dc_comp_slacks`) y tope por capacidad del corredor DC equivalente.
     - Resolución conjunta VSC + HVDC dentro de `LinearMultiContingencies.compute` y cableado de `vsc_flow` / `vsc_vars.flows` en `ntc_opf_strict.py` y `ntc_opf.py` (saltando el *droop* AC en disparos de ramas DC).
  3. Corrección integral de `VSC` Pmode 1 (`grid.vsc_devices`, `inter_space_vsc`) con paridad exacta respecto a `HvdcLine`.

### 3.11. Supresión de la Segunda Etapa (*Least Control Effort*) y Objetivo Directo
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`run_linear_ntc_opf_strict`).
* **Problema:** La segunda etapa (`solve_least_control_effort`) fijaba el intercambio y minimizaba el esfuerzo de control en PST y HVDC, replegando tomas preventivas y penalizando la evacuación HVDC hacia Francia, lo que volcaba flujo sobre la red AC pirenaica y deformaba la solución óptima.
* **Corrección aplicada:** Eliminación de la llamada a `solve_least_control_effort` en la formulación estricta. El cálculo NTC preventivo se resuelve en una única etapa maximizando directamente el corte físico neto de interconexión (AC + HVDC + VSC).

### 3.12. Clave de Reparto GSK en `InstalledPower` sobre $P_{\max}$ Activo y Unificación Screening–LP
* **Ubicación:** `src/VeraGridEngine/Compilers/circuit_to_data.py`, `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`get_transfer_power_scaling_per_bus`) y `ntc_driver.py`.
* **Problema:** `AvailableTransferMode.InstalledPower` acumulaba `elm.Snom` (MVA aparentes) sin filtrar unidades inactivas o no despachables. En el caso MOU existen 1.976 generadores con $S_{\text{nom}} \ne P_{\max}$, lo que distorsionaba el reparto nodal en el LP y diluía las sensibilidades $\alpha$ calculadas en el screening respecto a las proporciones reales del optimizador.
* **Corrección aplicada:**
  1. Acumulación de `installed_p` y `bus_data.installed_power` en `circuit_to_data.py` exclusivamente a partir de $P_{\max}$ de generadores activos y despachables (`active` y `dispatchable`).
  2. Uso consistente de `gen_data_t.get_pmax_per_bus() / Sbase` en `ntc_opf_strict.py`, unificando la clave de reparto entre el screening $\alpha$ y las restricciones del LP.

### 3.13. Eliminación de la Susceptancia Parásita en Factores Lineales
* **Ubicación:** `src/VeraGridEngine/Simulations/LinearFactors/linear_analysis.py` (`LinearAnalysis`, líneas 411 y 430).
* **Problema:** Para evitar singularidades se inyectaba una susceptancia artificial $y_s = 0{,}01\text{ p.u.}$ en $B_{\text{bus}}$ sobre enlaces HVDC en consigna fija y convertidores VSC sin estatismo, creando caminos virtuales AC a través de enlaces de continua y alterando la matriz PTDF.
* **Corrección aplicada:** Fijación de $y_s = 0{,}0$ en todo enlace o convertidor que no opere en estatismo angular pasivo (`88d9842f6`).

### 3.14. Unificación Estricta, Exposición de Opciones en GUI y Alineación de Defaults
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_options.py`, `ntc_ts_driver.py`, `ntc_opf.py`, `ntc_opf_strict.py`, `src/VeraGrid/Gui/Main/MainWindow.ui`, `MainWindow.py`, `configuration.py` y `simulations.py` (`df483d637`).
* **Problema:** En la GUI faltaban controles relevantes o tenían valores por defecto incompatibles con la formulación validada: `include_hvdc_exchange_sensitivity` estaba oculto y activo por defecto (mezclando sensibilidades DC sin control visible del usuario), `corrective_ntc_contingencies_checkBox` no existía en pantalla, y tanto `strict_ntc_formulation_checkBox` como `consider_ntc_contingencies_checkBox` arrancaban desmarcados. Además, los umbrales de dispersión PTDF/LODF por defecto en la GUI (`0,05`) descartaban sensibilidades inferiores al 5 % en las ejecuciones manuales respecto al valor validado (`0,001`).
* **Corrección aplicada:**
  1. Exposición en la GUI de `include_hvdc_exchange_sensitivity_checkBox` (por defecto `False`, aplicando en el LP estricto la unión $\max(|\alpha_{\text{AC}}|, |\alpha_{\text{ACDC}}|)$ cuando el usuario lo marca) y `corrective_ntc_contingencies_checkBox` (por defecto `False`, cálculo preventivo estricto).
  2. Alineación de valores por defecto tanto en `OptimalNetTransferCapacityOptions` como en la GUI (`MainWindow.ui` / `MainWindow.py` / `configuration.py` / `simulations.py`):
     - `strict_formulation = True`
     - `consider_contingencies = True`
     - `corrective_contingencies = False`
     - `include_hvdc_exchange_sensitivity = False`
     - `ptdf_threshold = 0,001` y `lodf_threshold = 0,001`
  3. **Aviso de despliegue para usuarios con instalación previa:** VeraGrid persiste las últimas preferencias en `~/.VeraGrid/gui_config.json` (en Windows `C:\Users\<usuario>\.VeraGrid\gui_config.json`). Tras actualizar a esta versión, se recomienda borrar o restablecer dicho fichero para que la interfaz adopte los nuevos valores por defecto validados.

---

## 4. Punto Pendiente en el Volcado de Resultados al Snapshot (`ntc_driver.py`)

* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_driver.py` (`get_ntc_operating_point_as_opf_results`, líneas 160–165).
* **Observación técnica:** Al exportar el punto de operación NTC hacia resultados OPF/Power Flow, la función calcula un desbalance base (`imbalance = gen + batt - load`) y lo redistribuye proporcionalmente sobre todos los generadores de la red antes de aplicar `bus_change`:
  ```python
  imbalance = float(gen_injection.sum() + battery_injection.sum() + load_injection.sum())
  balanced_gen -= gen_injection / gen_total * imbalance
  ```
* **Recomendación de cierre:** Suprimir esta redistribución global previa sobre generadores ajenos al intercambio comercial para que el punto exportado conserve intacto el despacho base fuera de las áreas $\mathcal{A}_1$ y $\mathcal{A}_2$.

---

## 5. Verificación Reproducible (Red Pública IEEE 14 y Test Unitario VSC)

### 5.1. Demostración en Red Canónica Pública (IEEE 14 del Issue 372)
Sobre el fichero público del repositorio (`IEEE14 - ntc areas_voltages_hvdc_shifter_l10free.gridcal`):
1. **Efecto del signo del desfasador (PST):**
   - Con signo upstream ($-\tau$): la rama 7 satura erróneamente al 100 % (20 MW), reduciendo el TTC N-1 a **78,07 MW**.
   - Con signo corregido ($+\tau$): flujo físico coherente y TTC N-1 de **96,26 MW**.
2. **Efecto de límites duros frente a holguras térmicas (rama crítica limitada a 4 MW):**
   - Formulación `regular`: TTC inflado a **17,05 MW** tolerando **12,00 MW** de sobrecarga térmica post-contingencia (400 % de carga).
   - Formulación estricta (`strict`): TTC modulado de forma segura a **10,49 MW** con **Slack = 0,00 MW** y flujo exacto en **4,00 MW** (100 %).

### 5.2. Test Automatizado de Equivalencia HVDC ↔ VSC en Pmode 3 N-1
En `src/tests/NTC/test_ntc_strict_formulation.py` se ha incorporado la prueba unitaria `test_vsc_pmode3_passive_n1_droop_matches_hvdc`, que construye tres variantes equivalentes del mismo sistema AC/DC con estatismo pasivo `Pdc_angle_droop` y verifica la coincidencia numérica exacta entre:
1. Enlace clásico `HvdcLine` (`type_0_free`),
2. Pareja `VSC` conectada a un nudo DC común, y
3. Pareja `VSC` conectada mediante dos nudos DC unidos por una `DcLine`.

*(Nota de entorno sobre la batería `test_ntc_strict_formulation.py`: el test nuevo `test_vsc_pmode3_passive_n1_droop_matches_hvdc` y los 14 tests estrictos existentes pasan; persisten 3 fallos preexistentes en upstream no relacionados con estos cambios: `test_issue_372_5`, `test_ntc_vsc_contingencies` y `test_issue_372_congested_thermal_slacks`.)*

---

## 6. Referencia de Implementación y Conclusión

La implementación completa de referencia se encuentra publicada en la rama **`ntc-finalist-6x`** del repositorio `ramferan/VeraGrid`:

| Commit | Alcance Principal |
|---|---|
| `9bafe2e37` .. `fe71818fe` | Signo PST $+\tau$, cota libre $\Delta$, reintento HiGHS sin presolve, parser PST (`Pt`) y máscara 4.8 |
| `d82fc9c93` | Filtrado paritario $(m, c)$ en screening *lazy* y resolución directa de HiGHS vía fichero `.lp` |
| `e121b55f7` / `bddf34b89` | Contingencias `PowerPercentage` (`'%'`) y derrame exacto con signo mediante `add_clipped_flow` |
| `a7ee9de73` / `88d9842f6` | GSK activo sobre $P_{\max}$, retirada de Fase 2 (*least control effort*) y eliminación de $y_s = 0{,}01$ |
| `1bfb15b8d` | Estatismo N-1 HVDC en lazo cerrado, unión $\alpha$ configurable, VSC Pmode 1 y `presolve="off"` en HiGHS |
| `f34347d23` | Re-evaluación *lazy* tras cambios de saturación N-1, escalado `sbase` en polo y propagación de `include_hvdc_exchange_sensitivity` |
| `df483d637` | Exposición de opciones NTC faltantes en la GUI y alineación de valores por defecto validados |
| `39e8713f9` | Soporte completo de estatismo pasivo N-1 en lazo cerrado para convertidores `VSC` y prueba de paridad HVDC ↔ VSC |

Con estos cambios, la formulación estricta de VeraGrid 6.x es plenamente operativa, convergente en las 12 horas de referencia con **0,00 MW de holguras**, coherente entre modelos **`HvdcLine` y `VSC`** tanto en Pmode 3 como en Pmode 1, y conforme a la normativa europea CACM / SO GL y P.O. 3.1. Quedamos a vuestra disposición para coordinar la revisión e integración de estos cambios en la rama principal de VeraGrid.
