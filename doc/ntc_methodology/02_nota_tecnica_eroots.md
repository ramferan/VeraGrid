# Nota Técnica: Correcciones y Cierre del Módulo NTC en VeraGrid 6

**Para:** Equipo de Desarrollo de eRoots / VeraGrid  
**De:** Equipo de Red Eléctrica de España (REE)  
**Fecha:** 20 de noviembre de 2026  
**Asunto:** Propuesta consolidada de correcciones, alineación metodológica y validación final del módulo NTC/TTC en VeraGrid 6.x (`ntc-finalist-6x`)

---

## 1. Motivación y Alcance

Desde REE valoramos muy positivamente los avances arquitectónicos y metodológicos incorporados en VeraGrid 6.x, en particular el modelado físico de la respuesta pasiva de enlaces HVDC/VSC en modo *angle droop* (`Pmode 3`), el cribado *Lazy* vectorizado fuera del LP, la formulación compacta de contingencias (`CompactContingencies`), el recorte exacto de flujo (`add_clipped_flow`) y la infraestructura multiterminal para convertidores `VSC`.

Durante nuestras campañas de pruebas sobre escenarios representativos de interconexión ($\text{TRM} = 400\text{ MW}$, con enlaces en consigna modulable `Pmode 1` y en estatismo angular pasivo `Pmode 3`) y sobre las redes públicas del repositorio, identificamos un conjunto acotado de incidencias entre las dos formulaciones disponibles (`regular` y `strict`), los factores lineales de continua, la librería externa de optimización y los controles expuestos en la interfaz gráfica (GUI).

El propósito de esta nota es compartir de forma clara y reproducible **las correcciones técnicas ya implementadas en la rama propuesta `ntc-finalist-6x`** (§2), presentar después **los resultados de validación** obtenidos tanto en la hora de referencia como en la batería completa de 12 horas estacionales comparando modelos `HvdcLine` y `VSC` (§3), dejar constancia de un punto pendiente menor en el volcado de resultados (§4) y documentar su reproducibilidad completa sobre redes públicas y pruebas automatizadas (§5).

---

## 2. Correcciones Técnicas Propuestas en `ntc-finalist-6x`

Las mejoras recogidas a continuación parten directamente de la base actual de VeraGrid 6.x, respetando sus convenciones físicas (incluida la definición de desfasadores PST $f_m = b_k(\theta_{\text{from}} - \theta_{\text{to}} - \tau_m)$, idéntica al flujo AC, al flujo DC lineal y a `LinearAnalysis`) y buscando en todo momento mantener limpio el diseño del modelo de datos y de la interfaz de usuario.

### 2.1. Integración del Screening *Lazy* Matricial en la Formulación Estricta
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`run_linear_ntc_opf_strict`) y `ntc_opf.py` (`screen_contingency_violations`).
* **Problema:** En la formulación `strict`, se intentaban incorporar de golpe todas las restricciones rama × contingencia al problema lineal (decenas de millones de filas en redes reales de gran tamaño), provocando agotamiento de memoria y bloqueos de ejecución.
* **Corrección aplicada:** Integración del cribado *Lazy* vectorizado en NumPy (`screen_contingency_violations`), añadiendo al LP únicamente las parejas rama-contingencia activas o violadas en cada iteración hasta certificar ausencia total de sobrecargas en todo el conjunto de contingencias.

### 2.2. Liberación de la Cota Inferior del Intercambio en Rondas *Lazy*
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` y `ntc_opf.py`.
* **Problema:** En cada iteración *lazy* se re-imponía la cota `lb = 0` sobre la variable de intercambio $\Delta P$. En horas importadoras o en puntos base ya congestionados donde el sentido óptimo requiere reducir el flujo ($\Delta P < 0$), esa restricción artificial volvía infactible el problema.
* **Corrección aplicada:** Se fija `transfer_lb = -INFINITY`, permitiendo que el dominio comercial admita tanto exportación positiva como reducción/importación negativa dentro de los límites térmicos de la red.

### 2.3. Balance Comercial Bilateral Estricto 1:1 sin Holguras de Área ni Térmicas
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`structural_area_balance=True`).
* **Problema:** La presencia de variables de holgura térmica ($s \ge 0$) o de área permitía absorber sobrecargas reales a cambio de una penalización numérica y desvirtuaba el término $\Delta_1$ en la función objetivo cuando existía vinculación algebraica con las holguras de área.
* **Corrección aplicada:** Desplazamiento bilateral simétrico puro ($\Delta P_{A_1} = -\Delta P_{A_2} = \Delta P$) sin variables de holgura térmica ni de área en la formulación estricta.

### 2.4. Parser de Carga de Red (`pack_unpack.py`)
* **Ubicación:** `src/VeraGridEngine/IO/veragrid/pack_unpack.py`.
* **Problema:** Al cargar ficheros `.gridcal`: (a) `tap_phase_control_mode_prof` se inicializaba por defecto a `Fixed`, anulando el modo de control `Pt` de los transformadores desfasadores; y (b) el parser de contingencias instanciaba elementos aunque la columna `contingency_enabled` valiese `0`.
* **Corrección aplicada:** Preservación de `prof[0] = Pt` en el perfil de control de fase durante la importación y filtrado explícito `enabled != 0` al construir los grupos de contingencia.

### 2.5. Robustez de Tipos en el Enum `AvailableTransferMode`
* **Ubicación:** `src/VeraGridEngine/Simulations/ATC/available_transfer_capacity_driver.py` (`compute_dP`).
* **Problema:** `compute_dP` comparaba el modo de transferencia mediante enteros (`mode == 1`), mientras que VeraGrid 6 utiliza enumerados con valor textual (`AvailableTransferMode.InstalledPower`). Las llamadas con el enumerado caían silenciosamente en la rama por defecto ($P_0$).
* **Corrección aplicada:** Soporte directo y uniforme de los miembros `AvailableTransferMode` dentro de `compute_dP`.

### 2.6. Contingencias Parciales de Polo en `HvdcLine` y Disparo de Cables `DcLine` en `VSC`
* **Ubicación:** `src/VeraGridEngine/Simulations/LinearFactors/linear_analysis.py` (`LinearMultiContingency.pole_spill`, `LinearMultiContingencies.compute`) y `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`compute_pole_spill_expressions`).
* **Contexto y problema:**
  1. Para representar el disparo de un polo en un enlace `HvdcLine` bipolar, desde REE habíamos probado inicialmente añadir un parámetro propio `n_poles` en el código; sin embargo, ese planteamiento contaminaba tanto el dispositivo `HvdcLine` como la interfaz gráfica. En lugar de `n_poles`, en VeraGrid basta reutilizar la operación estándar `ContingencyOperationTypes.PowerPercentage` (`prop = '%'`, por ejemplo `value = 50.0` para modelar la pérdida del 50 % de capacidad nominal), manteniendo intacto el objeto `HvdcLine` y la GUI.
  2. Cuando el enlace se modela de forma física detallada mediante convertidores **`VSC`** y un cable `DcLine` por polo entre barras de continua, la contingencia de polo **no** se aplica con `PowerPercentage` sobre el convertidor `VSC`, sino disparando directamente el cable `DcLine` afectado (`prop = 'active'`, `value = 0.0`). Hasta ahora faltaba la reducción automática de esa contingencia en subredes DC de 2 terminales para recalcular la capacidad superviviente sobre los cables paralelos sanos y propagar el derrame AC equivalente.
* **Corrección aplicada:**
  1. Soporte nativo de `PowerPercentage` (`frac = value / 100.0`) para `HvdcLine` y `VSC`, más traducción automática de disparos de `DcLine` (`active=0`) en corredores `VSC` de 2 terminales: el motor calcula la nueva capacidad superviviente $R_{\text{surv}}$ del corredor con los cables restantes y convierte cualquier excedente pre-falta en el dipolo AC equilibrado entre los nudos de los convertidores.
  2. Expresión canónica compartida de derrame con signo (`pole_spill`):
     $$\Delta f_{\text{spill}} = \operatorname{sign}(f) \cdot \max\left(0,\; |f| - R_{\text{surv}}\right)$$
     formulada exactamente dentro del LP (`compute_pole_spill_expressions`) mediante `add_clipped_flow` (`spill = flow - carried`, con `carried` acotado en $[-R_{\text{surv}}, +R_{\text{surv}}]$) y las cotas propias de cada enlace.

### 2.7. Filtrado Paritario $(m, c)$ en Screening *Lazy* y Re-evaluación de Saturación N-1
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf.py` (`screen_contingency_violations`, `get_contingency_monitorable_branches`) y `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py`.
* **Problema:**
  1. Anteriormente, `get_contingency_monitorable_branches` evaluaba una envolvente global por rama ($\alpha_{\max}[m] = \max_c |\alpha_{N-1}[m, c]|$). Si una rama interna lejana tenía sensibilidad en *alguna* contingencia remota del sistema, quedaba marcada como supervisable bajo *todas* las demás contingencias $c$, introduciendo restricciones falsas ante contingencias que apenas modificaban su flujo ($\operatorname{MLODF} \approx 0$) o que no tenían impacto de intercambio sobre ella.
  2. Cuando los enlaces HVDC/VSC en estatismo cambiaban su estado de saturación entre una ronda *lazy* y la siguiente, las restricciones $(m, c)$ añadidas en rondas anteriores con otro régimen de saturación podían quedar desactualizadas si no se re-evaluaban con un identificador nuevo por ronda.
* **Corrección aplicada:**
  1. Criterio paritario por cada par rama-contingencia $(m, c)$: una restricción entra únicamente si combina impacto físico real de la contingencia ($|\operatorname{MLODF}_{m,k}| > 0{,}05$ para algún $k \in c$), sensibilidad al intercambio ($|\alpha_m^N| > 0{,}05$ o $\max_{k \in c} |\alpha_{N-1}[m,k]| > 0{,}05$) y superación efectiva del límite térmico post-contingencia ($|f_m^{(c)}| > \text{Rate}_m^{(c)}$).
  2. Re-evaluación automática de pares ya incorporados (`recheck_already`) cada vez que cambia el conjunto de enlaces saturados en N-1, generando nombres únicos por ronda (`_r{lazy_round}`) (`f34347d23`).

### 2.8. Bug de `PuLP` con HiGHS y Resolución Vía Fichero `.lp` (`presolve="off"`)
* **Ubicación:** `src/VeraGridEngine/Utils/MIP/pulp_interface.py` (`PulpLpModel.solve`).
* **Problema (externo a VeraGrid):** Se detectó un defecto propio de la librería de terceros **`PuLP`** en su puente en memoria hacia HiGHS (`HiGHS_CMD` / `highspy` vía API de `PuLP`): en ciertos vértices degenerados o modelos mixtos, dicho puente omitía las marcas de integralidad sobre variables binarias (`LpBinary`) y provocaba falsos óptimos o falsas infactibilidades en el *presolve* y *postsolve* interno. Es importante destacar que el error reside en `PuLP`, no en la lógica de VeraGrid.
* **Corrección aplicada:** Exportación directa del modelo desde `PuLP` mediante `model.writeLP(lp_path)` y resolución directa a través del lector nativo C++ `highspy.Highs().readModel(lp_path)` con `presolve="off"` y reintento robusto, conservando el 100 % de las variables binarias y eliminando por completo las falsas infactibilidades.

### 2.9. Estatismo Pasivo N-1 en Lazo Cerrado (`Pmode 3`)
* **Ubicación:** `src/VeraGridEngine/Simulations/LinearFactors/linear_analysis.py` (`LinearMultiContingency.compute_hvdc_droop_addon`, `compute_hvdc_droop_lp_addon`) y `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py`.
* **Problema:** En `strict`, la respuesta pasiva de los enlaces en modo *angle droop* ante contingencias AC (`droop_addon`) no llegaba a acoplarse en el LP. Además, si esa respuesta se calcula en **lazo abierto** (tomando directamente el salto angular post-falta sin realimentación), se sobrestima el alivio de las líneas AC: en la realidad física, al reaccionar el enlace DC descargando la frontera AC, la diferencia angular se frena y el propio enlace reduce su excursión respecto al cálculo sin realimentación. A su vez, esa reacción HVDC redistribuye flujos por las líneas supervivientes usando la matriz lineal **post-contingencia** (`cDF`), y no únicamente el PTDF intacto del estado N.
* **Corrección aplicada:** Formulación acoplada en **lazo cerrado** mediante la impedancia de Thévenin post-contingencia $Z_{\text{th}}^{(c)}$ entre los terminales del enlace —incorporando tanto la realimentación eléctrica como la proyección exacta de las inyecciones DC sobre las líneas AC tras la falta (`cDF`)— tanto en el cribado vectorizado *lazy* como en las expresiones lineales del LP estricto (`compute_hvdc_droop_lp_addon`).

### 2.10. Integración Completa de Convertidores `VSC` (`Pmode 1` y `Pmode 3`)
* **Ubicación:** `src/VeraGridEngine/Simulations/LinearFactors/linear_analysis.py` (`LinearAnalysis`, `LinearMultiContingencies.compute`), `ntc_opf_strict.py` y `ntc_opf.py`.
* **Problema:** Aunque la estructura general de convertidores `VSC` ya estaba presente, su comportamiento presentaba tres carencias prácticas al ejecutar NTC:
  1. En `Pmode 1`, el módulo buscaba el atributo `grid.vsc_converters` en lugar de `grid.vsc_devices` y faltaba cablear su aportación al flujo neto de frontera (`vsc_vars.inter_space_vsc`).
  2. Al montar la matriz de susceptancias $B_{\text{bus}}$ incluyendo nudos de continua aislados de la malla AC, aparecían componentes singulares sin referencia eléctrica que rompían los factores lineales de `Pmode 3`.
  3. Faltaba calcular la capacidad efectiva del corredor completo (`VSC` y cables `DcLine` paralelos en cada subred DC de 2 terminales) y acoplar conjuntamente `VSC` (`Pdc_angle_droop`) + `HvdcLine` en el lazo cerrado N-1 (evitando además aplicar *droop* AC cuando la contingencia dispara una rama de continua).
* **Corrección aplicada:**
  1. Corrección integral de `VSC` en `Pmode 1` (`grid.vsc_devices` y contabilidad de frontera `inter_space_vsc`).
  2. Construcción de $B_{\text{bus}}$ exclusivamente sobre las ramas AC pasivas en servicio con una referencia angular por componente conexa AC (eliminando singularidades de buses DC).
  3. Retorno exacto de corriente del corredor DC sobre los convertidores reguladores `Vm_dc`, acotación de cada corredor de 2 terminales al mínimo físico entre convertidores y suma de cables `DcLine` paralelos supervivientes, y resolución conjunta `VSC + HvdcLine` dentro de `LinearMultiContingencies.compute` (con paridad exacta frente a `HvdcLine`).

### 2.11. Supresión de la Segunda Etapa (*Least Control Effort*)
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`run_linear_ntc_opf_strict`) y `ntc_opf.py` (`solve_least_control_effort`, `add_base_control_effort`).
* **Problema:** Aunque la segunda etapa (`solve_least_control_effort`) fija la variable de intercambio en el valor obtenido en la Etapa 1 (`lb = ub = value`), en la formulación estricta introducía tres defectos técnicos concretos:
  1. **Ejecución fuera del bucle *lazy* sin re-cribado N-1:** se resolvía una única vez sobre el subconjunto parcial de contingencias activadas en la Etapa 1; al replegar tomas de desfasadores y reducir flujos HVDC/VSC hacia la malla AC paralela, podía provocar sobrecargas térmicas en contingencias que no estaban activas en la Etapa 1 y que ya no se volvían a comprobar.
  2. **Penalización hacia `0 MW` en lugar del punto base ($P_0$):** `add_base_control_effort` minimizaba $|P_{\text{hvdc}}|$ y $|P_{\text{vsc}}|$ hacia cero (en vez de su consigna inicial $P_0$), haciéndolo incluso cuando el flujo HVDC era una variable en estatismo pasivo (`Pmode 3`), lo que forzaba el vaciado artificial de los corredores de continua hacia las líneas AC.
  3. **Coste computacional y fragilidad numérica:** añadía una segunda resolución completa del MILP con una restricción de igualdad en coma flotante (`lb == ub`) propensa a infactibilidades numéricas (de ahí el *fallback* de advertencia existente), cuando el propósito del cálculo NTC es determinar el óptimo seguro de transferencia y no optimizar el coste de maniobra.
* **Corrección aplicada:** Eliminación de la llamada a `solve_least_control_effort` en la formulación estricta; el problema se resuelve en una única etapa dentro del bucle *lazy* maximizando directamente el intercambio físico neto de interconexión (AC + HVDC + VSC) con seguridad N-1 verificada sobre el conjunto completo de contingencias.

### 2.12. Clave de Reparto GSK en `InstalledPower` sobre $P_{\max}$ Activo y Despachable
* **Ubicación:** `src/VeraGridEngine/Compilers/circuit_to_data.py`, `src/VeraGridEngine/Simulations/NTC/ntc_opf_strict.py` (`get_transfer_power_scaling_per_bus`) y `ntc_driver.py`.
* **Problema:** `AvailableTransferMode.InstalledPower` acumulaba la potencia aparente nominal (`Snom`) sin filtrar generadores inactivos o no despachables. En redes con generadores donde $S_{\text{nom}} \ne P_{\max}$ o con unidades fuera de servicio, esa discrepancia alteraba los pesos nodales del LP respecto a las sensibilidades $\alpha$ evaluadas en el cribado previo.
* **Corrección aplicada:** Acumulación de `installed_p` y `bus_data.installed_power` en `circuit_to_data.py` exclusivamente sobre la potencia activa máxima $P_{\max}$ de generadores en servicio y despachables, unificando el criterio con `gen_data_t.get_pmax_per_bus() / Sbase` en el LP estricto y en el cálculo de sensibilidades.

### 2.13. Eliminación de la Susceptancia Parásita en Factores Lineales
* **Ubicación:** `src/VeraGridEngine/Simulations/LinearFactors/linear_analysis.py`.
* **Problema:** Con objeto de evitar singularidades, se inyectaba una susceptancia artificial $y_s = 0{,}01\text{ p.u.}$ en $B_{\text{bus}}$ sobre enlaces HVDC de consigna fija y convertidores VSC sin estatismo, creando caminos virtuales AC en paralelo a los enlaces de continua y perturbando la matriz PTDF.
* **Corrección aplicada:** Fijación de $y_s = 0{,}0$ en todo enlace o convertidor que no opere en estatismo angular pasivo (`88d9842f6`).

### 2.14. Nuevos Controles Necesarios en la Interfaz Gráfica (GUI)
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_options.py`, `src/VeraGrid/Gui/Main/MainWindow.ui`, `MainWindow.py`, `configuration.py` y `simulations.py`.
* **Problema:** Faltaban dos controles esenciales para poder reproducir desde pantalla los modos operativos validados: `include_hvdc_exchange_sensitivity` estaba oculto y siempre activo, y `corrective_contingencies` carecía de selector visible.
* **Corrección aplicada:** Exposición explícita en la GUI de ambos selectores:
  * `include_hvdc_exchange_sensitivity_checkBox`: permite activar o desactivar la inclusión de sensibilidades de despacho HVDC en el criterio de monitorización (aplicando la unión $\max(|\alpha_{\text{AC}}|, |\alpha_{\text{ACDC}}|)$ en el LP estricto cuando el usuario lo marca).
  * `corrective_ntc_contingencies_checkBox`: permite conmutar explícitamente entre cálculo preventivo puro y redespacho correctivo post-contingencia.

### 2.15. Configuración por Defecto Recomendada en Opciones y GUI
* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_options.py`, `ntc_ts_driver.py` y módulos GUI (`df483d637`).
* **Problema:** Los valores de partida en la interfaz arrancaban con la formulación estricta y las contingencias desmarcadas, y con umbrales de dispersión (`0,05`) distintos a los empleados en la validación (`0,001`).
* **Corrección aplicada:** Alineación uniforme de los valores por defecto tanto en `OptimalNetTransferCapacityOptions` como en la GUI:
  * `strict_formulation = True`
  * `consider_contingencies = True`
  * `corrective_contingencies = False`
  * `include_hvdc_exchange_sensitivity = False`
  * `ptdf_threshold = 0,001` y `lodf_threshold = 0,001`
* **Aviso práctico de actualización:** VeraGrid almacena las últimas preferencias de usuario en `~/.VeraGrid/gui_config.json` (en Windows `C:\Users\<usuario>\.VeraGrid\gui_config.json`). Tras actualizar desde una instalación previa, conviene eliminar o restablecer dicho archivo para que la GUI adopte automáticamente los valores por defecto validados.

---

## 3. Resultados de Validación

Una vez aplicadas todas las correcciones descritas en el apartado anterior sobre `ntc-finalist-6x`, se ejecutó la validación sistemática sobre el modelo de referencia España–Francia ($\text{TRM} = 400\text{ MW}$, sentido exportador $\text{ES} \to \text{FR}$) con los enlaces HVDC modelados tanto en su forma clásica (`HvdcLine`) como mediante convertidores (`VSC` + nudos DC + cables `DcLine`).

### 3.1. Diagnóstico y Paridad en la Hora Canónica de Referencia (Hora 13, Modo `Pmode 3`)

Antes de observar las cifras de la tabla, conviene distinguir las dos bases de código comparadas:
* **Rama oficial *upstream* (`VeraGrid 6.7.20`):** código base actual del repositorio principal de VeraGrid sin las correcciones de esta propuesta.
* **Rama propuesta (`ntc-finalist-6x`):** código con las correcciones descritas en la §2.

En esta prueba puntual de diagnóstico todos los enlaces regulados operan en **modo `Pmode 3`** (*angle droop* pasivo, `corrective_contingencies=False`, `include_hvdc_exchange_sensitivity=False`, `skip_generation_limits=True`):

| Motor / Rama y Formulación (`Pmode 3`) | TTC (MW) | NTC (MW) | Slack Térmico | Slack Área | Carga Máx. | Diagnóstico Técnico |
|---|---:|---:|---:|---:|---:|---|
| **VeraGrid 6.7.20 *upstream* (`regular`)** | 8.002,26 | 7.602,26 | 6.080,77 MW | Activo | > 100 % | Inválido: absorbe sobrecargas post-falta en variables *slack*. |
| **VeraGrid 6.7.20 *upstream* (`strict`)** | — | — | — | — | — | Infactible sin *lazy screening*, cota $\Delta \ge 0$ ni puente `.lp`. |
| **`strict` sin droop N-1 (`HvdcLine`)** | 5.298,61 | 4.898,61 | 0,00 MW | 0,00 MW | 193,27 %* | Sin realimentación pasiva Pmode 3 en N-1 (*). |
| **`ntc-finalist-6x` (`HvdcLine`, polo `%`)** | **5.726,88** | **5.326,88** | **0,00 MW** | **0,00 MW** | **100,00 %** | **Validado:** lazo cerrado $Z_{\text{th}}^{(c)}$ + `cDF`, polo `%`, límites duros. |
| **`ntc-finalist-6x` (`VSC`, cable único)** | **5.726,88** | **5.326,88** | **0,00 MW** | **0,00 MW** | **100,00 %** | **Paridad exacta 0,00 MW** frente a `HvdcLine`. |
| **`ntc-finalist-6x` (`VSC`, 2 `DcLine` paralelas)** | **5.726,88** | **5.326,88** | **0,00 MW** | **0,00 MW** | **100,00 %** | **Paridad exacta 0,00 MW** disparando un cable (`active=0`). |

> \* *Nota:* Sin el acoplamiento del estatismo pasivo N-1 dentro del LP, al reevaluar posteriormente la respuesta real del enlace frente a una contingencia de circuito de 400 kV aparece una sobrecarga post-falta del `193,27 %` sobre una línea de interconexión. Al acoplar el estatismo en lazo cerrado y con proyección exacta `cDF` dentro del propio optimizador, la carga máxima converge exactamente al **`100,00 %`** en N y N-1.

---

### 3.2. Validación Completa en 12 Horas Canónicas (`HvdcLine` vs `VSC`, `Pmode 3` y `Pmode 1`)

Esta batería verifica tres propiedades conjuntas en 12 horas estacionales representativas del modelo de referencia (`[13, 22, 29, 66, 98, 144, 641, 1813, 3684, 3967, 8056, 8280]`): (1) convergencia óptima y estable del LP sin fallos numéricos en escenarios exportadores e importadores, (2) cumplimiento estricto de todos los límites térmicos en N y N-1 sin variables de holgura ni sobrecargas por encima del 100 % del `rate`, y (3) equivalencia numérica entre la representación clásica `HvdcLine` y la representación detallada mediante convertidores `VSC` + nudos DC + `DcLine` bajo exactamente las mismas opciones (`strict_formulation=True`, `consider_contingencies=True`, `corrective_contingencies=False`, `include_hvdc_exchange_sensitivity=False`, `skip_generation_limits=True`, $\text{TRM}=400\text{ MW}$).

#### Tabla 3.2.a — Validación en Modo `Pmode 3` (Estatismo Angular Pasivo)

| Hora | Ref. base (sin droop N-1) (MW) | `HvdcLine` `ntc-finalist-6x` (MW) | `VSC` `ntc-finalist-6x` (MW) | Dif. `VSC − HVDC` |
|---:|---:|---:|---:|---:|
| **13** | 5.297,0 | **5.726,88** | **5.726,88** | **0,00 MW** |
| **22** | 6.515,0 | **6.501,15** | **6.501,15** | **0,00 MW** |
| **29** | 8.853,0 | **8.901,49** | **8.901,49** | **0,00 MW** |
| **66** | 6.699,0 | **6.734,86** | **6.734,86** | **0,00 MW** |
| **98** | 8.624,0 | **8.670,33** | **8.670,33** | **0,00 MW** |
| **144** | 9.010,0 | **9.042,30** | **9.042,30** | **0,00 MW** |
| **641** | 5.199,0 | **5.224,90** | **5.224,90** | **0,00 MW** |
| **1813** | 3.759,0 | **3.792,90** | **3.792,90** | **0,00 MW** |
| **3684** | −3.055,0 | **−3.043,40** | **−3.043,40** | **0,00 MW** |
| **3967** | 3.433,0 | **3.447,73** | **3.447,73** | **0,00 MW** |
| **8056** | 9.033,0 | **9.108,12** | **9.108,12** | **0,00 MW** |
| **8280** | 8.994,0 | **9.063,07** | **9.063,07** | **0,00 MW** |

#### Tabla 3.2.b — Validación en Modo `Pmode 1` (Consigna Modulable)

| Hora | Ref. base (MW) | `HvdcLine` `ntc-finalist-6x` (MW) | `VSC` `ntc-finalist-6x` (MW) | Dif. `VSC − HVDC` |
|---:|---:|---:|---:|---:|
| **13** | 6.034,0 | **6.048,16** | **6.048,16** | **0,00 MW** |
| **22** | 7.562,0 | **7.562,54** | **7.562,54** | **0,00 MW** |
| **29** | 9.576,0 | **9.733,09** | **9.733,09** | **0,00 MW** |
| **66** | 7.676,0 | **7.737,88** | **7.737,88** | **0,00 MW** |
| **98** | 9.524,0 | **10.112,05** | **10.112,05** | **0,00 MW** |
| **144** | 9.369,0 | **9.396,96** | **9.396,96** | **0,00 MW** |
| **641** | 6.170,0 | **6.228,79** | **6.228,79** | **0,00 MW** |
| **1813** | 5.106,0 | **5.115,62** | **5.115,62** | **0,00 MW** |
| **3684** | −2.102,0 | **−2.102,04** | **−2.102,04** | **0,00 MW** |
| **3967** | 4.964,0 | **4.964,70** | **4.964,70** | **0,00 MW** |
| **8056** | 9.391,0 | **9.403,61** | **9.403,61** | **0,00 MW** |
| **8280** | 9.882,0 | **9.943,03** | **9.943,03** | **0,00 MW** |

* **Verificación de factibilidad y límites térmicos:** Todas las ejecuciones (`12/12` horas en las cuatro series anteriores: `HvdcLine Pmode 3`, `VSC Pmode 3`, `HvdcLine Pmode 1` y `VSC Pmode 1`) finalizaron en estado **`Optimal`** sin holguras térmicas ni de área (`total_slack_mw = 0,00 MW`) y con una carga máxima exactamente igual al **`100,00 %`** en todo elemento monitorizado tanto en estado base N como en post-contingencia N-1. Ninguna línea supera su límite térmico (`rate`).
* **Equivalencia exacta `HvdcLine` ↔ `VSC`:** En las 24 parejas horarias evaluadas la discrepancia máxima entre ambas modelizaciones es inferior a **`0,000001 MW`** ($\max|\Delta_{\text{Pmode 3}}| = 0{,}000000\text{ MW}$, $\max|\Delta_{\text{Pmode 1}}| = 0{,}000001\text{ MW}$). En la Hora 13 se verificó adicionalmente el modelado físico con **dos cables `DcLine` paralelos** por bipolo (`VSC` sin cota limitante en convertidor y disparo `active=0` sobre cada cable), reproduciendo idénticos `5.726,88 MW` en `Pmode 3` y `6.048,16 MW` en `Pmode 1` (diferencia `+0,000000 MW` respecto a `HvdcLine`).

---

## 4. Punto Pendiente en el Volcado de Resultados al Snapshot (`ntc_driver.py`)

* **Ubicación:** `src/VeraGridEngine/Simulations/NTC/ntc_driver.py` (`get_ntc_operating_point_as_opf_results`, líneas 160–165).
* **Observación técnica:** Al exportar el punto de operación NTC hacia resultados de Flujo de Potencia / OPF, la función calcula un desbalance base (`imbalance = gen + batt - load`) y lo redistribuye proporcionalmente sobre todos los generadores de la red antes de sumar `bus_change`:
  ```python
  imbalance = float(gen_injection.sum() + battery_injection.sum() + load_injection.sum())
  balanced_gen -= gen_injection / gen_total * imbalance
  ```
* **Recomendación de cierre:** Suprimir esta redistribución global previa sobre generadores ajenos al intercambio comercial para que el punto exportado conserve intacto el despacho base fuera de las áreas $\mathcal{A}_1$ y $\mathcal{A}_2$.

---

## 5. Verificación Reproducible sobre Redes Públicas y Batería de Tests

Todas las pruebas automatizadas y los guiones de reproducibilidad incorporados en el código utilizan **exclusivamente redes públicas del repositorio y mallas sintéticas construidas en código**, sin ninguna referencia ni dependencia hacia redes confidenciales de REE.

### 5.1. Demostración Reproducible en la Red Pública IEEE 14
Ejecutando `herramientas/reproduce_slack_ieee.py` sobre la versión final de `ntc-finalist-6x` con la red pública `IEEE14 - ntc areas_voltages_hvdc_shifter_l10free.gridcal`:
1. **Efecto de límites duros frente a holguras térmicas (rama crítica limitada a `4,0 MW`):**
   * En modo **preventivo puro** (`corrective_contingencies=False`): la formulación `regular` devuelve un intercambio inflado de **`41,06 MW`** inter-área (`dSbus = 34,49 MW`) absorbiendo **`9,84 MW`** en holguras térmicas post-contingencia; la formulación `strict` detecta con rigor la infactibilidad física ante la falta de alivio preventivo.
   * Con **redespacho correctivo HVDC** (`corrective_contingencies=True`): la formulación `strict` converge limpiamente a **`26,19 MW`** inter-área (`dSbus = 19,63 MW`) con **`0,00 MW` de slack**, balance bilateral 1:1 exacto y flujo en la rama crítica exactamente al `100,00 %` (`4,00 MW`).
2. **Coherencia con la convención estándar de desfasadores ($-\tau$):** en la misma red pública con `TapPhaseControl.Pt` en sentido $\mathcal{A}_2 \to \mathcal{A}_1$ y consigna fija HVDC, la formulación estricta converge a **`78,07 MW`** inter-área con `0,00 MW` de slack, bajo la misma convención $-\tau$ del flujo AC, del flujo DC lineal y de `LinearAnalysis`.

### 5.2. Pruebas Automatizadas en `src/tests/NTC/`
En `src/tests/NTC/test_ntc_strict_formulation.py` (ejecución íntegra sobre redes públicas y sintéticas) se han añadido y validado dos pruebas específicas de equivalencia `HvdcLine` ↔ `VSC`:
* `test_vsc_pmode3_passive_n1_droop_matches_hvdc`: valida la paridad en `Pmode 3` con estatismo pasivo N-1 entre `HvdcLine`, `VSC` con nudo DC común y `VSC` con dos nudos DC y `DcLine`.
* `test_vsc_parallel_dcline_pole_outage_matches_hvdc`: valida la equivalencia entre `HvdcLine` con contingencia `PowerPercentage` (`'%'` al 50 %) y un bipolo `VSC` con dos cables `DcLine` paralelos en servicio y disparo `active=0` por cable, tanto en `Pmode 3` como en `Pmode 1`.

**Resultados ejecutados sobre la versión final:**
* `pytest -q src/tests/NTC/test_ntc_strict_formulation.py src/tests/NTC/test_ntc_preflight.py`: **`27 passed, 0 failed`**.
* `pytest -q src/tests/NTC/test_ntc.py -k "372"`: **`5 passed, 0 failed`**.
* Los tres tests que anteriormente fallaban en la batería pública (`test_issue_372_5`, `test_ntc_vsc_contingencies` y `test_issue_372_congested_thermal_slacks`) han quedado igualmente corregidos y verificados en verde sin recurrir a ningún caso privado.

---

## 6. Referencia de Implementación y Conclusión

La implementación completa propuesta para revisión se encuentra consolidada en la rama **`ntc-finalist-6x`** del repositorio `ramferan/VeraGrid`:

| Commit | Alcance Principal |
|---|---|
| `9bafe2e37` .. `fe71818fe` | Cota libre $\Delta$, reintento HiGHS sin presolve, parser PST (`Pt`) y máscara unificada |
| `d82fc9c93` | Filtrado paritario $(m, c)$ en screening *lazy* y resolución directa de HiGHS vía fichero `.lp` |
| `e121b55f7` / `bddf34b89` | Contingencias `PowerPercentage` (`'%'`) y derrame exacto con signo mediante `add_clipped_flow` |
| `a7ee9de73` / `88d9842f6` | GSK activo sobre $P_{\max}$, supresión de Fase 2 (*least control effort*) y eliminación de $y_s = 0{,}01$ |
| `1bfb15b8d` | Estatismo N-1 HVDC en lazo cerrado, unión $\alpha$ configurable, VSC `Pmode 1` y `presolve="off"` en HiGHS |
| `f34347d23` | Re-evaluación *lazy* tras cambios de saturación N-1 y propagación de `include_hvdc_exchange_sensitivity` |
| `df483d637` | Exposición de opciones NTC faltantes en la GUI y alineación de valores por defecto validados |
| `39e8713f9` | Soporte de estatismo pasivo N-1 en lazo cerrado para convertidores `VSC` y prueba de paridad HVDC ↔ VSC |
| `5ed92e261` | Proyección exacta post-contingencia `cDF` en N-1, disparo nativo de cables `DcLine` en VSC y cierre `27/27` en tests públicos |

Con estas correcciones, la formulación estricta de VeraGrid 6.x resulta plenamente operativa, converge de forma robusta en las 12 horas de prueba con **`0,00 MW` de holguras** y **carga máxima ≤ 100,00 %**, mantiene coherencia numérica completa entre **`HvdcLine` y `VSC`** tanto en `Pmode 3` como en `Pmode 1`, y queda verificada mediante pruebas automatizadas reproducibles sobre redes públicas. Quedamos a vuestra disposición para revisar e integrar estos cambios en la rama principal de VeraGrid.
