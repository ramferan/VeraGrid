# Nota Técnica: Ajustes Metodológicos y Formulación del Módulo NTC en VeraGrid 6

**Para:** Equipo de Desarrollo de eRoots / VeraGrid  
**De:** Equipo de Red Eléctrica de España (REE)  
**Fecha:** 28 de octubre de 2026  
**Asunto:** Propuesta de alineación metodológica para el módulo NTC/TTC en VeraGrid 6.x

---

## 1. Motivación y Alcance

Desde REE valoramos muy positivamente los avances metodológicos introducidos en las versiones recientes de VeraGrid. En particular, queremos destacar la mayor precisión alcanzada en la modelización de la respuesta física de enlaces HVDC en modo Pmode3 ante disparos en la red AC (estatismo frecuencia/ángulo), el método de contingencias compactas (`CompactContingencies`) como una estructura analítica innovadora para abordar contingencias acopladas, el desarrollo del screening de contingencias *Lazy* fuera del LP y las mejoras en el recorte de flujo (`add_clipped_flow`).

Con el objetivo de converger en una implementación oficial de NTC en VeraGrid 6 que satisfaga plenamente los requisitos de cálculo de capacidad comercial de REE, hemos realizado una auditoría comparativa entre la formulación de referencia del MOU (GridCal 4.8) y las formulaciones `regular` y `strict` de VeraGrid 6.7.20.

Aunque por confidencialidad no podemos compartir directamente los ficheros de red del modelo peninsular, compartimos en esta nota la tabla comparativa de resultados obtenidos en el escenario de referencia, trasladamos los ajustes metodológicos requeridos con su fundamento operativo/regulatorio, y aportamos los resultados correspondientes sobre una red canónica pública de vuestro propio repositorio (IEEE 14).

---

## 2. Comparativa de Resultados en el Escenario de Referencia (MOU Hora 13 EF Pmode3)

A continuación resumimos los resultados observados al contrastar empíricamente las distintas formulaciones sobre el punto de operación de referencia de la interconexión España–Francia (fichero oficial `snapshots/MOU_pmode3_h13.gridcal`, dirección Exportación España → Francia, Pmode3 con estatismo angular, TRM = 400 MW):

| Formulación / Motor | TTC (MW) | NTC (MW) | Slack Térmico | Slack Área | Estado Solver | Tiempo | Artefacto JSON de Validación |
|---|---:|---:|---:|---:|:---:|---:|:---|
| **4.8 MOU Original (Referencia)** | **5.297,10** | **4.897,10** | **0,00 MW** | **0,00 MW** | `Optimal`* | 104 s | `datos/legacy_orig_mou_pmode3_h13_ef_n1.json` |
| **4.8 Mejoras (Clipped + Lazy)** | **5.297,10** | **4.897,10** | **0,00 MW** | **0,00 MW** | `Optimal`* | 111 s | `datos/legacy_improvements_mou_pmode3_h13_ef_n1_lazy.json` |
| **eRoots 6.7.20 `regular` (Correctivo defecto)** | 8.002,26 | 7.602,26 | 6.080,77 MW | Activo* | `Relaxed` | 172 s | `datos/modern_mou_pmode3_h13_ef_regular.json` |
| **eRoots 6.7.20 `regular` (Preventivo forzado)** | 8.002,26 | 7.602,26 | 4.070,37 MW | Activo* | `Relaxed` | 171 s | `datos/modern_eroots_regular_preventive_h13_ef.json` |
| **eRoots 6.7.20 `strict` (Oficial upstream)** | — | — | — | — | `Infeasible` | 38 s | `datos/modern_mou_pmode3_h13_ef_strict.json` |
| **VeraGrid (formulación preventiva estricta propuesta por REE)** | **5.171,18** | **4.771,18** | **0,00 MW** | **0,00 MW** | **`Optimal`** | **23,12 s** | `datos/veragrid_strict_preventive_clean_h13_ef.json` |
| **VeraGrid (correctivo acotado REE: SRAP / PST)** | **5.399,58** | **4.999,58** | **0,00 MW** | **0,00 MW** | **`Optimal`** | **~24 s** | Trazabilidad metodológica REE (Sección 7) |
| **VeraGrid (correctivo teórico EPC: modulación libre HVDC)** | **7.439,57** | **7.039,57** | **0,00 MW** | **0,00 MW** | **`Optimal`** | **~25 s** | Techo asintótico 8 GW VSC sin rampa |

> **Observaciones y Diagnóstico de Ingeniería:**  
> - (*) En 4.8 el flag interno reportaba `converged=False` por un casting espurio de códigos C de CBC (`OPTIMAL=0`), pero la solución física es exacta y libre de holguras.  
> - En **eRoots 6.7.20 `regular`**, tanto en modo correctivo (6.080,77 MW slack) como en preventivo forzado (4.070,37 MW slack), el TTC alcanza invariablemente **8.002,26 MW**. Esto demuestra empíricamente que la inflación desmedida a 8 GW **no depende del modo preventivo vs correctivo**, sino de la presencia de variables de holgura en el balance de área y en los límites térmicos (`Relaxed`), que permiten al optimizador «financiar» intercambios comerciales ficticios violando la capacidad admisible del transporte.  
> - En **eRoots 6.7.20 `strict` oficial**, el modelo resulta infactible debido al error de signo en desfasadores ($-\tau$) y a la cancelación algebraica del intercambio en el balance nodal.  
> - En **VeraGrid (formulación preventiva estricta propuesta por REE)**, al corregir el signo de los desfasadores ($+\tau$), imponer balancín 1:1 estricto e integrar el screening Lazy sobre restricciones duras, el cálculo converge a **5.171,18 MW** con **Slack = 0,00 MW** y estado **`Optimal`** en solo 23 s (paridad física del 97,7% respecto al MOU 4.8, explicándose los 126 MW restantes por el acoplamiento angular pasivo en bornes AC de convertidores).
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

### 3.5. Formulación Preventiva: Incorporación del Screening Lazy a la Formulación Estricta

* **Reconocimiento del desarrollo Lazy:**  
  Queremos reconocer expresamente que el motor de **admisión Lazy** (`screen_contingency_violations`), desarrollado por vosotros en `ntc_opf.py`, es una mejora metodológica excelente. Realizar el screening de contingencias en NumPy fuera del LP e inyectar al solver únicamente las restricciones de las contingencias que realmente violan límites (~128 grupos de 7.670) es la mejor práctica de la industria para optimizar problemas de gran escala.
* **Causa de los tiempos de cómputo en `strict` original:**  
  En la formulación estricta original (`ntc_opf_strict.py`), este motor Lazy no estaba integrado; el código intentaba inyectar por fuerza bruta las restricciones de todas las ramas para todas las contingencias directamente en el LP (hasta 61 millones de restricciones potenciales), saturando la memoria y dilatando la resolución durante horas.
* **Ajuste propuesto:**  
  Reutilizar vuestro motor de screening y admisión *Lazy* de `ntc_opf.py` dentro de la formulación `strict` con límites duros, aplicando el filtro de ramas monitorizadas con sensibilidad al intercambio ($\alpha \ge 5\%$). Esto aúna la rapidez del Lazy (~23 s por hora) con el rigor de la NTC comercial sin variables de holgura.
* **Observación de código a verificar (`ntc_opf.py:1839` vs `1848`):**  
  Hemos detectado adicionalmente una divergencia en el criterio de filtrado entre rutas: la ruta preventiva evalúa la sensibilidad en contingencia $\alpha_{N-1}$ (`alpha_n1`), mientras que la ruta compacta/correctiva emplea la lógica del caso base `branch_vars.monitor_logic` (sensibilidad $\alpha_N$). Convendría unificar para que ambas evalúen de forma consistente la sensibilidad post-falta $\alpha_{N-1}$.

---

### 3.6. Acoplamiento del Intercambio en Contingencias y Gestión de Holguras Térmicas

* **Mecanismo de desacoplo del intercambio en contingencias:**  
  Al evaluar las contingencias, el flujo en estado base N se evalúa como un valor numérico fijo en lugar de mantenerse como una variable ligada al escalón comercial $\Delta$. Al no figurar $\Delta$ como variable libre para reducir el flujo dentro de esa ecuación de contingencia, el optimizador no tiene margen para modular el intercambio a la baja y aliviar la congestión. La única variable disponible para evitar la infactibilidad es la holgura térmica ($s$), forzando al optimizador a pagar la penalización y activar slacks térmicos masivos.
* **Límites blandos vs límites duros (Demostración en modo preventivo):**  
  Al forzar `corrective_contingencies=False` en la versión regular de eRoots, el TTC sigue alcanzando invariablemente **8.002,26 MW** con **4.070,37 MW de holgura térmica**. Esto demuestra empíricamente que la inflación a 8 GW no depende de preventivo vs correctivo, sino de la presencia de **límites térmicos blandos con variables de holgura**. Mientras el optimizador disponga de la válvula de absorber sobrecargas térmicas mediante penalizaciones en la función objetivo, inflará el intercambio comercial al máximo en lugar de restringirlo a la capacidad física real de la red.
* **Ajuste propuesto:**  
  Formular límites térmicos estrictos (duros, sin variables de holgura) acoplados directamente a la variable de intercambio $\Delta$. Ante cualquier contingencia vinculante, el optimizador modula de forma natural el escalón comercial hasta la cota físicamente segura, determinando la verdadera NTC firme del sistema.

---

## 4. Trazabilidad y Volcado de Resultados en el Snapshot de Red

* **Fundamento del requisito de REE:**  
  Fidelidad del resultado mostrado. Cuando un operador o analista solicita volcar o inspeccionar el resultado de la optimización NTC en el visor del snapshot del modelo (display de resultados), espera visualizar de forma exacta las cifras y consignas resueltas por el optimizador, sin que el propio proceso de volcado altere los números.
* **Situación observada (`ntc_driver.py:163`):**  
  Al ejecutar `get_ntc_operating_point_as_opf_results` para preparar el objeto de resultados, el código calcula un supuesto desbalance global de la red y modifica arbitrariamente la generación de todos los generadores del modelo:
  ```python
  imbalance = float(gen_injection.sum() + battery_injection.sum() + load_injection.sum())
  balanced_gen -= gen_injection / gen_total * imbalance
  ```
  En cualquier red de transporte real existen pérdidas de transporte ($P_{loss}$), por lo que la suma de generación nunca iguala a la carga. Si un caso base requiere un rebalanceo, este debe gestionarse de forma explícita al inicio del proceso de cálculo o dejarse al criterio previo del usuario. Lo que no resulta admisible es alterar a posteriori el despacho en el instante de volcar los resultados, pues en ese momento se deja de mostrar la solución que el optimizador acaba de calcular.
* **Ajuste propuesto:**  
  1. Eliminar la redistribución global de desbalance en el volcado de resultados. El resultado presentado en el snapshot/visor debe reflejar con total fidelidad el despacho resuelto, aplicando los incrementos de potencia $\Delta P$ **exclusivamente sobre las áreas participantes en el intercambio**.
  2. Mantener una clara separación entre trasladar la solución al visor/display para su inspección visual interactiva frente a sobreescribir los objetos persistentes del circuito (.gridcal), acción que en todo caso debe requerir confirmación explícita del usuario para no modificar su modelo de partida de forma desapercibida.

---

## 5. Optimización en Series Temporales: Reutilización de Matrices Topológicas (PTDF y LODF)

* **Fundamento del requisito de REE:**  
  Eficiencia computacional en estudios anuales (8.760 horas) y series temporales. En la operativa de planificación y mercado, la inmensa mayoría de las horas de un estudio comparten exactamente la misma topología de red (mismo estado de líneas e interruptores en servicio).
* **Diagnóstico de tiempos de CPU:**  
  Comprobar y compilar el circuito numérico básico (`numerical_circuit`) toma apenas ~1 s por caso, lo cual es perfectamente asumible. Sin embargo, compilar y factorizar la matriz de admitancias para recalcular la matriz PTDF completa y los factores de distribución LODF de las 7.670 contingencias en cada una de las 8.760 horas consume entre 15 y 20 s adicionales por paso. Esto supone un desperdicio de decenas de horas de cómputo innecesario a lo largo de un estudio anual.
* **Propuesta implementada por REE (Reutilización por firma topológica):**  
  Sin requerir un rediseño mayor de la estructura interna de inyecciones del motor, la solución pragmática y altamente eficiente consiste en **calcular y almacenar en caché la matriz PTDF y los LODFs asociados a la topología**:
  1. Se evalúa una firma topológica instantánea (*hash* del estado de ramas e interruptores).
  2. Si la topología no ha variado respecto a la hora anterior, se reutilizan directamente las matrices PTDF y LODF ya calculadas, evitando recalcularlas en cada paso.
  3. En cada hora únicamente se actualiza el vector de inyecciones horarias y se resuelve el LP con el screening Lazy.
  
  Con este sencillo ajuste, el tiempo de preparación previa por hora cae de ~20 s a fracciones de segundo, haciendo perfectamente viable la resolución de series temporales completas en plazos operativos.

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
| **VeraGrid (formulación preventiva estricta propuesta por REE)** | 150,26 MW | **96,26 MW** | **0,00 MW** | **0,00 MW** | ~2 s | Balancín 1:1 estricto, signo $+\tau$ corregido y flujo físico natural. |

#### Diagnóstico de flujos de enlace en el caso IEEE 14:
1. **Flujo AC base de partida:**  
   En el flujo de cargas AC, el desfasador de la rama 8 (`branch 8`, $\tau = +0.0593$ rad / $3,4^\circ$, límite 50 MW) transporta un flujo base positivo de **$+10,65$ MW** hacia el nudo 9.
2. **Impacto del signo del desfasador bajo N-1:**
   * **Con el signo actual de VeraGrid ($-\tau$):**  
     El flujo por la rama 8 desciende erróneamente a **$+0,68$ MW** (un freno de $-3,45$ MW opuesto al adelanto angular físico). Simultáneamente, la rama 7 (`branch 7`, con $\tau = -0.05$ rad y límite 20 MW) invierte su signo a $-(-0.05) = +0.05$ rad, forzando potencia espuria hasta **saturar al 100% su capacidad (20,00 MW)**. Esta saturación artificial estrangula el TTC bajo N-1 de 96,26 MW a 78,07 MW (-18,19 MW de capacidad perdida ficticiamente).
   * **Con el signo corregido ($+\tau$):**  
     El flujo por la rama 8 pasa a **$+4,13$ MW**, coherente con el empuje angular positivo, la rama 7 no se satura artificialmente y la capacidad N-1 alcanza de forma natural los **96,26 MW**.

---

### 6.2. Caso Canónico IEEE 14 Congestionado: Demostración Empírica de Absorción de Sobrecargas en Slacks

Para disponer de un banco de pruebas mínimo, abierto y reproducible al 100% que evidencie la diferencia conceptual entre ambas formulaciones sin requerir datos confidenciales de la red peninsular, hemos adaptado la red canónica `IEEE14 - ntc areas_voltages_hvdc_shifter_l10free.gridcal` introduciendo una congestión determinante bajo contingencia N-1:

1. **Diseño de la red de prueba (cuello de botella N-1 inducido):**  
   - Se mantiene la topología de 2 áreas con enlace HVDC y transformador desfasador (PST).
   - Se ajusta la capacidad máxima (*rating*) de la rama transfronteriza `transformers2w[11]` (Trafo 11 / Branch 7) a **4,0 MW**.
   - En estado base N, la rama opera holgada (< 4,0 MW).
   - Ante la contingencia de una línea paralela, el flujo desviado supera ampliamente la capacidad admisible de la rama.

2. **Resultados empíricos obtenidos:**  
   Al ejecutar ambas formulaciones sobre este caso canónico, se observan los siguientes comportamientos:

| Métrica / Parámetro de Simulación | Formulación `regular` (eRoots 6.7.20) | VeraGrid (formulación preventiva estricta propuesta por REE) | Impacto Físico / Diagnóstico |
|---|---:|---:|---|
| **Capacidad Comercial TTC (MW)** | **17,05 MW** | **10,49 MW** | **Inflación del +62,5%** en regular (+6,56 MW ficticios). |
| **Holgura Térmica Total (Slack)** | **12,00 MW** | **0,00 MW** | Regular absorbe la sobrecarga en variables de holgura. |
| **Holgura Térmica en Contingencia N-1** | **12,00 MW** | **0,00 MW** | El exceso completo se concentra en la rama crítica (Trafo 11). |
| **Flujo Real Post-Contingencia en Rama Crítica** | **16,00 MW** | **4,00 MW** | En regular, la rama queda cargada al **400% de su capacidad nominal**. |
| **Estado de Solución del Solver** | `Relaxed` (*Optimal with relaxed limits*) | **`Optimal`** | Estricto resuelve con optimalidad física pura y balance 1:1. |
| **Desbalance Nodal Residual ($\sum \Delta P$)** | $6,9 \times 10^{-16}$ MW | $6,9 \times 10^{-16}$ MW | Ambos conservan balance algebraico de inyecciones. |
| **Tiempo de Ejecución** | ~ 0,03 s | ~ 1,11 s | Ejecución instantánea en ambas variantes. |
| **Artefacto JSON de Salida** | `herramientas/ieee14_congested_regular.json` | `herramientas/ieee14_congested_strict.json` | Trazabilidad numérica completa disponible. |

3. **Disponibilidad del script y artefactos de prueba:**  
   Para facilitar la consulta directa y contrastar los resultados de forma inmediata, hemos dejado disponible el script Python autocontenido `reproduce_slack_ieee.py` junto con los ficheros JSON correspondientes (`ieee14_congested_regular.json` e `ieee14_congested_strict.json`) en el repositorio (`herramientas/` y `datos/`). El script toma directamente el fichero canónico de red ya existente en vuestra suite de pruebas (`src/tests/data/grids/IEEE14 - ntc areas_voltages_hvdc_shifter_l10free.gridcal`), por lo que no requiere descargas adicionales.

4. **Diagnóstico Técnico: Demostración de Límites Blandos vs Límites Duros:**  
   - Este caso mínimo de 14 nudos reproduce exactamente el comportamiento observado en la red peninsular sin requerir datos confidenciales:
     1. En estado base N, la red opera con holgura (flujo de 3,2 MW frente a 4,0 MW de límite).
     2. Ante la contingencia N-1, se induce una congestión severa que satura la rama al 400% (16,0 MW).
   - **En la formulación `regular` (límites blandos):** Al entrar el flujo base como valor numérico fijo y permitir holguras en la restricción, el solver no puede reducir el intercambio comercial y absorbe toda la sobrecarga en variables de holgura ($s = 12,00$ MW). Como resultado, la capacidad se infla artificialmente a **17,05 MW** bajo un estado `Relaxed`.
   - **En la formulación estricta propuesta por REE (límites duros):** Al estar el límite térmico directamente vinculado a la variable de intercambio $\Delta$ sin holguras, el optimizador modula la transferencia de forma segura hasta **10,49 MW** con **Slack = 0,00 MW** y estado **`Optimal`**.
   - Esto demuestra de manera concluyente que la aparición de holguras térmicas en VeraGrid no se debe a un mal acondicionamiento numérico de la red peninsular, sino al uso de límites blandos y al desacoplo del intercambio en la evaluación de contingencias. Para la operación real del sistema eléctrico, una capacidad calculada violando límites térmicos es inoperable.

Por todo ello, proponemos consolidar en el núcleo oficial de VeraGrid la formulación de NTC con límites duros y balancín 1:1, combinándola con el motor de screening *Lazy* para garantizar rapidez computacional y máxima seguridad operativa.


---

## 7. Próximos Pasos y Colaboración

Desde el equipo de REE ponemos a vuestra disposición el código con estos ajustes consolidados y los scripts de prueba para su revisión conjunta y posterior integración en el núcleo oficial de VeraGrid.
