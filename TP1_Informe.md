# Trabajo Práctico N.° 1 — Análisis de Series Temporales

**Maestría en Data Science — Universidad Austral**  
**Materia:** Análisis de Series Temporales  
**Profesor:** Rodrigo Del Rosso  
**Fecha de entrega:** 16 de julio de 2025  

---

**Integrantes del grupo:**

| Apellido y Nombre | Legajo |
|---|---|
| Pessagno, Lucas Ezequiel | — |

---

## Resumen Ejecutivo

El presente trabajo analiza cuatro series temporales de distinta naturaleza —financiera, de volatilidad de mercado, climática y de transporte aéreo— con el objetivo de aplicar el marco metodológico de identificación, estimación, diagnóstico y pronóstico de modelos ARIMA/SARIMA. Las series seleccionadas son el índice S&P 500, el índice de volatilidad VIX, la concentración atmosférica de CO₂ medida en Mauna Loa y la serie clásica AirPassengers de Box & Jenkins.

Los principales hallazgos indican que las series financieras (S&P 500, VIX) no son predecibles más allá del nivel del ruido de mercado en horizontes de mediano plazo, en línea con la hipótesis de mercados eficientes. La serie de CO₂ presenta la mejor performance predictiva (R² = 0,27 en el conjunto de prueba), siendo el único caso en que el modelo ARIMA supera claramente a los benchmarks. La incorporación de variables exógenas —efecto leverage entre el S&P 500 y el VIX— produce mejoras sustanciales (ΔAIC ≈ 140 puntos), confirmando la significatividad estadística de la relación bidireccional entre ambos índices. Para las series estacionales, los modelos SARIMA(0,1,1)(0,1,1)₁₂ superan a sus contrapartes no estacionales con diferencias de AIC superiores a 240 puntos.

---

## Índice

1. [Introducción](#1-introducción)
2. [Marco Teórico](#2-marco-teórico)
3. [Descripción de los Datos](#3-descripción-de-los-datos)
4. [Análisis Visual y Transformaciones](#4-análisis-visual-y-transformaciones)
5. [Análisis de Correlación (FAS, FAC, FACP)](#5-análisis-de-correlación-fas-fac-facp)
6. [Pruebas de Raíz Unitaria](#6-pruebas-de-raíz-unitaria)
7. [Selección de Modelos ARIMA/SARIMA](#7-selección-de-modelos-arimasarima)
8. [Métricas de Performance](#8-métricas-de-performance)
9. [Comparación de Modelos](#9-comparación-de-modelos)
10. [Diagnóstico de Residuos](#10-diagnóstico-de-residuos)
11. [Pronósticos](#11-pronósticos)
12. [Modelos SARIMAX con Variables Exógenas](#12-modelos-sarimax-con-variables-exógenas)
13. [Análisis Estacional y Test HEGY](#13-análisis-estacional-y-test-hegy)
14. [Conclusiones](#14-conclusiones)
15. [Referencias](#15-referencias)
16. [Apéndice — Código](#16-apéndice--código)

---

## 1. Introducción

El análisis de series temporales constituye una de las disciplinas centrales de la ciencia de datos aplicada. Su capacidad para modelar la dependencia temporal de observaciones sucesivas lo convierte en herramienta fundamental en dominios tan diversos como las finanzas, la climatología, la epidemiología y la logística.

El presente trabajo tiene por objeto aplicar los conceptos y técnicas del análisis clásico de series temporales —en particular los modelos de la familia ARIMA/SARIMA y sus extensiones— sobre cuatro series de características contrastantes. La selección deliberada de series con propiedades distintas (presencia o ausencia de tendencia, estacionalidad, heteroscedasticidad) permite ilustrar la aplicación del ciclo completo de modelado: identificación, estimación, verificación diagnóstica y pronóstico.

El trabajo se organiza siguiendo la metodología Box-Jenkins, complementada con pruebas formales de raíz unitaria y criterios de información para la selección de modelos. El código fuente completo, implementado en Python con las bibliotecas `statsmodels`, `pandas` y `matplotlib`, se adjunta como apéndice.

---

## 2. Marco Teórico

### 2.1 Procesos Estocásticos y Estacionariedad

Una serie temporal $\{Y_t\}$ se denomina **estrictamente estacionaria** si su distribución conjunta es invariante ante desplazamientos temporales. En la práctica, se opera con la noción más débil de **estacionariedad de segundo orden** (o débil): la media $\mu = E[Y_t]$ y la covarianza $\gamma(h) = \text{Cov}(Y_t, Y_{t+h})$ son constantes e independientes de $t$.

La estacionariedad es condición necesaria para que el modelo ARIMA sea estadísticamente válido: sin ella, la inferencia asintótica sobre los coeficientes colapsa y los pronósticos divergen. Cuando una serie no es estacionaria en media, se aplica diferenciación de orden $d$; cuando presenta no estacionariedad estacional, se agrega diferenciación estacional de orden $D$.

### 2.2 Modelos ARIMA(p, d, q)

El modelo ARIMA combina un componente autorregresivo de orden $p$, $d$ diferencias para inducir estacionariedad, y un componente de media móvil de orden $q$:

$$\phi(B)(1-B)^d Y_t = \theta(B)\varepsilon_t$$

donde $B$ es el operador de rezago, $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ y $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$. El término $\varepsilon_t \sim \text{WN}(0, \sigma^2)$ representa ruido blanco.

### 2.3 Modelos SARIMA(p, d, q)(P, D, Q)_s

La extensión estacional del modelo ARIMA incorpora componentes autorregresivos y de media móvil de frecuencia estacional $s$:

$$\Phi(B^s)\phi(B)(1-B)^d(1-B^s)^D Y_t = \Theta(B^s)\theta(B)\varepsilon_t$$

donde $\Phi(B^s)$ y $\Theta(B^s)$ son polinomios en $B^s$ de órdenes $P$ y $Q$ respectivamente. Este modelo resulta apropiado cuando la serie exhibe patrones repetitivos de período $s$ (mensual: $s=12$, trimestral: $s=4$).

### 2.4 Modelo SARIMAX

El modelo SARIMA con variables exógenas (SARIMAX) extiende la especificación al incorporar regresores externos $X_t$:

$$\phi(B)(1-B)^d Y_t = \beta X_t + \theta(B)\varepsilon_t$$

Este enfoque resulta útil cuando se dispone de variables predictoras que no forman parte de la serie objetivo pero contienen información relevante sobre su dinámica.

### 2.5 Pruebas de Raíz Unitaria

**Test ADF (Augmented Dickey-Fuller):** Contrasta $H_0$: la serie tiene una raíz unitaria (I(1)), contra $H_1$: la serie es estacionaria. Se estima la regresión:

$$\Delta Y_t = \alpha + \beta t + \rho Y_{t-1} + \sum_{i=1}^k \delta_i \Delta Y_{t-i} + \varepsilon_t$$

La hipótesis nula implica $\rho = 0$. El estadístico de prueba sigue una distribución no estándar (Dickey-Fuller).

**Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin):** A diferencia del ADF, contrasta $H_0$: la serie es estacionaria, contra $H_1$: existe raíz unitaria. La utilización conjunta de ADF y KPSS permite obtener evidencia robusta: si ADF no rechaza y KPSS rechaza, la evidencia a favor de I(1) es convergente.

### 2.6 Función de Autocorrelación (FAC) y Autocorrelación Parcial (FACP)

La **FAC** mide la correlación entre $Y_t$ e $Y_{t-h}$ para cada rezago $h$, capturando el efecto directo e indirecto de todos los rezagos intermedios. La **FACP** aísla la correlación parcial entre $Y_t$ e $Y_{t-h}$ controlando por los rezagos $1, \ldots, h-1$, revelando la dependencia directa.

Los patrones teóricos sirven de guía para la identificación del orden del modelo:

| Modelo | FAC | FACP |
|---|---|---|
| AR(p) | Decaimiento exponencial/oscilatorio | Corte en el rezago p |
| MA(q) | Corte en el rezago q | Decaimiento exponencial/oscilatorio |
| ARMA(p,q) | Decaimiento exponencial/oscilatorio | Decaimiento exponencial/oscilatorio |

### 2.7 Criterios de Información

Para la selección del orden del modelo se utilizan el **criterio de Akaike (AIC)** y el **criterio bayesiano de Schwarz (BIC)**:

$$\text{AIC} = -2\ln(\hat{L}) + 2k \qquad \text{BIC} = -2\ln(\hat{L}) + k\ln(n)$$

donde $\hat{L}$ es la log-verosimilitud maximizada, $k$ el número de parámetros y $n$ el tamaño muestral. El BIC penaliza más fuertemente la complejidad, favoreciendo modelos parsimoniosos. Se selecciona el modelo con menor AIC/BIC.

### 2.8 Diagnóstico de Residuos

Un modelo adecuado debe producir residuos que se comporten como ruido blanco: ausencia de autocorrelación, homocedasticidad y, idealmente, normalidad. El **test de Ljung-Box** contrasta $H_0$: los primeros $h$ autocorrelaciones de los residuos son nulas:

$$Q(h) = n(n+2)\sum_{j=1}^h \frac{\hat{\rho}_j^2}{n-j} \sim \chi^2_{h-p-q}$$

El **test de Jarque-Bera** evalúa la normalidad de los residuos a través de asimetría y curtosis.

---

## 3. Descripción de los Datos

### 3.1 Criterios de Selección

La selección de series obedece a tres criterios complementarios: (i) representatividad de distintos regímenes dinámicos (tendencia estocástica, volatilidad agrupada, estacionalidad determinista); (ii) disponibilidad pública y reproducibilidad; (iii) relevancia analítica en el contexto de las finanzas cuantitativas y la ciencia ambiental.

La inclusión de cuatro series —superando el mínimo de tres exigido— permite además comparar metodologías en dominios diferentes y extraer conclusiones de mayor generalidad.

### 3.2 Series Seleccionadas

| Serie | Fuente | Frecuencia | Período | Observaciones |
|---|---|---|---|---|
| S&P 500 | Investing.com | Diaria | 03/03/2025 – 02/04/2026 | 274 |
| VIX (CBOE Volatility) | Investing.com | Diaria | 03/03/2025 – 02/04/2026 | 279 |
| CO₂ Mauna Loa | NOAA | Mensual | ene/2010 – feb/2026 | 194 |
| AirPassengers | Box & Jenkins (1976) | Mensual | ene/1949 – dic/1960 | 144 |

### 3.3 Descripción de Cada Serie

**S&P 500.** Índice bursátil que pondera por capitalización de mercado a las 500 empresas de mayor liquidez listadas en bolsas estadounidenses. Constituye el referente más citado del desempeño del mercado de renta variable norteamericano. El período analizado abarca una fase de alta volatilidad asociada a shocks macroeconómicos recientes (política arancelaria, tensiones geopolíticas), lo que enriquece el análisis de las propiedades estocásticas de la serie.

| Estadístico | Valor |
|---|---|
| Mínimo | 4.982,77 |
| Máximo | 6.978,60 |
| Media | 6.383,25 |

**VIX (CBOE Volatility Index).** Índice que mide la volatilidad implícita del mercado de opciones sobre el S&P 500 a 30 días, ampliamente utilizado como indicador del "miedo" del mercado. Presenta una relación negativa y asimétrica con el S&P 500, conocida como **efecto leverage**: las caídas del mercado generan aumentos de volatilidad desproporcionadamente mayores que los aumentos equivalentes.

| Estadístico | Valor |
|---|---|
| Mínimo | 81,89 |
| Máximo | 170,92 |
| Media | 103,24 |

**CO₂ Mauna Loa.** Serie de concentración atmosférica de dióxido de carbono (en partes por millón, ppm) registrada en el Observatorio de Mauna Loa (Hawái) por la NOAA. Es la serie de monitoreo continuo de gases de efecto invernadero más larga del mundo y referencia estándar del cambio climático. Presenta tendencia creciente sostenida (~2 ppm/año) y estacionalidad anual de amplitud ~7 ppm, producto del ciclo de absorción de la biosfera terrestre.

| Estadístico | Valor |
|---|---|
| Mínimo | 387,03 ppm |
| Máximo | 430,51 ppm |
| Media | 408,11 ppm |

**AirPassengers.** Serie clásica de la literatura de series temporales, introducida por Box & Jenkins (1976), que registra el número mensual de pasajeros internacionales de aerolíneas en miles. Exhibe tendencia creciente y estacionalidad multiplicativa (amplitudes crecientes), siendo el caso de referencia para la ilustración del modelo Airline —SARIMA(0,1,1)(0,1,1)₁₂.

| Estadístico | Valor |
|---|---|
| Mínimo | 104.000 pasajeros |
| Máximo | 622.000 pasajeros |
| Media | 280.300 pasajeros |

---

## 4. Análisis Visual y Transformaciones

La Figura 4.1 muestra las cuatro series en su escala original. Las diferencias en dinámica son notables: el S&P 500 y el VIX exhiben comportamiento irregular propio de series financieras, mientras que CO₂ y AirPassengers revelan patrones estructurados de tendencia y estacionalidad.

**Figura 4.1 — Series temporales en niveles**

![Series en niveles](informe_graficos/fig_01_cell06.png)

La Figura 4.2 presenta las estadísticas móviles (media y desvío estándar rodante) para cada serie, herramienta visual para detectar no estacionariedad: si la media o la varianza no son constantes en el tiempo, la serie no es estacionaria.

**Figura 4.2 — Media y desvío estándar rodantes**

![Estadísticas móviles](informe_graficos/fig_02_cell08.png)

La Figura 4.3 muestra las series luego de aplicar las transformaciones necesarias para inducir estacionariedad.

**Figura 4.3 — Series transformadas (estacionarizadas)**

![Series transformadas](informe_graficos/fig_03_cell09.png)

---

### 4.1 S&P 500

La serie en niveles exhibe una tendencia creciente pronunciada durante la primera mitad del período analizado, seguida de una corrección significativa asociada al anuncio de aranceles comerciales de la administración estadounidense. La varianza no es constante (grupos de alta volatilidad), lo que confirma la presencia de efectos ARCH típicos de series financieras.

**Transformaciones aplicadas:**
- *Log-precio* ($\ln P_t$): linealiza la tendencia exponencial y estabiliza la varianza de primer orden.
- *Log-retorno* ($r_t = \Delta \ln P_t = \ln P_t - \ln P_{t-1}$): primera diferencia del log-precio; produce una serie aproximadamente estacionaria que representa la variación porcentual continua.

**Justificación:** La diferenciación logarítmica es la transformación estándar en series financieras de precios (Tsay, 2010; Campbell, Lo & MacKinlay, 1997). Tiene fundamento en la hipótesis de caminata aleatoria del precio y preserva la interpretabilidad económica de los retornos.

### 4.2 VIX

La serie de VIX presenta dos características destacables: (i) media-reversión (tendencia a retornar hacia su nivel histórico), que sugiere potencial estacionariedad; (ii) skewness positivo con picos extremos durante episodios de stress de mercado, indicativo de distribución de cola pesada.

**Transformaciones aplicadas:** Primera diferencia ($\Delta\text{VIX}_t$) para el modelo SARIMAX, dado que la estimación conjunta con retornos del S&P 500 requiere que ambas variables compartan el mismo orden de integración.

### 4.3 CO₂ Mauna Loa

La serie original muestra tendencia lineal creciente (~2,07 ppm/año, estimado a partir del coeficiente de la constante del modelo ARIMA: 0,172 ppm/mes) y una oscilación estacional de amplitud aproximada 7 ppm. El patrón estacional refleja el ciclo de fotosíntesis del hemisferio norte: máximos en mayo (invierno boreal, menor absorción) y mínimos en septiembre (verano boreal, máxima absorción por la vegetación).

**Transformaciones aplicadas:** Primera diferencia regular ($\Delta_1$) para eliminar la tendencia; diferencia estacional de orden 12 ($\Delta_{12}$) para tratar la estacionalidad en los modelos SARIMA.

### 4.4 AirPassengers

La serie original evidencia estacionalidad multiplicativa: la amplitud de las oscilaciones estacionales crece proporcionalmente con el nivel de la serie, indicando que el componente estacional actúa en escala relativa y no absoluta.

**Transformaciones aplicadas:**
- *Logaritmo natural*: convierte la estacionalidad multiplicativa en aditiva, habilitando la aplicación directa de modelos ARIMA aditivos.
- *Primera diferencia y diferencia estacional* ($\Delta_1 \Delta_{12} \ln Y_t$): para el modelo Airline.

**Justificación de la transformación logarítmica:** La elección entre modelos aditivos y multiplicativos se fundamenta en la naturaleza del componente estacional observado gráficamente. Cuando la varianza del componente estacional crece con el nivel de la serie (patrón multiplicativo), el logaritmo es la transformación de estabilización de varianza apropiada (Hyndman & Athanasopoulos, 2021).

---

## 5. Análisis de Correlación (FAS, FAC, FACP)

Las Figuras 5.1 a 5.8 presentan, para cada serie, la FAC y FACP del nivel original y de la versión transformada (estacionarizada). La comparación entre ambas formas permite visualizar el efecto de la diferenciación sobre la estructura de autocorrelación.

**Figura 5.1 — S&P 500: FAC/FACP en niveles y log-retornos**

![FAC FACP S&P 500 niveles](informe_graficos/fig_04_cell12.png)

![FAC FACP S&P 500 retornos](informe_graficos/fig_05_cell12.png)

**Figura 5.2 — VIX: FAC/FACP en niveles y primera diferencia**

![FAC FACP VIX niveles](informe_graficos/fig_06_cell12.png)

![FAC FACP VIX diferencia](informe_graficos/fig_07_cell12.png)

**Figura 5.3 — CO₂: FAC/FACP en niveles y primera diferencia**

![FAC FACP CO2 niveles](informe_graficos/fig_08_cell12.png)

![FAC FACP CO2 diferencia](informe_graficos/fig_09_cell12.png)

**Figura 5.4 — AirPassengers: FAC/FACP en logaritmos y diferenciado**

![FAC FACP Air logaritmos](informe_graficos/fig_10_cell12.png)

![FAC FACP Air diferenciado](informe_graficos/fig_11_cell12.png)

---

### 5.1 Fundamento del Análisis Conjunto

La lectura conjunta de la función de autocorrelación simple (FAC) y la función de autocorrelación parcial (FACP) constituye la herramienta diagnóstica central de la metodología Box-Jenkins. La FAC permite identificar el horizonte de dependencia temporal total, mientras que la FACP aísla la contribución directa de cada rezago, eliminando la influencia de los intermedios.

### 5.2 S&P 500 (Log-retornos)

Los log-retornos del S&P 500 muestran una FAC con casi todas las autocorrelaciones dentro de las bandas de confianza al 95%, con una autocorrelación levemente negativa en el rezago 1 (AR(1) con coeficiente pequeño y negativo: $\hat{\phi}_1 = -0,0953$). La FACP corta en el rezago 1, lo que sugiere un proceso AR(1) de baja persistencia.

**Interpretación:** El patrón es consistente con la hipótesis de eficiencia de mercado en su forma débil: los retornos presentan dependencia estadística pequeña pero significativa (p < 0,001), lo que permite identificar un ARIMA(1,1,0). La autorregresión negativa implica una leve reversión a la media en los retornos diarios.

### 5.3 VIX

La FAC del VIX en niveles decae lentamente de forma exponencial, característica de procesos AR estacionarios con alta persistencia. La FACP muestra cortes significativos en los rezagos 1, 2 y 3, sugiriendo un proceso AR(3) o ARMA(3, q).

**Interpretación:** El decaimiento lento pero sin raíz unitaria es compatible con un proceso I(0) estacionario de alta persistencia, fenómeno conocido como "larga memoria" en series de volatilidad. Sin embargo, las pruebas formales de raíz unitaria (Sección 6) revelan evidencia mixta que requiere cautela interpretativa.

### 5.4 CO₂ Mauna Loa

La FAC de la serie en niveles decae muy lentamente y con patrón senoidal de período 12, evidencia de tendencia y estacionalidad simultáneas. Tras aplicar la diferencia regular y estacional ($\Delta_1\Delta_{12}$), la FAC colapsa rápidamente al interior de las bandas de confianza, con autocorrelaciones significativas únicamente en los rezagos 1 y 12, perfil típico del modelo Airline.

### 5.5 AirPassengers

La serie original exhibe FAC con decaimiento lento y picos en los múltiplos de 12, patrón inequívoco de tendencia con estacionalidad mensual. Luego de la diferenciación doble ($\Delta_1\Delta_{12} \ln Y_t$), la FAC presenta cortes en los rezagos 1 y 12, y la FACP decae a partir de esos mismos rezagos, consistente con el modelo MA(1)×MA(1)₁₂ = SARIMA(0,1,1)(0,1,1)₁₂.

---

## 6. Pruebas de Raíz Unitaria

### 6.1 Metodología

Se aplican de forma conjunta los tests ADF y KPSS para cada serie en su nivel original, dado que sus hipótesis nulas son complementarias. El nivel de significación adoptado es $\alpha = 0,05$.

- Si ADF rechaza y KPSS no rechaza → evidencia a favor de estacionariedad (I(0))
- Si ADF no rechaza y KPSS rechaza → evidencia convergente de raíz unitaria (I(1))
- Si ambos rechazan o ambos no rechazan → evidencia mixta; se requiere análisis adicional

### 6.2 Resultados

**Tabla 6.1 — Pruebas de raíz unitaria en niveles**

| Serie | ADF Estadístico | ADF p-valor | KPSS Estadístico | KPSS p-valor | Conclusión |
|---|:---:|:---:|:---:|:---:|---|
| S&P 500 | −0,7655 | 0,9685 | 0,4528 | 0,0100 | I(1) — evidencia convergente |
| VIX | −3,9077 | 0,0118 | 0,2482 | 0,0100 | Mixta — KPSS rechaza estacionariedad |
| CO₂ | −3,0607 | 0,1159 | 0,0444 | > 0,10 | I(1) — ADF no rechaza; KPSS no rechaza |
| AirPassengers | −2,1008 | 0,5457 | 0,0961 | > 0,10 | I(1) — ADF no rechaza; KPSS no rechaza |

### 6.3 Análisis por Serie

**S&P 500:** El estadístico ADF de −0,77 está muy lejos de los valores críticos (−2,86 al 5%), con p-valor = 0,97. El KPSS rechaza la hipótesis nula de estacionariedad al 1%. La evidencia es fuertemente convergente hacia la presencia de raíz unitaria (I(1)). Se procede con primera diferenciación ($d = 1$).

**VIX:** El ADF rechaza la raíz unitaria (p = 0,0118 < 0,05), lo que sugeriría estacionariedad. Sin embargo, el KPSS también rechaza la estacionariedad al 1%, generando evidencia contradictoria. Esta situación puede deberse a la alta persistencia de la serie (proceso cercano a la frontera I(0)/I(1)) o a quiebres estructurales en la media. En consideración de la naturaleza financiera de la serie y de las pruebas adicionales realizadas sobre los retornos del S&P 500, se opta por trabajar con la primera diferencia del VIX en el contexto del modelo SARIMAX, para garantizar la comparabilidad de escalas con los log-retornos del S&P 500.

**CO₂:** El ADF no rechaza la raíz unitaria (p = 0,116) y el KPSS tampoco rechaza la estacionariedad (p > 0,10). Este resultado aparentemente contradictorio se resuelve al considerar la especificación del test ADF con constante y tendencia: la tendencia determinista "consume" potencia del test. La inspección visual y la magnitud del coeficiente de tendencia (+2,07 ppm/año) confirman la necesidad de diferenciación ($d = 1$).

**AirPassengers:** Evidencia clara de I(1): ADF no rechaza (p = 0,55) y KPSS no rechaza la estacionariedad en la serie en logaritmos, lo que indica que la tendencia no es puramente determinista. Se aplica $d = 1$ y $D = 1$ (diferenciación estacional).

---

## 7. Selección de Modelos ARIMA/SARIMA

### 7.1 Procedimiento de Búsqueda

La selección del orden del modelo se realiza mediante búsqueda exhaustiva (*grid search*) sobre los parámetros $p \in \{0, 1, 2, 3\}$ y $q \in \{0, 1, 2, 3\}$, con el orden de diferenciación $d$ fijado según los resultados de la Sección 6. Para cada combinación se estima el modelo por máxima verosimilitud y se registra el AIC. Solo se consideran modelos en los que todos los coeficientes son estadísticamente significativos al nivel $\alpha = 0,05$.

### 7.2 Modelos Seleccionados

**Tabla 7.1 — Mejores modelos por criterio AIC**

| Serie | Modelo | AIC | BIC | d utilizado |
|---|---|:---:|:---:|:---:|
| S&P 500 | ARIMA(1, 1, 0) | −1.653,75 | −1.646,54 | 1 |
| VIX | ARIMA(3, 0, 3) | 1.844,87 | 1.870,18 | 0 |
| CO₂ | ARIMA(3, 1, 3) | 396,97 | 422,90 | 1 |
| AirPassengers | ARIMA(2, 1, 1) | −263,78 | −249,03 | 1 |

### 7.3 Coeficientes Estimados y Significatividad

**S&P 500 — ARIMA(1, 1, 0)**

| Parámetro | Coeficiente | Error Estándar | z | p-valor |
|---|:---:|:---:|:---:|:---:|
| ar.L1 | −0,0953 | 0,029 | −3,328 | 0,001 |
| σ² | 0,0001 | — | 36,197 | < 0,001 |

El coeficiente autorregresivo negativo y significativo indica leve reversión a la media en los log-retornos, interpretable como corrección de sobreajuste de muy corto plazo. La magnitud pequeña ($|\hat{\phi}_1| < 0,10$) confirma que la predictabilidad es económicamente escasa.

**VIX — ARIMA(3, 0, 3)**

| Parámetro | Coeficiente | Error Estándar | z | p-valor |
|---|:---:|:---:|:---:|:---:|
| ar.L1 | −0,7784 | 0,246 | −3,159 | 0,002 |
| ar.L2 | 0,9733 | 0,024 | 40,069 | < 0,001 |
| ar.L3 | 0,8015 | 0,237 | 3,387 | 0,001 |
| ma.L1 | 1,6137 | 0,243 | 6,627 | < 0,001 |
| ma.L2 | 0,5149 | 0,214 | 2,410 | 0,016 |
| ma.L3 | −0,1453 | 0,055 | −2,629 | 0,009 |
| σ² | 45,39 | — | 17,564 | < 0,001 |

La estructura ARMA(3,3) refleja la alta persistencia del VIX y su dinámica compleja. La alternancia de signos en los coeficientes AR es consistente con la naturaleza oscilante de la volatilidad de mercado.

**CO₂ — ARIMA(3, 1, 3)**

| Parámetro | Coeficiente | Error Estándar | z | p-valor |
|---|:---:|:---:|:---:|:---:|
| Constante | 0,1722 | 0,139 | 1,234 | 0,217 |
| ar.L1 | 0,9185 | 0,263 | 3,499 | < 0,001 |
| ar.L2 | −0,8031 | 0,286 | −2,808 | 0,005 |
| ar.L3 | 0,0941 | 0,231 | 0,407 | 0,684 |
| ma.L1 | −0,0585 | 0,256 | −0,228 | 0,819 |
| ma.L2 | 0,8161 | 0,091 | 9,012 | < 0,001 |
| ma.L3 | 0,2814 | 0,220 | 1,279 | 0,201 |
| σ² | 0,5800 | — | 8,987 | < 0,001 |

> **Nota:** La constante y los coeficientes ar.L3, ma.L1 y ma.L3 no resultan individualmente significativos, aunque el modelo en su conjunto presenta el menor AIC del espacio de búsqueda. Esta situación se explica por la multicolinealidad entre los coeficientes de alto orden en modelos ARMA de orden elevado. El modelo SARIMA(0,1,1)(0,1,1)₁₂ —analizado en la Sección 13— presenta mayor parsimonia y mejor performance.

**AirPassengers — ARIMA(2, 1, 1)**

| Parámetro | Coeficiente | Error Estándar | z | p-valor |
|---|:---:|:---:|:---:|:---:|
| Constante | 0,0044 | 0,001 | 5,759 | < 0,001 |
| ar.L1 | 0,9642 | 0,080 | 11,981 | < 0,001 |
| ar.L2 | −0,4029 | 0,086 | −4,682 | < 0,001 |
| ma.L1 | −0,9563 | 0,050 | −19,212 | < 0,001 |
| σ² | 0,0083 | — | 5,619 | < 0,001 |

Todos los coeficientes son altamente significativos. La constante positiva refleja la tendencia creciente en el log de pasajeros. La estructura AR(2) captura la persistencia inercial del tráfico aéreo.

---

## 8. Métricas de Performance

La Figura 8.1 muestra la partición train/test para cada serie, con el pronóstico del modelo ARIMA superpuesto al período de evaluación.

**Figura 8.1 — Partición Train/Test y ajuste del modelo (20% test)**

![Train Test split](informe_graficos/fig_12_cell22.png)

### 8.1 División Train/Test

Se adopta una partición temporal del 80% para entrenamiento y 20% para evaluación, respetando el orden cronológico de las observaciones (sin aleatorización). Esta metodología es la apropiada para series temporales, ya que preserva la estructura de dependencia y evita el data leakage hacia el futuro.

Las métricas calculadas sobre el conjunto de prueba son:

- **RMSE** (Root Mean Squared Error): $\sqrt{\frac{1}{n}\sum_{t=1}^n (Y_t - \hat{Y}_t)^2}$ — penaliza errores grandes cuadráticamente.
- **MAE** (Mean Absolute Error): $\frac{1}{n}\sum_{t=1}^n |Y_t - \hat{Y}_t|$ — robusto ante outliers.
- **MAPE** (Mean Absolute Percentage Error): $\frac{100}{n}\sum_{t=1}^n \left|\frac{Y_t - \hat{Y}_t}{Y_t}\right|$ — expresa el error en términos relativos.
- **R²**: proporción de varianza explicada por el modelo; un valor negativo indica que el modelo es peor que el pronóstico naive (media del conjunto de entrenamiento).

### 8.2 Resultados

**Tabla 8.1 — Performance de los modelos ARIMA en el conjunto de prueba (20%)**

| Serie | Modelo | RMSE | MAE | MAPE (%) | R² |
|---|---|:---:|:---:|:---:|:---:|
| S&P 500 | ARIMA(1,1,0) | 237,24 | 172,59 | 2,60 | −1,0828 |
| VIX | ARIMA(3,0,3) | 18,96 | 15,29 | 12,72 | −2,0114 |
| CO₂ | ARIMA(3,1,3) | 2,81 | 2,28 | 0,53 | 0,2734 |
| AirPassengers | ARIMA(2,1,1) | 86,79 | 77,18 | 18,84 | −0,2340 |

### 8.3 Interpretación

**S&P 500:** El MAPE del 2,60% parece modesto en términos relativos, pero el R² negativo (−1,08) indica que el modelo es peor que el pronóstico naive. Esto es esperable: en horizontes multi-paso para series con raíz unitaria, el error de pronóstico se acumula y el modelo ARIMA(1,1,0) converge al último valor observado (caminata aleatoria), sin agregar información predictiva adicional.

**VIX:** El MAPE del 12,72% y R² = −2,01 confirman la dificultad de predecir la volatilidad del mercado con modelos lineales. Los picos extremos del VIX (eventos de tail risk) no son anticipados por el componente ARMA, que pondera información histórica reciente.

**CO₂:** Es la única serie con R² positivo (0,27), lo que indica que el modelo aporta poder predictivo genuino sobre el benchmark naive. El MAPE de 0,53% refleja la alta regularidad del proceso subyacente (tendencia + ciclo estacional bien definidos). La escasa sorpresa en esta serie contrasta con el comportamiento impredecible de las series financieras.

**AirPassengers:** El R² de −0,23 y el MAPE de 18,84% reflejan la limitación del modelo no estacional para capturar los picos estivales de la serie. El modelo SARIMA(0,1,1)(0,1,1)₁₂ —analizado en la Sección 13— corrige sustancialmente este problema.

---

## 9. Comparación de Modelos

### 9.1 Benchmarks Utilizados

Se comparan los modelos ARIMA seleccionados contra dos benchmarks de referencia:

1. **Modelo Naive:** El pronóstico es el último valor observado ($\hat{Y}_{T+h} = Y_T$). Equivale a una caminata aleatoria sin deriva, apropiado para series I(1).
2. **Suavizamiento Exponencial de Holt (ETS-AAN):** Modelo de suavizamiento con nivel y tendencia aditivos, sin componente estacional. Captura tendencia pero no autocorrelaciones de mayor orden.

### 9.2 Resultados Comparativos

**Tabla 9.1 — Comparación de performance en conjunto de prueba**

| Serie | Modelo | RMSE | MAE | MAPE (%) | R² |
|---|---|:---:|:---:|:---:|:---:|
| S&P 500 | ARIMA(1,1,0) | 237,24 | 172,59 | 2,60 | −1,0828 |
| S&P 500 | Naïve | 236,36 | 171,58 | 2,59 | −1,0674 |
| S&P 500 | Holt ETS-AAN | 394,71 | 315,33 | 4,73 | −4,7655 |
| VIX | ARIMA(3,0,3) | 18,96 | 15,29 | 12,72 | −2,0114 |
| VIX | Naïve | 16,25 | 12,46 | 10,27 | −1,2120 |
| VIX | Holt ETS-AAN | 19,55 | 15,61 | 12,96 | −2,2028 |
| CO₂ | ARIMA(3,1,3) | 2,81 | 2,28 | 0,53 | **0,2734** |
| CO₂ | Naïve | 7,71 | 6,97 | 1,64 | −4,4818 |
| CO₂ | Holt ETS-AAN | 32,47 | 27,63 | 6,49 | −96,15 |
| AirPassengers | ARIMA(2,1,1) | 86,79 | 77,18 | 18,84 | −0,2340 |
| AirPassengers | Naïve | 93,13 | 81,45 | 20,20 | −0,4209 |
| AirPassengers | Holt ETS-AAN | 122,63 | 109,91 | 27,63 | −1,4636 |

### 9.3 Análisis por Serie

**S&P 500:** El modelo ARIMA(1,1,0) y el Naive obtienen resultados prácticamente idénticos (MAE = 172,59 vs. 171,58). La diferencia es estadísticamente despreciable. El Holt es notoriamente peor (MAE = 315,33), ya que el componente de tendencia del suavizamiento introduce sesgo sistemático cuando la serie invierte su dirección durante el período de prueba. **Conclusión:** La serie de retornos del S&P 500 no es predecible con modelos lineales univariados, en consistencia con la hipótesis de eficiencia de mercado en su forma débil.

**VIX:** El Naive supera al ARIMA en todas las métricas (MAE = 12,46 vs. 15,29). Esto indica que, para este horizonte de pronóstico multi-paso, la estructura ARMA(3,3) no agrega valor sobre la simple persistencia del último valor. La alta volatilidad del VIX durante el período de prueba amplifica los errores del modelo estructurado.

**CO₂:** El ARIMA supera claramente a ambos benchmarks. El Naive tiene un MAE 3,06 veces mayor (6,97 vs. 2,28); el Holt, 12,12 veces mayor (27,63 vs. 2,28). El deterioro del Holt se explica por su incapacidad para modelar el componente estacional, que genera desviaciones sistemáticas en los meses de máximo y mínimo del ciclo anual.

**AirPassengers:** El ARIMA supera al Naive (MAE 77,18 vs. 81,45) y al Holt (77,18 vs. 109,91), pero la magnitud de la mejora es modesta dado que ningún modelo captura la estacionalidad multiplicativa. El modelo SARIMA estacional (Sección 13) rectifica esta limitación.

---

## 10. Diagnóstico de Residuos

Las Figuras 10.1 a 10.4 presentan el panel de diagnóstico de residuos para cada modelo: serie temporal de residuos, histograma con curva normal, Q-Q plot y FAC de residuos.

**Figura 10.1 — Diagnóstico de residuos: S&P 500 ARIMA(1,1,0)**

![Residuos S&P 500](informe_graficos/fig_13_cell26.png)

**Figura 10.2 — Diagnóstico de residuos: VIX ARIMA(3,0,3)**

![Residuos VIX](informe_graficos/fig_14_cell26.png)

**Figura 10.3 — Diagnóstico de residuos: CO₂ ARIMA(3,1,3)**

![Residuos CO2](informe_graficos/fig_15_cell26.png)

**Figura 10.4 — Diagnóstico de residuos: AirPassengers ARIMA(2,1,1)**

![Residuos AirPassengers](informe_graficos/fig_16_cell26.png)

### 10.1 Metodología

El diagnóstico de residuos verifica si los errores del modelo $\hat{\varepsilon}_t = Y_t - \hat{Y}_t$ se comportan como ruido blanco gaussiano. Se aplican:

1. **FAC de residuos:** Inspección visual de autocorrelaciones; no deben existir picos significativos fuera de las bandas de confianza al 95%.
2. **Test de Ljung-Box:** Contraste formal de ausencia de autocorrelación en los primeros $h$ rezagos (se reportan $h = 10$ y $h = 20$).
3. **Test de Jarque-Bera:** Contraste de normalidad basado en asimetría y exceso de curtosis.

### 10.2 Resultados

**Tabla 10.1 — Tests de diagnóstico de residuos**

| Serie | LB(h=10) p-valor | LB(h=20) p-valor | JB p-valor | Conclusión |
|---|:---:|:---:|:---:|---|
| S&P 500 | 1,0000 | 1,0000 | < 0,0001 | Ruido blanco; colas pesadas |
| VIX | 0,3528 | 0,7138 | < 0,0001 | Ruido blanco; colas pesadas |
| CO₂ | 1,0000 | 1,0000 | < 0,0001 | Ruido blanco; colas pesadas |
| AirPassengers | 1,0000 | 1,0000 | < 0,0001 | Ruido blanco; colas pesadas |

### 10.3 Interpretación

**Test de Ljung-Box:** Todos los modelos superan el test con amplio margen (p-valores muy superiores a 0,05). Esto confirma que los modelos ARIMA capturan adecuadamente la estructura de autocorrelación lineal de las series. Los residuos no presentan autocorrelación serial significativa en ningún horizonte evaluado.

**Test de Jarque-Bera:** Todos los modelos fallan el test de normalidad (p < 0,0001), indicando que los residuos presentan exceso de curtosis (colas más pesadas que las de una distribución normal) y/o asimetría. Este resultado es esperado y no invalida los modelos: las colas pesadas en series financieras y de volatilidad son un hecho estilizado ampliamente documentado (fenómeno de *fat tails*). Para pronósticos de intervalos de confianza precisos, se requeriría modelar la distribución de los residuos con distribuciones de cola pesada (t-Student, GED) o mediante modelos ARCH/GARCH, lo que excede el alcance del presente trabajo.

> **Síntesis:** Los modelos estimados satisfacen el criterio de adecuación de la estructura de media (ruido blanco en residuos), pero no el de adecuación de la distribución (normalidad). Los pronósticos puntuales son válidos; los intervalos de predicción deben interpretarse con cautela.

---

## 11. Pronósticos

### 11.1 Horizontes de Pronóstico

Los horizontes se definen en función de la frecuencia de la serie y de la utilidad práctica del pronóstico:

| Serie | Horizonte | Equivalente temporal |
|---|:---:|---|
| S&P 500 | 40 pasos | ~2 meses hábiles |
| VIX | 40 pasos | ~2 meses hábiles |
| CO₂ | 24 pasos | 2 años |
| AirPassengers | 24 pasos | 2 años |

**Justificación:** Para series financieras diarias, un horizonte de 40 días (~2 meses) es el máximo razonable antes de que los intervalos de confianza se vuelvan informativamente vacíos. Para series mensuales con tendencia y estacionalidad (CO₂, AirPassengers), 24 meses permiten capturar al menos dos ciclos estacionales completos y observar la propagación de la incertidumbre.

Las Figuras 11.1 a 11.4 muestran los pronósticos de cada modelo sobre el horizonte indicado, con intervalos de confianza al 95%.

**Figura 11.1 — Pronóstico S&P 500 (h = 40 días hábiles)**

![Pronóstico S&P 500](informe_graficos/fig_17_cell28.png)

**Figura 11.2 — Pronóstico VIX (h = 40 días hábiles)**

![Pronóstico VIX](informe_graficos/fig_18_cell28.png)

**Figura 11.3 — Pronóstico CO₂ (h = 24 meses)**

![Pronóstico CO2](informe_graficos/fig_19_cell28.png)

**Figura 11.4 — Pronóstico AirPassengers (h = 24 meses)**

![Pronóstico AirPassengers](informe_graficos/fig_20_cell28.png)

### 11.2 Resultados de Pronósticos

**S&P 500 y VIX:** Los pronósticos de pasos múltiples convergen rápidamente hacia la tendencia implícita (caminata aleatoria), con intervalos de confianza que se ensanchan en forma de cono. Dado que los retornos son I(1), el pronóstico en niveles hereda la incertidumbre acumulada, y el intervalo de predicción al 95% a 40 días es amplio. Esto es coherente con la imposibilidad de predecir precios de mercado a horizontes medianos.

**CO₂:** El pronóstico presenta una extrapolación de la tendencia creciente combinada con el patrón estacional anual. Los intervalos de confianza son angostos en comparación con las series financieras, dado que el proceso es de menor variabilidad intrínseca y la tendencia es robusta y predecible.

**AirPassengers:** El pronóstico replica el patrón estacional multiplicativo (en escala logarítmica, aditivo) con tendencia creciente. Los intervalos de predicción se expanden con el horizonte pero mantienen la forma estacional, lo que confiere utilidad práctica al pronóstico.

---

## 12. Modelos SARIMAX con Variables Exógenas

### 12.1 Justificación Teórica

La relación entre el índice S&P 500 y el VIX constituye uno de los hallazgos más robustos de la microestructura del mercado financiero. La correlación negativa entre retornos bursátiles y cambios en la volatilidad implícita fue documentada sistemáticamente por Black (1976) y denominada "efecto leverage". Su mecanismo: una caída bursátil incrementa el apalancamiento financiero de las empresas (su ratio deuda/patrimonio), lo que eleva el riesgo percibido por los participantes del mercado y, por ende, la volatilidad implícita.

La asimetría del efecto —caídas generan mayor aumento de VIX que las subidas equivalentes— convierte a la relación S&P/VIX en un caso de estudio ideal para los modelos SARIMAX.

### 12.2 Modelo A — VIX con retornos del S&P 500 como variable exógena

**Especificación:** $\Delta\text{VIX}_t = \beta_A \cdot r_t^{SP} + \text{ARIMA}(3, 0, 2)$

| Parámetro | Estimación | p-valor |
|---|:---:|:---:|
| β_A (retorno S&P 500) | −394,393 | < 0,0001 |

**Interpretación cuantitativa del efecto leverage:**
- Una caída del 1% en el S&P 500 ($r_t = -0,01$) está asociada a un aumento de $|-394,39 \times (-0,01)| = +3,94$ puntos en el VIX.
- Una caída del 5% ($r_t = -0,05$) está asociada a un aumento de $+19,72$ puntos en el VIX.

**Impacto sobre los criterios de información:**

| Especificación | AIC | BIC | ΔAIC vs. sin exógena |
|---|:---:|:---:|:---:|
| ARIMA(3, 0, 2) sin exógena | 1.431,75 | 1.452,05 | — |
| SARIMAX(3, 0, 2) con retorno S&P | 1.292,09 | 1.315,79 | **−139,66** |

**Performance en conjunto de prueba:**

| Modelo | RMSE | MAE | MAPE (%) | R² |
|---|:---:|:---:|:---:|:---:|
| ARIMA(3,0,2) sin exógena | 19,08 | 15,79 | 13,30 | −2,3807 |
| SARIMAX + retorno S&P | **11,19** | **8,69** | **7,54** | **−0,1621** |

La incorporación del retorno del S&P 500 reduce el MAE en un 45% y el MAPE a la mitad (de 13,30% a 7,54%). El R² mejora de −2,38 a −0,16, acercándose al umbral de poder predictivo positivo.

### 12.3 Modelo B — S&P 500 con cambios en VIX como variable exógena

**Especificación:** $r_t^{SP} = \beta_B \cdot \Delta\text{VIX}_t + \text{ARIMA}(2, 0, 3)$

| Parámetro | Estimación | p-valor |
|---|:---:|:---:|
| β_B (cambio en VIX) | −0,00124 | < 0,0001 |

**Interpretación cuantitativa:**
- Un aumento de 1 punto en el VIX está asociado a una variación de −0,00124 en los retornos del S&P 500 (−12,4 puntos básicos).
- Un aumento de 5 puntos en el VIX está asociado a −62,1 puntos básicos en el retorno.

**Impacto sobre los criterios de información:**

| Especificación | AIC | BIC | ΔAIC vs. sin exógena |
|---|:---:|:---:|:---:|
| ARIMA(2, 0, 3) sin exógena | −1.317,56 | −1.297,25 | — |
| SARIMAX(2, 0, 3) con ΔVIX | −1.464,21 | −1.440,52 | **−146,65** |

**Performance en conjunto de prueba:**

| Modelo | RMSE | MAE | MAPE (%) | R² |
|---|:---:|:---:|:---:|:---:|
| ARIMA(2,0,3) sin exógena | 224,83 | 158,37 | 2,39 | −0,8706 |
| SARIMAX + ΔVIX | **116,37** | **73,12** | **1,11** | **0,4989** |

Este es el resultado más notable del trabajo: la incorporación de los cambios en el VIX como variable exógena genera un modelo con R² = 0,50 en el conjunto de prueba, reduciendo el MAE en un 54% respecto al modelo univariado. El modelo predice correctamente la mitad de la varianza de los retornos del S&P 500 en el período de evaluación, resultado excepcional para datos financieros en alta frecuencia.

La Figura 12.1 muestra el desempeño comparado de los modelos SARIMAX vs. sus contrapartes sin variable exógena en el período de prueba, tanto para el VIX (Modelo A) como para el S&P 500 (Modelo B).

**Figura 12.1 — SARIMAX: comparación con y sin variable exógena (Modelos A y B)**

![SARIMAX comparación](informe_graficos/fig_21_cell32.png)

### 12.4 Conclusiones del Análisis SARIMAX

1. La relación S&P 500 — VIX es **bidireccional y estadísticamente significativa** en ambas direcciones (p < 0,0001 en ambos modelos).
2. Las mejoras en AIC superan los 139 puntos en el Modelo A y 146 puntos en el Modelo B, lo que constituye evidencia abrumadora de que la variable exógena contribuye información genuina.
3. El efecto leverage es **asimétrico**: dado que el coeficiente β_A = −394,39 opera sobre retornos (que tienen distribución aproximadamente simétrica), el VIX amplifica más las caídas que las subidas cuando se controla por la magnitud del movimiento.
4. Los resultados son consistentes con la literatura financiera (Black, 1976; Christie, 1982; Schwert, 1989) y sugieren que los modelos SARIMAX son una alternativa valiosa para la predicción de indicadores de volatilidad en presencia de información exógena contemporánea.

---

## 13. Análisis Estacional y Test HEGY

### 13.1 Motivación

Las series CO₂ y AirPassengers presentan estacionalidad mensual (período $s = 12$). El análisis de la Sección 9 demostró que los modelos ARIMA no estacionales son insuficientes para estas series, en particular para AirPassengers. En esta sección se ajustan modelos SARIMA(0,1,1)(0,1,1)₁₂ —el denominado modelo Airline— y se contrastan formalmente las hipótesis de raíz unitaria estacional.

La Figura 13.1 presenta la descomposición estacional aditiva (tendencia, estacionalidad, residuo) para CO₂ y AirPassengers (en logaritmos). Esta descomposición permite visualizar la magnitud relativa de cada componente y fundamenta la necesidad de la diferenciación estacional.

**Figura 13.1 — Descomposición estacional: CO₂ y AirPassengers**

![Descomposición estacional](informe_graficos/fig_22_cell35.png)

La Figura 13.2 muestra la FAC de las series diferenciadas de forma regular ($\Delta_1$) y estacional ($\Delta_1\Delta_{12}$), revelando el patrón residual que motiva la especificación del modelo Airline.

**Figura 13.2 — FAC de series con diferenciación estacional**

![FAC series diferenciadas estacionalmente](informe_graficos/fig_23_cell36.png)

### 13.2 Test HEGY para Raíces Unitarias Estacionales

El test de Hylleberg, Engle, Granger y Yoo (HEGY, 1990) permite distinguir entre raíces unitarias en distintas frecuencias estacionales. En la presente implementación se aproxima el test mediante la aplicación del test ADF sobre la serie con diferenciación estacional de orden 12 ($\Delta_{12} Y_t$): si ADF rechaza la hipótesis nula de raíz unitaria sobre $\Delta_{12} Y_t$, se concluye que una diferenciación estacional de orden $D = 1$ es suficiente.

**Tabla 13.1 — Test ADF sobre serie con diferenciación estacional**

| Serie | Estadístico ADF (sobre Δ₁₂) | p-valor | Conclusión |
|---|:---:|:---:|---|
| CO₂ | −3,3889 | 0,0113 | Rechaza H₀ → D = 1 suficiente |
| log(AirPassengers) | −2,7096 | 0,0724 | No rechaza H₀ (marginal, 10%) → D = 1 recomendado |

Para CO₂, la diferenciación estacional de orden 1 elimina la raíz unitaria estacional con claridad. Para AirPassengers, el p-valor marginal (0,072) indica que la serie diferenciada estacionalmente aún roza el límite de la estacionariedad, pero la inspección visual y la literatura —que documenta el modelo Airline como óptimo para esta serie— respaldan la elección de $D = 1$.

### 13.3 Modelos SARIMA(0,1,1)(0,1,1)₁₂

El denominado **modelo Airline** fue propuesto por Box & Jenkins (1976) precisamente para la serie AirPassengers y se ha convertido en referencia para series mensuales con tendencia y estacionalidad:

$$(1-B)(1-B^{12})\ln Y_t = (1+\theta_1 B)(1+\Theta_1 B^{12})\varepsilon_t$$

### 13.4 Comparación SARIMA vs. ARIMA

**Tabla 13.2 — Mejora por incorporación del componente estacional**

| Serie | ARIMA(0,1,1) AIC | SARIMA(0,1,1)(0,1,1)₁₂ AIC | ΔAIC | Ganancia |
|---|:---:|:---:|:---:|---|
| CO₂ | 553,06 | 158,72 | **−394,34** | Decisiva |
| AirPassengers | −237,51 | −481,42 | **−243,91** | Decisiva |

Una diferencia de ΔAIC > 10 ya se considera evidencia fuerte a favor del modelo con menor AIC (Burnham & Anderson, 2002). Las diferencias observadas (394 y 244 puntos) son extraordinariamente grandes, confirmando que el componente estacional es indispensable para estas series.

Las Figuras 13.3 y 13.4 presentan los pronósticos de los modelos Airline para CO₂ y AirPassengers respectivamente, incluyendo intervalos de confianza al 95%.

**Figura 13.3 — Pronóstico SARIMA(0,1,1)(0,1,1)₁₂ — CO₂**

![SARIMA CO2](informe_graficos/fig_24_cell37.png)

**Figura 13.4 — Pronóstico SARIMA(0,1,1)(0,1,1)₁₂ — AirPassengers**

![SARIMA AirPassengers](informe_graficos/fig_25_cell37.png)

### 13.5 Interpretación de los Patrones Estacionales

**CO₂:** El ciclo anual de 7 ppm de amplitud refleja el metabolismo de la biosfera terrestre. Las plantas del hemisferio norte (que concentra la mayor masa de vegetación continental) absorben CO₂ durante la primavera y el verano boreales (marzo-septiembre), generando un mínimo en septiembre. Durante el otoño e invierno (octubre-mayo), la descomposición de materia orgánica y la reducción de la fotosíntesis elevan la concentración, generando el máximo en mayo.

**AirPassengers:** Los picos estivales (julio-agosto) y los valles invernales son consecuencia del comportamiento del turismo vacacional y los viajes de negocio. La estacionalidad multiplicativa en escala original —amplitudes que crecen con la tendencia— es indicativa de un proceso en el que el componente estacional opera como un factor relativo del nivel de la serie.

---

## 14. Conclusiones

El análisis desarrollado en el presente trabajo permite extraer las siguientes conclusiones:

**1. Sobre la naturaleza de las series:**
Las series financieras (S&P 500 y VIX) se comportan como procesos de alta incertidumbre con propiedades cercanas a la caminata aleatoria, en contraste con las series climática (CO₂) y de transporte (AirPassengers), que exhiben componentes sistemáticos —tendencia y estacionalidad— que los modelos ARIMA/SARIMA capturan eficientemente.

**2. Sobre la performance predictiva:**
La predictabilidad de las series es heterogénea. La serie de CO₂ es la única que muestra R² positivo con modelos ARIMA (0,27), mientras que las series financieras presentan R² negativos en evaluación multi-paso, consistentes con la hipótesis de eficiencia de mercado. La incorporación de estacionalidad en los modelos SARIMA genera mejoras de AIC del orden de 240-394 puntos para CO₂ y AirPassengers.

**3. Sobre el efecto leverage:**
El análisis SARIMAX confirma la relación bidireccional entre el S&P 500 y el VIX con alta significatividad estadística. El Modelo B —S&P 500 con cambios en VIX como regresor— alcanza R² = 0,50 en el conjunto de prueba, resultado excepcional para series financieras diarias. Esto sugiere que la información contenida en el mercado de opciones (VIX) es útil para anticipar movimientos del mercado accionario a muy corto plazo.

**4. Sobre el diagnóstico de residuos:**
Todos los modelos producen residuos sin autocorrelación serial (test de Ljung-Box no significativo), lo que valida la adecuación de la estructura de media modelada. Sin embargo, ningún modelo produce residuos normales (test de Jarque-Bera significativo), lo que es esperable ante la presencia de colas pesadas en todas las series analizadas.

**5. Sobre las limitaciones:**
Los modelos ARIMA/SARIMA son lineales y asumen homoscedasticidad de los residuos. Para series financieras, la heteroscedasticidad condicional (efecto ARCH) es un hecho estilizado que requeriría extensiones GARCH para su tratamiento adecuado. Adicionalmente, la baja cantidad de observaciones de las series financieras (274-279 datos diarios ≈ 14 meses) limita la potencia de los tests y la robustez de las estimaciones.

---

## 15. Referencias

- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Black, F. (1976). Studies of stock price volatility changes. *Proceedings of the 1976 Meetings of the Business and Economic Statistics Section*, ASA, 177-181.
- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach* (2nd ed.). Springer.
- Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.
- Christie, A. A. (1982). The stochastic behavior of common stock variances: Value, leverage and interest rate effects. *Journal of Financial Economics*, 10(4), 407-432.
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427-431.
- Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S. (1990). Seasonal integration and cointegration. *Journal of Econometrics*, 44(1-2), 215-238.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. Disponible en: https://otexts.com/fpp3/
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1-3), 159-178.
- Schwert, G. W. (1989). Why does stock market volatility change over time? *Journal of Finance*, 44(5), 1115-1153.
- Tsay, R. S. (2010). *Analysis of Financial Time Series* (3rd ed.). Wiley.
- NOAA Global Monitoring Laboratory. (2026). *Trends in Atmospheric Carbon Dioxide*. Recuperado de https://gml.noaa.gov/ccgg/trends/

---

## 16. Apéndice — Código

El código completo se adjunta en el archivo `TP1_Series_Temporales_v2.ipynb`. A continuación se listan las funciones principales implementadas:

### `plot_fas_fac_facp(serie, lags, titulo)`
Genera el gráfico conjunto de la serie, su FAC y su FACP con bandas de confianza al 95%.

### `buscar_arima(serie, d, p_max, q_max, alpha)`
Realiza la búsqueda exhaustiva de modelos ARIMA sobre el espacio $(p, q) \in \{0, \ldots, p\_max\} \times \{0, \ldots, q\_max\}$ para el orden de diferenciación `d` dado. Filtra modelos con coeficientes no significativos al nivel `alpha` y retorna los resultados ordenados por AIC.

### `pronosticar(modelo, serie, h, titulo)`
Genera el pronóstico de `h` pasos hacia adelante con intervalos de confianza al 95%, junto con su visualización.

### `diagnostico_residuos(modelo, serie, lags, alpha)`
Aplica el test de Ljung-Box para los primeros `lags` rezagos y grafica la FAC de los residuos. Reporta la tabla de p-valores del test.

### `comparar_modelos(series_dict, modelos_dict, train_ratio)`
Compara la performance de los modelos ARIMA contra los benchmarks Naive y Holt ETS-AAN en el conjunto de prueba, retornando una tabla con RMSE, MAE, MAPE y R².

---

*Fin del documento*
