REFERENCIAS POR VARIABLE — PROYECTO KAGGLE POBREZA COLOMBIA
=============================================================

VARIABLES DE HOGARES
---------------------

Clase / Dominio (zona urbana vs. rural)
  Obando Rozo, N., & Andrián, L. G. (2015). Measuring changes in poverty in
  Colombia: The 2000s. Inter-American Development Bank.
  https://publications.iadb.org/publications/english/document/Measuring-Changes-in-Poverty-in-Colombia-The-2000s.pdf

Depto (departamento como fixed effect)
  World Bank. (2019). Poverty and shared prosperity in Colombia: Background
  paper for policy notes. World Bank Group.
  https://documents1.worldbank.org/curated/en/657941560749443721/pdf/Poverty-and-Shared-Prosperity-in-Colombia-Background-Paper-for-Policy-Notes.pdf

P5090 (tenencia de vivienda)
  Shi, Y., Deng, M., & Liu, H. (2021). Is poverty predictable with machine
  learning? A study of DHS data from Kyrgyzstan. Social Indicators Research,
  159(1), 367-390. https://doi.org/10.1007/s11205-021-02763-6

Nper / Npersug (tamaño del hogar)
  Marrugo-Arnedo, C. A., Del Risco-Serje, K. P., Marrugo-Arnedo, V.,
  Herrera-Llamas, J. A., & Pérez-Valbuena, G. J. (2015). Determinants of
  poverty in the Colombian Caribbean region. Economía, 15, 47-69.
  https://www.researchgate.net/publication/282897486


VARIABLES DE PERSONAS (agregadas al hogar)
--------------------------------------------

P6210 / P6210s1 (nivel educativo)
  Marrugo-Arnedo et al. (2015). Op. cit.

  Corral, P., Molina, I., & Nguyen, M. (2024). Poverty mapping in the age of
  machine learning. Journal of Development Economics, 171, 103383.
  https://doi.org/10.1016/j.jdeveco.2024.103383

Oc, Des, Ina, Pet (estado laboral)
  World Bank. (2019). Op. cit.

P6430 (posición ocupacional — cuenta propia vs. formal)
  Solano, A., Sauma, P., & Trejos, J. D. (2022). A machine learning proposal
  to predict poverty. Agronomía Costarricense, 46(2), 84-98.
  https://www.scielo.sa.cr/scielo.php?script=sci_arttext&pid=S0379-39822022000400084

P6920 (cotización a pensión — proxy de formalidad)
  World Bank. (2019). Op. cit.

P6090 / P6100 (afiliación y régimen de salud)
  Marrugo-Arnedo et al. (2015). Op. cit.

n_menores_18 (número de menores en el hogar)
  Nkurunziza, E., Niyibizi, A., & Ingabire, M. (2024). Enhancing poverty
  classification in developing countries through machine learning. Cogent
  Economics & Finance, 12(1).
  https://doi.org/10.1080/23322039.2024.2444374

tasa_ocupacion (ratio ocupados/PET)
  Oxford Poverty and Human Development Initiative (OPHI). (2024). Colombia MPI.
  University of Oxford.
  https://ophi.org.uk/national-mpi-directory/colombia-mpi


JUSTIFICACIÓN DE EXCLUSIÓN DE VARIABLES DE INGRESO (data leakage)
-------------------------------------------------------------------

Argumento general — uso de proxies en lugar de ingreso directo
  Nguyen, C. V., & Tran, D. T. (2018). Proxy means tests to identify the
  income poor. International Journal of Social Economics, 45(1).
  https://doi.org/10.1177/0021909617709486

Variables legítimas de PMT (ocupación sí, ingreso monetario no)
  Browne, J., Moll, P., & Ravallion, M. (2018). A poor means test?
  Econometric targeting in Africa. Journal of Development Economics, 134,
  109-124. https://doi.org/10.1016/j.jdeveco.2018.05.004

Fundamento original del Proxy Means Test en América Latina
  Grosh, M., & Baker, J. L. (1995). Proxy means tests for targeting social
  programs (LSMS Working Paper No. 118). World Bank.
  https://documents1.worldbank.org/curated/en/750401468776352539/pdf/multi-page.pdf

Riesgo de usar predictores fuertes directos en lugar de proxies
  Montgomery, M. R., Gragnolati, M., Burke, K. A., & Paredes, E. (2000).
  Measuring living standards with proxy variables. Demography, 37(2), 155-174.
  https://doi.org/10.2307/2648118
