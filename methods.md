# Общие тезисы #

Наблюдаемая траектория судна:

$$ X=\{x_i\}_{i=1}^{N}, \qquad x_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i). $$

Предсказание и ground truth:

$$ P=\{(t_i,\mathrm{lat}_i,\mathrm{lon}_i,\hat y_i)\}_{i=1}^{N}, \qquad G=\{(t_i,\mathrm{lat}_i,\mathrm{lon}_i,y_i)\}_{i=1}^{N}. $$

Здесь \(y_i=0\) — аномалия, \(y_i>0\) — штатное состояние. Бинарная метка аномалии:

$$ a_i=\mathbf1[y_i=0]. $$

Ускорение далее обозначается \(a_i^{\mathrm{acc}}\), чтобы не смешивать его с \(a_i\).

Метод \(M\) сопоставляет точке, окну или траектории anomaly score \(s^{(M)}\). Формула \(s_i^{(M)}=M(X_{\le i})\) относится только к online/causal-вариантам. Оконные и batch-методы используют окно, матрицу окон или весь набор точек. Обучение только на нормальных данных нужно forecasting- и reconstruction-методам; DBSCAN, LOF и Isolation Forest могут принимать весь набор признаков.

Online/causal: ARIMA, фильтр Калмана, particle filter; при причинной реализации также TCN и HMM. Окно или весь набор: Hampel, DBSCAN, LOF, Isolation Forest, One-Class SVM, RPCA, LSTM-автоэнкодер, DONUT, Anomaly Transformer; batch-GP использует все наблюдения.

Порог и перевод в разметку. Здесь \(\hat a_i=1\) означает аномалию, тогда как в исходной схеме классов аномалия — это метка \(0\):

$$ \hat a_i^{(M)}=\mathbf1[s_i^{(M)}>\tau_M], \qquad \hat y_i^{(M)}=\begin{cases} 0, & s_i^{(M)}>\tau_M,\\ 1, & s_i^{(M)}\le\tau_M. \end{cases} $$

Конкретный положительный класс при \(y_i>0\) — отдельная задача классификации. Для бинарной оценки \(\hat a_i=\mathbf1[\hat y_i=0]\).

# ARIMA для прогнозирования и anomaly detection #

Модель прогнозирует следующее значение по предыдущим и помечает точку как аномальную при большом расхождении с прогнозом.

Qin et al. не фиксируют порядок ARIMA\((2,1,1)\). В статье: скользящее окно, порядок по AIC/BIC, переобучение после сдвига окна, short-step прогноз, экспоненциально взвешенное усреднение предсказаний, детекция по относительной ошибке (порог порядка 5–15% в зависимости от ряда).

Ниже — наша адаптация. Порядок \((2,1,1)\) выбран нами. Нормировка \(|r_t|/\sigma_t\) — статистический score, а не формула статьи.

Для одномерного ряда модель ARIMA\((p,d,q)\):

$$ \phi(B)(1-B)^d x_t = c+\theta(B)\varepsilon_t, \qquad \varepsilon_t\sim\mathcal N(0,\sigma^2), $$

$$ \phi(B)=1-\phi_1B-\ldots-\phi_pB^p, \qquad \theta(B)=1+\theta_1B+\ldots+\theta_qB^q. $$

Фиксированный вариант:

$$ \Delta x_t = c+\phi_1\Delta x_{t-1}+\phi_2\Delta x_{t-2}+\varepsilon_t+\theta_1\varepsilon_{t-1}. $$

$$ r_t=x_t-\hat x_t, \qquad s_t=\frac{|r_t|}{\sigma_t}, \qquad s_t>\tau. $$

## Частный случай для \(P\) и \(G\) ##

Предпочтительно сначала перейти в локальные метрические координаты. Две независимые модели широты и долготы — упрощение:

$$ \mathrm{lat}_i \sim \mathrm{ARIMA}(2,1,1), \qquad \mathrm{lon}_i \sim \mathrm{ARIMA}(2,1,1). $$

$$ s_i=d_{\mathrm{geo}}(\mathbf z_i,\hat{\mathbf z}_i), \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

Метка \(y_i=0\) при построении прогноза не нужна.

**Источник:** Qin Yu, Lyu Jibin, Lirui Jiang. An Improved ARIMA-Based Traffic Anomaly Detection Algorithm for Wireless Sensor Networks // International Journal of Distributed Sensor Networks. 2016. Vol. 2016. Article ID 9653230. 9 p. DOI: [10.1155/2016/9653230](https://doi.org/10.1155/2016/9653230). [PDF](docs/methods/An%20Improved%20ARIMA-Based%20Traffic%20Anomaly%20Detection%20Algorithm%20for%20Wireless%20Sensor%20Networks.pdf).

# Калмановский фильтр с моделью постоянной скорости #

Фильтр ведёт скрытое состояние «положение и скорость» и считает измерение аномальным, если оно плохо согласуется с предсказанным продолжением движения. Kalman (1960) задаёт фильтр, но не детектор аномалий: normalized innovation squared и критерий \(\chi^2\) — наша статистическая адаптация.

## Общая математическая постановка ##

Состояние и модель постоянной скорости:

$$ \mathbf x_i= \begin{pmatrix} p_x\\ p_y\\ v_x\\ v_y \end{pmatrix}_i, \qquad \mathbf x_i=A_i\mathbf x_{i-1}+\mathbf w_i, \qquad \mathbf w_i\sim\mathcal N(0,Q_i), $$

$$ A_i= \begin{pmatrix} 1&0&\Delta t_i&0\\ 0&1&0&\Delta t_i\\ 0&0&1&0\\ 0&0&0&1 \end{pmatrix}. $$

Измерение:

$$ \mathbf z_i=H\mathbf x_i+\mathbf v_i, \qquad \mathbf v_i\sim\mathcal N(0,R), \qquad H= \begin{pmatrix} 1&0&0&0\\ 0&1&0&0 \end{pmatrix}. $$

Инновация и её ковариация:

$$ \mathbf r_i=\mathbf z_i-H\hat{\mathbf x}_{i|i-1}, \qquad S_i=HP_{i|i-1}H^T+R. $$

$$ s_i=\mathbf r_i^T S_i^{-1}\mathbf r_i. $$

При корректно заданной гауссовской модели инновации \(s_i\sim\chi^2_2\), и \(s_i>\chi^2_{2,\,1-\alpha}\) означает аномальное измерение. Из одной формулы фильтра это не следует.

## Частный случай для \(P\) и \(G\) ##

Широта и долгота — угловые координаты, поэтому постоянную скорость задаём в локальной метрической проекции:

$$ (p_x,p_y)_i=\operatorname{Project}(\mathrm{lat}_i,\mathrm{lon}_i), \qquad \mathbf z_i=(p_x,p_y)_i^T. $$

Скорости оценивает фильтр. Затем \(\hat a_i=\mathbf1[s_i>\tau]\).

**Источник:** Kalman R. E. A New Approach to Linear Filtering and Prediction Problems // Journal of Basic Engineering. 1960. Vol. 82, No. 1. P. 35–45. DOI: [10.1115/1.3662552](https://doi.org/10.1115/1.3662552).

# Bootstrap Particle Filter / SIR #

Вместо одного прогноза держится набор гипотез о движении. Аномалия — измерение, маловероятное для всех гипотез сразу. Particle filter может представлять мультимодальное распределение состояния; развилки и резкие повороты требуют соответствующей transition model, а не только замены фильтра Калмана.

## Общая математическая постановка ##

$$ p(\mathbf x_i\mid z_{1:i}) \approx \sum_{j=1}^{K}w_i^{(j)} \delta(\mathbf x_i-\mathbf x_i^{(j)}). $$

Шаг SIR: resample по весам \(w_{i-1}\); propagate \(\mathbf x_i^{(j)}\sim p(\mathbf x_i\mid\mathbf x_{i-1}^{(j)})\); weight и normalize:

$$ \tilde w_i^{(j)} = p(\mathbf z_i\mid\mathbf x_i^{(j)}), \qquad w_i^{(j)} = \frac{\tilde w_i^{(j)}}{\sum_{k=1}^{K}\tilde w_i^{(k)}}. $$

Без resampling это SIS, а не SIR. После resample веса равны \(1/K\), поэтому

$$ s_i=-\log\left(\frac{1}{K}\sum_{j=1}^{K}p(\mathbf z_i\mid\mathbf x_i^{(j)})\right). $$

## Частный случай для \(P\) и \(G\) ##

Та же модель постоянной скорости, что у фильтра Калмана, в локальных метрических координатах, но с \(K\) гипотезами. \(\hat a_i=\mathbf1[s_i>\tau]\).

**Источник:** Arulampalam M. S., Maskell S., Gordon N., Clapp T. A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking // IEEE Transactions on Signal Processing. 2002. Vol. 50, No. 2. P. 174–188. DOI: [10.1109/78.978374](https://doi.org/10.1109/78.978374). [PDF](docs/methods/A%20Tutorial%20on%20Particle%20Filters%20for%20Online.pdf).

# Gaussian Process Regression #

Нормальная траектория описывается функцией времени с оценкой неопределённости. Точка аномальна, если лежит далеко от этой функции относительно предсказанного разброса.

Smith et al. — другой алгоритм: GP вместе с extreme value theory, расхождением Хеллингера и NMF для batch- и streaming-детекции морских треков. Ближе к формулам ниже работа Penacho Riveiros et al.: GP на номинальных траекториях и проверка новой траектории по распределению остатков при заданном false-positive rate. RBF-ядро и квадратичная форма — классическая GP-регрессия, не алгоритм Smith et al.

## Общая математическая постановка ##

$$ f(t)\sim GP(m(t),k(t,t')), \qquad k(t,t') = \sigma_f^2 \exp \left( -\frac{(t-t')^2}{2\ell^2} \right). $$

$$ y_i=f(t_i)+\varepsilon_i, \qquad \varepsilon_i\sim\mathcal N(0,\sigma_n^2), \qquad f(t_*)\mid\mathcal D \sim \mathcal N(\mu_*,\sigma_*^2). $$

Для векторного прогноза в локальных метрических координатах

$$ s_i=(\mathbf z_i-\boldsymbol\mu_i)^T \Sigma_i^{-1} (\mathbf z_i-\boldsymbol\mu_i). $$

При нормальности \(s_i\sim\chi^2_2\).

## Частный случай для \(P\) и \(G\) ##

Два независимых GP, \(f_x(t)\) и \(f_y(t)\), дают диагональную ковариацию. Ненулевая связь координат требует multi-output GP. При batch-обучении GP видит весь набор наблюдений, а не только префикс до \(i\).

**Источники:**

- Smith M., Reece S., Roberts S., Psorakis I., Rezek I. Maritime Abnormality Detection Using Gaussian Processes // Knowledge and Information Systems. 2014. Vol. 38, No. 3. P. 717–741. DOI: [10.1007/s10115-013-0685-z](https://doi.org/10.1007/s10115-013-0685-z).
- Penacho Riveiros A., Bastianello N., Barreau M. Model-free Anomaly Detection for Dynamical Systems with Gaussian Processes. 2026. 6 p. [PDF](docs/methods/Model-free%20Anomaly%20Detection%20for.pdf).

# Hidden Markov Model #

Toloue и Jahan строят пять HMM с дискретными наблюдениями \((S,\Delta S,\mathrm{Drift},\Delta C)\): unexpected stop, risky speed, drift, spiral, normal. Модели обучаются Baum–Welch и сравниваются алгоритмом Forward на всей траектории.

Ниже — наша point-wise адаптация той же идеи, а не формула статьи: гауссовские HMM и отношение локальных predictive likelihood нормального и аномальных режимов.

$$ v_i=\frac{d_{\mathrm{geo}}\bigl((\mathrm{lat}_{i-1},\mathrm{lon}_{i-1}),(\mathrm{lat}_i,\mathrm{lon}_i)\bigr)}{t_i-t_{i-1}}, \qquad o_i=(v_i,\Delta v_i,\Delta\psi_i). $$

## Общая математическая постановка ##

$$ S_i\in\{1,\ldots,K\}, \qquad \pi_k=P(S_1=k), \qquad A_{jk}=P(S_i=k\mid S_{i-1}=j). $$

$$ o_i\mid S_i=k \sim \mathcal N(\mu_k,\Sigma_k). $$

$$ p(o_{1:N}\mid\theta) = \sum_{s_{1:N}} \pi_{s_1} \prod_{i=2}^{N}A_{s_{i-1}s_i} \prod_{i=1}^{N} p(o_i\mid S_i=s_i). $$

## Частный случай ##

Набор моделей \(\theta_N,\theta_{A_1},\ldots,\theta_{A_K}\). Point-wise score:

$$ s_i=\max_{k} \log \frac{P(o_i\mid o_{1:i-1},\theta_{A_k})}{P(o_i\mid o_{1:i-1},\theta_N)}, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Toloue K. F., Jahan M. V. Anomalous Behavior Detection of Marine Vessels Based on Hidden Markov Model // 2018 6th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2018. P. 10–12. [PDF](docs/methods/AnomalousBehaviorDetectionofMarineVessels.pdf).

# Hampel Filter #

В скользящем окне точка сравнивается с медианой соседей. Выброс — сильное отклонение относительно типичного разброса окна.

Roos-Hoefgeest Toribio et al. ускоряют фильтр, заменяя MAD оценкой mMAD. Ниже — стандартный Hampel, не эта замена. Центрированное окно \(\{x_{i-h},\ldots,x_{i+h}\}\) использует будущие точки. Для online-потока нужно хвостовое окно либо задержка решения на \(h\) отсчётов.

## Общая математическая постановка ##

$$ W_i=\{x_{i-h},\ldots,x_i,\ldots,x_{i+h}\}, \qquad m_i=\operatorname{median}(W_i), $$

$$ MAD_i=\operatorname{median}_{x\in W_i}|x-m_i|, \qquad s_i=\frac{|x_i-m_i|}{1.4826\,MAD_i+\varepsilon}. $$

Точка аномальна при \(s_i>\tau\).

## Частный случай ##

Фильтр применяется к скорости или ускорению, не к широте и долготе:

$$ s_i^{(v)}=\frac{|v_i-\operatorname{med}(v_{i-h:i+h})|}{1.4826\,MAD(v_{i-h:i+h})+\varepsilon}, \qquad a_i^{\mathrm{acc}}=\frac{v_i-v_{i-1}}{\Delta t_i}. $$

\(\hat a_i=\mathbf1[s_i^{(v)}>\tau_v]\). Метод рассчитан на единичные выбросы, не на длинный аномальный режим.

**Источник:** Roos-Hoefgeest Toribio M., Garnung Menéndez A., Roos-Hoefgeest Toribio S., Álvarez García I. A Novel Approach to Speed Up Hampel Filter for Outlier Detection // Sensors. 2025. Vol. 25. Article 3319. DOI: [10.3390/s25113319](https://doi.org/10.3390/s25113319). [PDF](docs/methods/A%20Novel%20Approach%20to%20Speed%20Up%20Hampel%20Filter%20for%20Outlier%20Detection.pdf).

# DBSCAN #

Точки группируются по плотности соседей. Аномалии — точки вне плотных кластеров. Это batch-метод: окрестность считается по всему набору, не по префиксу траектории.

## Общая математическая постановка ##

$$ N_\varepsilon(x)=\{x'\in X:d(x,x')\le\varepsilon\}. $$

Точка — core point при \(|N_\varepsilon(x)|\ge\mathrm{MinPts}\). Шум: \(c_i=-1\).

## Частный случай ##

Han et al. кластеризуют AIS по \((\mathrm{lat},\mathrm{lon},\mathrm{SOG},\mathrm{COG},\mathrm{Heading})\) с нормализацией и расстоянием Махаланобиса до кластера.

Наш вариант признаков для point-wise разметки:

$$ \mathbf f_i=\left( x_i^{\mathrm{proj}}, y_i^{\mathrm{proj}}, v_i, \Delta\psi_i \right), \qquad \mathbf u_i=\operatorname{scale}(\mathbf f_i). $$

$$ \hat a_i=\mathbf1[c_i=-1]. $$

**Источники:**

- Ester M., Kriegel H.-P., Sander J., Xu X. A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise // Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD-96). AAAI Press, 1996. P. 226–231. [PDF](docs/methods/A%20Density-Based%20Algorithm%20for%20Discovering%20Clusters.pdf).
- Han X., Armenakis C., Jadidi M. DBSCAN Optimization for Improving Marine Trajectory Clustering and Anomaly Detection // The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. 2020. Vol. XLIII-B4-2020. P. 455–461. DOI: [10.5194/isprs-archives-XLIII-B4-2020-455-2020](https://doi.org/10.5194/isprs-archives-XLIII-B4-2020-455-2020). [PDF](docs/methods/DBSCAN%20OPTIMIZATION%20FOR%20IMPROVING%20MARINE%20TRAJECTORY%20CLUSTERING.pdf).

# Local Outlier Factor #

Степень аномальности точки — насколько её локальная плотность ниже плотности ближайших соседей. LOF не является временной моделью: время входит только через признаки. Счёт идёт по всему набору точек.

## Общая математическая постановка ##

\(N_k(x)\) — \(k\) ближайших соседей.

$$ \operatorname{reachdist}_k(x,y)=\max\{k\text{-distance}(y),d(x,y)\}. $$

$$ \mathrm{lrd}_k(x)=\left( \frac{1}{|N_k(x)|} \sum_{y\in N_k(x)} \operatorname{reachdist}_k(x,y) \right)^{-1}. $$

$$ \mathrm{LOF}_k(x)=\frac{1}{|N_k(x)|} \sum_{y\in N_k(x)} \frac{\mathrm{lrd}_k(y)}{\mathrm{lrd}_k(x)}. $$

При \(\mathrm{LOF}_k(x)\approx 1\) плотность точки близка к соседям.

## Частный случай ##

$$ \mathbf f_i=\left( \Delta x_i, \Delta y_i, v_i, a_i^{\mathrm{acc}}, \Delta\psi_i \right), \qquad s_i=\mathrm{LOF}_k(\operatorname{scale}(\mathbf f_i)), \qquad \hat a_i=\mathbf1[s_i>\tau_{\mathrm{LOF}}]. $$

**Источник:** Breunig M. M., Kriegel H.-P., Ng R. T., Sander J. LOF: Identifying Density-Based Local Outliers // Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. Dallas, USA. ACM, 2000. P. 93–104. DOI: [10.1145/342009.335388](https://doi.org/10.1145/342009.335388). [PDF](<docs/methods/LOF_ Identifying Density-Based Local Outliers.pdf>).

# Isolation Forest #

Случайные деревья отделяют точки разрезами по признакам. Аномалия отделяется за меньшее число разрезов. Метод смотрит на весь набор и не требует отдельной выборки только нормальных точек.

## Общая математическая постановка ##

Средняя длина пути \(E[h(x)]\), нормировка через гармоническое число \(H\):

$$ c(n)=2H(n-1)-\frac{2(n-1)}{n}, \qquad s(x)=2^{-\frac{E[h(x)]}{c(n)}}. $$

## Частный случай ##

$$ \mathbf f_i=( x_i^{\mathrm{proj}}, y_i^{\mathrm{proj}}, v_i, a_i^{\mathrm{acc}}, \Delta\psi_i ), \qquad s_i=s(\mathbf f_i), \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

Здесь \(s(\mathbf f_i)\) — нормированный path-length score из формулы выше.

**Источник:** Liu F. T., Ting K. M., Zhou Z.-H. Isolation-Based Anomaly Detection // ACM Transactions on Knowledge Discovery from Data. 2012. Vol. 6, No. 1. Article 3. 39 p. DOI: [10.1145/2133360.2133363](https://doi.org/10.1145/2133360.2133363). [PDF](docs/methods/Isolation-Based%20Anomaly%20Detection.pdf).

# One-Class SVM #

На нормальных точках оценивается область высокой концентрации обучающего распределения. Объект вне этой области считается аномалией.

## Общая математическая постановка ##

$$ \min_{\mathbf w,\rho,\xi} \frac12\|\mathbf w\|^2 + \frac{1}{\nu n} \sum_{i=1}^{n}\xi_i -\rho, \qquad \mathbf w^T\phi(x_i)\ge\rho-\xi_i, \quad \xi_i\ge0. $$

Параметр \(\nu\in(0,1]\) — верхняя граница доли ошибок на обучении и нижняя граница доли support vectors.

$$ f(x)=\sum_i\alpha_i K(x_i,x)-\rho. $$

Аномалия при \(f(x)<0\).

## Частный случай ##

Обучение только на \(\mathcal D_N=\{\mathbf f_i:y_i>0\}\),

$$ \mathbf f_i=( x_i^{\mathrm{proj}}, y_i^{\mathrm{proj}}, v_i, a_i^{\mathrm{acc}}, \Delta\psi_i ), \qquad s_i=-f(\mathbf f_i), \qquad \hat a_i=\mathbf1[s_i>0]. $$

**Источник:** Schölkopf B., Platt J. C., Shawe-Taylor J., Smola A. J., Williamson R. C. Estimating the Support of a High-Dimensional Distribution // Neural Computation. 2001. Vol. 13, No. 7. P. 1443–1471. DOI: [10.1162/089976601750264965](https://doi.org/10.1162/089976601750264965). В `docs/methods/` лежит технический отчёт Microsoft Research MSR-TR-99-87 (27 November 1999; revised 18 September 2000): [PDF](docs/methods/Estimating%20the%20Support%20of%20a%20High-Dimensional%20Distribution.pdf).

# Robust PCA / Principal Component Pursuit #

Матрица раскладывается на низкоранговую часть и разреженные отклонения. Крупные элементы разреженной части — кандидаты в аномалии. Это batch-разложение матрицы, не causal-прогноз точки.

## Общая математическая постановка ##

$$ X=L+S, \qquad \min_{L,S} \|L\|_* + \lambda\|S\|_1. $$

В классической постановке Candès et al. \(\lambda=1/\sqrt{\max(n_1,n_2)}\).

## Частный случай ##

Одна последовательность \((t_i,\mathrm{lat}_i,\mathrm{lon}_i)\) в PCP не подаётся. Наша адаптация: матрица окон признаков, затем агрегация по окнам, содержащим точку \(i\):

$$ s_i=\frac{1}{|W_i|}\sum_{W:\, i\in W} \|S_W(i)\|, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Candès E. J., Li X., Ma Y., Wright J. Robust Principal Component Analysis? // Journal of the ACM. 2011. Vol. 58, No. 3. Article 11. P. 1–37. DOI: [10.1145/1970392.1970395](https://doi.org/10.1145/1970392.1970395). В `docs/methods/` лежит препринт arXiv:0912.3599v1 (18 December 2009): [PDF](docs/methods/Robust%20Principal%20Component%20Analysis.pdf).

# LSTM Autoencoder / EncDec-AD #

Сеть учится восстанавливать окна нормальной траектории. Аномалия — окно с нетипичной ошибкой восстановления. Окно не является causal-прогнозом одной точки.

## Общая математическая постановка ##

$$ X_i=(x_{i-L+1},\ldots,x_i), \qquad h_i=E_\theta(X_i), \qquad \hat X_i=D_\phi(h_i). $$

Обучение: \(\min_{\theta,\phi}\sum_i \|X_i-\hat X_i\|_2^2\).

В EncDec-AD Malhotra et al. вектор ошибки \(e_i=X_i-\hat X_i\) на нормальной проверочной выборке описывается средним \(\mu\) и ковариацией \(\Sigma\), score — расстояние Махаланобиса:

$$ s_i=(e_i-\mu)^T\Sigma^{-1}(e_i-\mu). $$

Сырой \(\|e_i\|_2^2\) — упрощение без этой нормировки.

## Частный случай ##

Окна координат траектории. Геодезическая ошибка только конечной точки окна — дальнейшее упрощение point-wise оценки, не формула статьи.

**Источник:** Malhotra P., Ramakrishnan A., Anand G., Vig L., Agarwal P., Shroff G. LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection // ICML 2016 Anomaly Detection Workshop. New York, 2016. arXiv: [1607.00148](https://arxiv.org/abs/1607.00148). [PDF](docs/methods/LSTM-based%20Encoder-Decoder%20for%20Multi-sensor%20Anomaly%20Detection.pdf).

# Temporal Convolutional Network #

Каузальная свёрточная сеть предсказывает продолжение нормального ряда. Аномалия — остаток, редкий для распределения ошибок на нормальных данных.

He и Zhao прогнозируют несколько шагов вперёд и смешивают признаки нескольких масштабов. Один шаг — частный случай.

## Общая математическая постановка ##

$$ \hat x_{t+1:t+H}=f_\theta(x_{t-L+1},\ldots,x_t), \qquad r_{t+h}=x_{t+h}-\hat x_{t+h}. $$

Остатки на нормальных данных: \(r_t\sim\mathcal N(\mu_r,\Sigma_r)\). Нулевое среднее — дополнительное допущение.

$$ s_t=(r_t-\mu_r)^T\Sigma_r^{-1}(r_t-\mu_r). $$

## Частный случай ##

$$ \mathbf z_i=(\mathrm{lat}_i,\mathrm{lon}_i)^T, \qquad \hat{\mathbf z}_i=f_\theta(\mathbf z_{i-L},\ldots,\mathbf z_{i-1}), \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** He Y., Zhao J. Temporal Convolutional Networks for Anomaly Detection in Time Series // Journal of Physics: Conference Series. 2019. Vol. 1213. Article 042050. DOI: [10.1088/1742-6596/1213/4/042050](https://doi.org/10.1088/1742-6596/1213/4/042050). [PDF](docs/methods/Temporal_Convolutional_Networks_for_Anomaly_Detect.pdf).

# DONUT — VAE для anomaly detection #

Вариационный автоэнкодер задаёт распределение окон. Окно с низкой вероятностью считается аномальным. DONUT сделан для сезонных KPI веб-приложений: modified ELBO, учёт пропусков и KDE-интерпретация реконструкции. Формулы ниже — общая VAE-постановка; перенос на окна траектории — наша адаптация, не алгоритм статьи.

## Общая математическая постановка ##

$$ q_\phi(z|x)=\mathcal N\left( \mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)) \right), \qquad p_\theta(x|z). $$

$$ \mathcal L(x)=\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}\left( q_\phi(z|x)\|p(z) \right). $$

Общий probabilistic score \(s_i=-\log p_\theta(x_i)\) не является точной формулой DONUT.

## Частный случай ##

Окна \((t,\mathrm{lat},\mathrm{lon})\). Если \(s_i>\tau\), то \(\hat a_i=1\).

**Источник:** Xu H., Chen W., Zhao N., Li Z., Bu J., Li Z., Liu Y., Zhao Y., Pei D., Feng Y., Chen J., Wang Z., Qiao H. Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications // Proceedings of The Web Conference 2018 (WWW '18). Lyon, France, 23–27 April 2018. ACM, 2018. 12 p. DOI: [10.1145/3178876.3185996](https://doi.org/10.1145/3178876.3185996). arXiv: [1802.03903](https://arxiv.org/abs/1802.03903). [PDF](docs/methods/Unsupervised%20Anomaly%20Detection%20via%20Variational%20Auto-Encoder.pdf).

# Anomaly Transformer #

Для окна последовательности модель совмещает ошибку восстановления и расхождение ассоциаций attention с prior-ассоциацией. Из-за adjacent-concentration bias аномалия часто имеет меньшую association discrepancy, поэтому в итоговом score стоит \(\mathrm{Softmax}(-\mathrm{AssDis})\), а не само расхождение.

## Общая математическая постановка ##

Окно \(X=(x_{1},\ldots,x_{L_{\mathrm{win}}})\). В слое \(\ell\) ассоциации

$$ A^{(\ell)}_{ij}=\operatorname{softmax}\left( \frac{Q_iK_j^T}{\sqrt d} \right) $$

сравниваются с prior-ассоциацией \(P^{(\ell)}\). \(L\) — число слоёв, не длина окна. Для позиции \(i\)

$$ \mathrm{AssDis}_i=\frac{1}{L}\sum_{\ell=1}^{L}\Bigl[\mathrm{KL}(A^{(\ell)}_i\|P^{(\ell)}_i)+\mathrm{KL}(P^{(\ell)}_i\|A^{(\ell)}_i)\Bigr]. $$

Итоговый point-wise критерий статьи:

$$ \mathrm{AnomalyScore}(X)=\mathrm{Softmax}(-\mathrm{AssDis})\odot \|X-\hat X\|_2^2. $$

## Частный случай ##

На вход подаётся \((\mathrm{lat}_i,\mathrm{lon}_i,v_i,\Delta\psi_i)\) либо \((t_i,\mathrm{lat}_i,\mathrm{lon}_i)\). \(\hat a_i=\mathbf1[s_i>\tau]\), где \(s_i\) — компонента \(\mathrm{AnomalyScore}\).

**Источник:** Xu J., Wu H., Wang J., Long M. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy // International Conference on Learning Representations (ICLR). 2022. arXiv: [2110.02642](https://arxiv.org/abs/2110.02642). [PDF](docs/methods/ANOMALY%20TRANSFORMER%20TIME%20SERIES%20ANOMALY.pdf).

# Spatio-Temporal GNN — STGVAD #

Состояния нескольких судов становятся вершинами одного графа. Сеть предсказывает нормальное продолжение; большое отклонение считается аномалией. Для оценки авторы вручную внесли аномалии в реальный AIS.

Пространственные связи задаются кластерами OPTICS, а не одним порогом \(d_{\mathrm{geo}}<\varepsilon\). Временные связи внутри скользящего окна соединяют состояния не только с непосредственным соседом; затем multi-ship граф связывает близкие суда.

## Общая математическая постановка ##

Вершина — состояние судна, не скорость:

$$ \mathbf u_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i,\mathrm{SOG}_i,\mathrm{COG}_i). $$

$$ G=(V,E), \qquad E=E_{\mathrm{time}}\cup E_{\mathrm{space}}. $$

Message passing:

$$ h_i^{(l+1)}=\sigma\left( W_0h_i^{(l)}+\sum_{j\in N(i)}\alpha_{ij}W_1h_j^{(l)} \right), \qquad H=\mathrm{GNN}_\theta(G,X). $$

$$ \hat x_i=g_\phi(H_{\le i}), \qquad s_i=d(x_i,\hat x_i). $$

## Частный случай ##

$$ s_i=d_{\mathrm{geo}}\left( (\mathrm{lat}_i,\mathrm{lon}_i), (\widehat{\mathrm{lat}}_i,\widehat{\mathrm{lon}}_i) \right), \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Kim J., Kim M., Hwang Y., Bae S., Cho D. J., Lee W., Park H. STGVAD: Spatio-Temporal Graph-based Vessel Behavior Anomaly Detection // IEEE Access. 2026. Vol. 14. P. 2152–2165. DOI: [10.1109/ACCESS.2025.3609783](https://doi.org/10.1109/ACCESS.2025.3609783) (принята в 2025). [PDF](docs/methods/STGVAD_Spatio-Temporal_Graph-Based_Vessel_Behavior.pdf).

# Maximum Physically Consistent Trajectory #

Из траектории выбирается самая длинная подпоследовательность, совместимая с моделью движения. Точки вне неё — кандидаты на выбросы: алгоритм проверяет физическую совместимость, а не доказывает, что каждое исключённое измерение ошибочно.

## Общая математическая постановка ##

В speed-bounded модели Custers et al. ограничена максимальная скорость. Нижняя граница \(v_-\) — дополнительное ограничение, не формула этой модели. Отдельно в статье рассмотрены модели с ограниченным ускорением.

$$ v(p_i,p_j)=\frac{d(p_i,p_j)}{t_j-t_i}, \qquad v(p_i,p_j)\le v_+. $$

$$ Q^\star=\arg\max_{Q\subseteq\{1,\ldots,N\}} |Q| $$

при согласованности всех последовательных элементов \(Q\).

## Частный случай для \(P\) и \(G\) ##

$$ v_{ij}=\frac{d_{\mathrm{geo}}\left( (\mathrm{lat}_i,\mathrm{lon}_i), (\mathrm{lat}_j,\mathrm{lon}_j) \right)}{t_j-t_i}, \qquad \hat a_i=\mathbf1[i\notin Q^\star(P)]. $$

**Источник:** Custers B., van de Kerkhof M., Meulemans W., Speckmann B., Staals F. Maximum Physically Consistent Trajectories // ACM Transactions on Spatial Algorithms and Systems. 2021. Vol. 7, No. 4. Article 17. 33 p. DOI: [10.1145/3452378](https://doi.org/10.1145/3452378). [PDF](docs/methods/Maximum%20Physically%20Consistent%20Trajectories.pdf).
