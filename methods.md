# Общие тезисы #

Наблюдаемая траектория судна представляется как последовательность точек:

$$ X=\{x_i\}_{i=1}^{N}, \qquad x_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i). $$

Модель \(M\), обученная на некотором наборе нормальных данных \(X_{\mathrm{train}}\), вычисляет для каждой точки anomaly score

$$ s_i^{(M)} = M(X_{\le i}), $$

после чего вводится порог \(\tau_M\):

$$ \hat a_i^{(M)} = \mathbf1[s_i^{(M)}>\tau_M]. $$

Предсказание можно привести к исходной схеме меток:

$$ \hat y_i^{(M)} = \begin{cases} 0, & s_i^{(M)}>\tau_M,\\ 1, & s_i^{(M)}\le\tau_M. \end{cases} $$

При наличии нескольких нормальных классов \(y_i>0\) конкретный положительный класс уже является отдельной задачей классификации. Для оценки бинарного обнаружения используются

$$ a_i=\mathbf1[y_i=0], \qquad \hat a_i=\mathbf1[\hat y_i=0]. $$

# Авторегрессионная модель ARIMA(2,1,1) #

Модель по нескольким предыдущим значениям строит прогноз следующего и помечает точку как аномальную, если фактическое измерение сильно от него отклоняется.

Для одномерного временного ряда \(x_t\) модель ARIMA\((p,d,q)\) задаётся как

$$ \phi(B)(1-B)^d x_t = c+\theta(B)\varepsilon_t, \qquad \varepsilon_t\sim\mathcal N(0,\sigma^2), $$

где \(B\) — оператор запаздывания,

$$ \phi(B)=1-\phi_1B-\ldots-\phi_pB^p, \qquad \theta(B)=1+\theta_1B+\ldots+\theta_qB^q. $$

В качестве конкретной модели выбираем ARIMA\((2,1,1)\):

$$ \Delta x_t = c+\phi_1\Delta x_{t-1}+\phi_2\Delta x_{t-2}+\varepsilon_t+\theta_1\varepsilon_{t-1}. $$

Аномальность измерения определяется через ошибку прогноза:

$$ r_t=x_t-\hat x_t. $$

При известной дисперсии прогноза

$$ s_t=\frac{|r_t|}{\sigma_t}. $$

Аномалия:

$$ s_t>\tau. $$

## Частный случай для \(P\) и \(G\) ##

Для траектории судна строятся две модели:

$$ \mathrm{lat}_i \sim \mathrm{ARIMA}(2,1,1), \qquad \mathrm{lon}_i \sim \mathrm{ARIMA}(2,1,1). $$

Получаем

$$ \hat{\mathbf z}_i = \begin{pmatrix} \widehat{\mathrm{lat}}_i\\ \widehat{\mathrm{lon}}_i \end{pmatrix}, \qquad \mathbf z_i = \begin{pmatrix} \mathrm{lat}_i\\ \mathrm{lon}_i \end{pmatrix}. $$

Ошибка:

$$ r_i=\mathbf z_i-\hat{\mathbf z}_i. $$

Для географических координат предпочтительнее использовать геодезическую дистанцию

$$ s_i=d_{\mathrm{geo}}(\mathbf z_i,\hat{\mathbf z}_i) $$

либо локальную метрическую проекцию.

Итоговая оценка:

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

Для ARIMA фактическая аномальная метка \(y_i=0\) в процессе построения прогноза не нужна.

**Источник:** Qin Yu, Lyu Jibin, Lirui Jiang. An Improved ARIMA-Based Traffic Anomaly Detection Algorithm for Wireless Sensor Networks // International Journal of Distributed Sensor Networks. 2016. Vol. 2016. Article ID 9653230. 9 p. DOI: [10.1155/2016/9653230](https://doi.org/10.1155/2016/9653230). [PDF](docs/methods/An%20Improved%20ARIMA-Based%20Traffic%20Anomaly%20Detection%20Algorithm%20for%20Wireless%20Sensor%20Networks.pdf).

# Калмановский фильтр с моделью постоянной скорости #

Здесь вместо общего KF выбираем конкретную модель состояния.

Фильтр ведёт скрытое состояние «положение и скорость» и считает измерение аномальным, если оно плохо согласуется с предсказанным продолжением движения.

## Общая математическая постановка ##

Состояние:

$$ \mathbf x_i= \begin{pmatrix} p_x\\ p_y\\ v_x\\ v_y \end{pmatrix}_i. $$

Модель движения:

$$ \mathbf x_i=A_i\mathbf x_{i-1}+\mathbf w_i, \qquad \mathbf w_i\sim\mathcal N(0,Q_i), $$

где

$$ A_i= \begin{pmatrix} 1&0&\Delta t_i&0\\ 0&1&0&\Delta t_i\\ 0&0&1&0\\ 0&0&0&1 \end{pmatrix}. $$

Измерение:

$$ \mathbf z_i=H\mathbf x_i+\mathbf v_i, \qquad \mathbf v_i\sim\mathcal N(0,R), \qquad H= \begin{pmatrix} 1&0&0&0\\ 0&1&0&0 \end{pmatrix}. $$

Инновация:

$$ \mathbf r_i=\mathbf z_i-H\hat{\mathbf x}_{i|i-1}. $$

Ковариация инновации:

$$ S_i=HP_{i|i-1}H^T+R. $$

Естественный статистический anomaly score — normalized innovation squared:

$$ s_i=\mathbf r_i^T S_i^{-1}\mathbf r_i. $$

При гауссовских предположениях

$$ s_i\sim\chi^2_2 $$

при отсутствии аномалии, поэтому

$$ s_i>\chi^2_{2,\,1-\alpha} $$

означает аномальное измерение.

## Частный случай для \(P\) и \(G\) ##

Из

$$ x_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i) $$

формируем

$$ \mathbf z_i= \begin{pmatrix} \mathrm{lat}_i\\ \mathrm{lon}_i \end{pmatrix}. $$

Скорости \(v_x,v_y\) являются скрытой частью состояния и оцениваются фильтром.

Затем

$$ \hat a_i=\mathbf1[s_i>\tau], \qquad \hat y_i= \begin{cases} 0,&s_i>\tau,\\ 1,&s_i\le\tau. \end{cases} $$

Такой вариант непосредственно соответствует ситуации, когда физически правдоподобная траектория имеет небольшие локальные ошибки измерений, а настоящий скачок вызывает большую инновацию.

**Источник:** Kalman R. E. A New Approach to Linear Filtering and Prediction Problems // Journal of Basic Engineering. 1960. Vol. 82, No. 1. P. 35–45. DOI: [10.1115/1.3662552](https://doi.org/10.1115/1.3662552).

# Bootstrap Particle Filter / SIR #

Вместо одного прогноза держится набор гипотез о движении. Аномалия — измерение, которое маловероятно сразу для всех этих гипотез.

## Общая математическая постановка ##

Состояние:

$$ \mathbf x_i\sim p(\mathbf x_i\mid\mathbf x_{i-1}). $$

Представим апостериорное распределение частицами:

$$ p(\mathbf x_i\mid z_{1:i}) \approx \sum_{j=1}^{K}w_i^{(j)} \delta(\mathbf x_i-\mathbf x_i^{(j)}). $$

Предсказание:

$$ \mathbf x_i^{(j)} \sim p(\mathbf x_i\mid\mathbf x_{i-1}^{(j)}). $$

Веса:

$$ \tilde w_i^{(j)} = w_{i-1}^{(j)} p(\mathbf z_i\mid\mathbf x_i^{(j)}), \qquad w_i^{(j)} = \frac{\tilde w_i^{(j)}}{\sum_{k=1}^{K}\tilde w_i^{(k)}}. $$

Предсказательная вероятность измерения:

$$ p(\mathbf z_i\mid z_{1:i-1}) \approx \sum_{j=1}^{K} w_{i-1}^{(j)} p(\mathbf z_i\mid\mathbf x_i^{(j)}). $$

Поэтому естественный score:

$$ s_i=-\log p(\mathbf z_i\mid z_{1:i-1}). $$

Большое значение \(s_i\) соответствует маловероятному измерению.

## Частный случай для \(P\) и \(G\) ##

Используем ту же модель движения постоянной скорости, что и в KF:

$$ \mathbf x_i=(\mathrm{lat}_i,\mathrm{lon}_i,v_{\mathrm{lat},i},v_{\mathrm{lon},i})^T, $$

но вместо единственного гауссовского состояния поддерживаем \(K\) гипотез движения.

Получаем

$$ s_i=-\log \left( \sum_{j=1}^{K} w_{i-1}^{(j)} p\bigl( (\mathrm{lat}_i,\mathrm{lon}_i) \mid \mathbf x_i^{(j)} \bigr) \right). $$

После пороговой операции:

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

Преимущество по сравнению с KF заключается в том, что распределение состояний не обязано быть одним гауссовским облаком. Это потенциально полезно для поворотов, развилок и других мультимодальных движений.

**Источник:** Arulampalam M. S., Maskell S., Gordon N., Clapp T. A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking // IEEE Transactions on Signal Processing. 2002. Vol. 50, No. 2. P. 174–188. DOI: [10.1109/78.978374](https://doi.org/10.1109/78.978374). [PDF](docs/methods/A%20Tutorial%20on%20Particle%20Filters%20for%20Online.pdf).

# Gaussian Process Regression с RBF-ядром #

Нормальная траектория описывается гладкой функцией времени вместе с оценкой неопределённости. Точка аномальна, если лежит далеко от этой функции относительно предсказанного разброса.

## Общая математическая постановка ##

Пусть

$$ f(t)\sim GP(m(t),k(t,t')), $$

где

$$ k(t,t') = \sigma_f^2 \exp \left( -\frac{(t-t')^2}{2\ell^2} \right). $$

Наблюдение:

$$ y_i=f(t_i)+\varepsilon_i, \qquad \varepsilon_i\sim\mathcal N(0,\sigma_n^2). $$

Для нового момента времени \(t_*\):

$$ f(t_*)\mid\mathcal D \sim \mathcal N(\mu_*,\sigma_*^2). $$

Если модель многомерная и прогнозируется вектор

$$ \mathbf z_i= \begin{pmatrix} \mathrm{lat}_i\\ \mathrm{lon}_i \end{pmatrix}, $$

то anomaly score естественно определить как

$$ s_i=(\mathbf z_i-\boldsymbol\mu_i)^T \Sigma_i^{-1} (\mathbf z_i-\boldsymbol\mu_i). $$

При нормальности:

$$ s_i\sim\chi^2_2. $$

## Частный случай для \(P\) и \(G\) ##

Строятся два GP:

$$ f_{\mathrm{lat}}(t), \qquad f_{\mathrm{lon}}(t). $$

Для точки \(i\):

$$ \boldsymbol\mu_i= \begin{pmatrix} \mu_{\mathrm{lat}}(t_i)\\ \mu_{\mathrm{lon}}(t_i) \end{pmatrix}. $$

Из \(P\) берём наблюдаемую координату

$$ \mathbf z_i= \begin{pmatrix} \mathrm{lat}_i\\ \mathrm{lon}_i \end{pmatrix}. $$

Тогда

$$ \hat a_i=\mathbf1\left[ (\mathbf z_i-\boldsymbol\mu_i)^T \Sigma_i^{-1} (\mathbf z_i-\boldsymbol\mu_i) >\tau \right]. $$

Этот подход особенно интересен для вашей задачи, потому что GP непосредственно моделирует неопределённость прогноза, а не только одну ожидаемую координату.

**Источники:**

- Smith M., Reece S., Roberts S., Psorakis I., Rezek I. Maritime Abnormality Detection Using Gaussian Processes // Knowledge and Information Systems. 2014. Vol. 38, No. 3. P. 717–741. DOI: [10.1007/s10115-013-0685-z](https://doi.org/10.1007/s10115-013-0685-z).
- Penacho Riveiros A., Bastianello N., Barreau M. Model-free Anomaly Detection for Dynamical Systems with Gaussian Processes. 2026. 6 p. [PDF](docs/methods/Model-free%20Anomaly%20Detection%20for.pdf).

# Hidden Markov Model #

В качестве объекта наблюдения удобно использовать признаки движения:

$$ o_i=(v_i,\Delta v_i,\Delta\psi_i), $$

где

$$ v_i=\frac{d_{\mathrm{geo}}(x_{i-1},x_i)}{\Delta t_i}. $$

Поведение судна представляется как переключение скрытых режимов движения. Точка аномальна, если текущее наблюдение лучше объясняется аномальным режимом, чем нормальным.

## Общая математическая постановка ##

Пусть

$$ S_i\in\{1,\ldots,K\}. $$

Начальное распределение:

$$ \pi_k=P(S_1=k). $$

Переходы:

$$ A_{jk}=P(S_i=k\mid S_{i-1}=j). $$

Для гауссовского HMM:

$$ o_i\mid S_i=k \sim \mathcal N(\mu_k,\Sigma_k). $$

Вероятность наблюдаемой последовательности:

$$ p(o_{1:N}\mid\theta) = \sum_{s_{1:N}} \pi_{s_1} \prod_{i=2}^{N}A_{s_{i-1}s_i} \prod_{i=1}^{N} p(o_i\mid S_i=s_i). $$

## Частный случай ##

Для каждой траектории

$$ P=\{(t_i,\mathrm{lat}_i,\mathrm{lon}_i,\hat y_i)\} $$

строится последовательность

$$ O(P)=\{o_i\}_{i=1}^{N}. $$

Один из вариантов конкретной модели — набор HMM для классов поведения:

$$ \theta_N,\theta_{A_1},\ldots,\theta_{A_K}, $$

где \(\theta_N\) соответствует нормальному движению, а остальные модели — различным аномальным режимам.

Для point-wise оценки удобно использовать отношение предсказательных вероятностей:

$$ s_i=\max_{k} \log \frac{P(o_i\mid o_{1:i-1},\theta_{A_k})}{P(o_i\mid o_{1:i-1},\theta_N)}. $$

Тогда

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

Это point-wise специализация исходной trajectory-level HMM-схемы. Она позволяет привести метод к вашей разметке \(y_i=0\) / \(y_i>0\).

Для вашей задачи этот метод особенно интересен для аномалий типа «спираль», изменение режима движения, неожиданные остановки и т. п., поскольку HMM моделирует не только отдельные значения, но и переходы между состояниями.

**Источник:** Toloue K. F., Jahan M. V. Anomalous Behavior Detection of Marine Vessels Based on Hidden Markov Model // 2018 6th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2018. P. 10–12. [PDF](docs/methods/AnomalousBehaviorDetectionofMarineVessels.pdf).

# Hampel Filter #

В скользящем окне точка сравнивается с медианой соседей. Выбросом считается сильное отклонение относительно типичного разброса в этом окне.

## Общая математическая постановка ##

Для окна

$$ W_i=\{x_{i-h},\ldots,x_i,\ldots,x_{i+h}\} $$

медиана:

$$ m_i=\operatorname{median}(W_i). $$

Медианное абсолютное отклонение:

$$ MAD_i=\operatorname{median}_{x\in W_i}|x-m_i|. $$

Стандартизованное отклонение:

$$ s_i=\frac{|x_i-m_i|}{1.4826\,MAD_i+\varepsilon}. $$

Точка аномальна:

$$ s_i>\tau. $$

## Частный случай ##

Применим фильтр к производным траектории, например к

$$ x_i=v_i $$

или

$$ x_i=a_i=\frac{v_i-v_{i-1}}{\Delta t_i}, $$

а не просто к широте и долготе.

Например,

$$ s_i^{(v)}=\frac{|v_i-\operatorname{med}(v_{i-h:i+h})|}{1.4826\,MAD(v_{i-h:i+h})+\varepsilon}. $$

Тогда

$$ \hat a_i=\mathbf1[s_i^{(v)}>\tau_v]. $$

В этом виде метод хорошо подходит для единичных выбросов, но не предназначен для длительных аномальных сегментов, которые сами образуют устойчивый локальный режим.

**Источник:** Roos-Hoefgeest Toribio M., Garnung Menéndez A., Roos-Hoefgeest Toribio S., Álvarez García I. A Novel Approach to Speed Up Hampel Filter for Outlier Detection // Sensors. 2025. Vol. 25. Article 3319. DOI: [10.3390/s25113319](https://doi.org/10.3390/s25113319). [PDF](docs/methods/A%20Novel%20Approach%20to%20Speed%20Up%20Hampel%20Filter%20for%20Outlier%20Detection.pdf).

# DBSCAN #

Точки группируются по плотности соседей. Аномалиями считаются те, которые не попали ни в один плотный кластер.

## Общая математическая постановка ##

Для множества объектов \(X\) и метрики \(d\) определяется \(\varepsilon\)-окрестность:

$$ N_\varepsilon(x)=\{x'\in X:d(x,x')\le\varepsilon\}. $$

Точка является core point, если

$$ |N_\varepsilon(x)|\ge\mathrm{MinPts}. $$

Точка, не относящаяся ни к одному кластеру, получает статус noise:

$$ c_i=-1. $$

## Частный случай ##

Для каждой точки формируется вектор движения, например

$$ \mathbf f_i=\left( x_i^{\mathrm{proj}}, y_i^{\mathrm{proj}}, v_i, \Delta\psi_i \right). $$

После нормализации признаков:

$$ \mathbf u_i=\operatorname{scale}(\mathbf f_i). $$

Запускаем

$$ \mathrm{DBSCAN}(\mathbf u_i,\varepsilon,\mathrm{MinPts}). $$

Тогда

$$ \hat a_i=\mathbf1[c_i=-1]. $$

То есть

$$ \hat y_i= \begin{cases} 0,&c_i=-1,\\ 1,&c_i\ne-1. \end{cases} $$

Для морских траекторий есть непосредственно соответствующее исследование, где DBSCAN применяется к AIS с пространственными и динамическими признаками.

**Источники:**

- Ester M., Kriegel H.-P., Sander J., Xu X. A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise // Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD-96). AAAI Press, 1996. P. 226–231. [PDF](docs/methods/A%20Density-Based%20Algorithm%20for%20Discovering%20Clusters.pdf).
- Han X., Armenakis C., Jadidi M. DBSCAN Optimization for Improving Marine Trajectory Clustering and Anomaly Detection // The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. 2020. Vol. XLIII-B4-2020. P. 455–461. DOI: [10.5194/isprs-archives-XLIII-B4-2020-455-2020](https://doi.org/10.5194/isprs-archives-XLIII-B4-2020-455-2020). [PDF](docs/methods/DBSCAN%20OPTIMIZATION%20FOR%20IMPROVING%20MARINE%20TRAJECTORY%20CLUSTERING.pdf).

# Local Outlier Factor #

Степень аномальности точки — насколько её локальная плотность ниже плотности ближайших соседей.

## Общая математическая постановка ##

Пусть \(N_k(x)\) — множество \(k\) ближайших соседей.

Reachability distance:

$$ \operatorname{reachdist}_k(x,y)=\max\{k\text{-distance}(y),d(x,y)\}. $$

Локальная плотность:

$$ \mathrm{lrd}_k(x)=\left( \frac{1}{|N_k(x)|} \sum_{y\in N_k(x)} \operatorname{reachdist}_k(x,y) \right)^{-1}. $$

LOF:

$$ \mathrm{LOF}_k(x)=\frac{1}{|N_k(x)|} \sum_{y\in N_k(x)} \frac{\mathrm{lrd}_k(y)}{\mathrm{lrd}_k(x)}. $$

При

$$ \mathrm{LOF}_k(x)\approx 1 $$

плотность точки соответствует соседям; большие значения указывают на выброс.

## Частный случай ##

Формируем

$$ \mathbf f_i=\left( \Delta x_i, \Delta y_i, v_i, a_i, \Delta\psi_i \right). $$

После нормализации:

$$ \mathbf u_i=\operatorname{scale}(\mathbf f_i). $$

Тогда

$$ s_i=\mathrm{LOF}_k(\mathbf u_i) $$

и

$$ \hat a_i=\mathbf1[s_i>\tau_{\mathrm{LOF}}]. $$

LOF особенно полезен в ситуации, когда «аномальность» определяется не глобальной редкостью, а отличием от локального режима.

**Источник:** Breunig M. M., Kriegel H.-P., Ng R. T., Sander J. LOF: Identifying Density-Based Local Outliers // Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. Dallas, USA. ACM, 2000. P. 93–104. DOI: [10.1145/342009.335388](https://doi.org/10.1145/342009.335388). [PDF](<docs/methods/LOF_ Identifying Density-Based Local Outliers.pdf>).

# Isolation Forest #

Случайные деревья отделяют точки разрезами по признакам. Аномалия отделяется за меньшее число разрезов, чем обычная точка.

## Общая математическая постановка ##

Строится множество случайных isolation trees.

Для точки \(x\) определяется средняя длина пути:

$$ E[h(x)]. $$

Нормировочный коэффициент:

$$ c(n)=2H(n-1)-\frac{2(n-1)}{n}, $$

где \(H(\cdot)\) — гармоническое число.

Anomaly score:

$$ s(x)=2^{-\frac{E[h(x)]}{c(n)}}. $$

Чем меньше путь, тем более изолирована точка и тем выше \(s(x)\).

## Частный случай ##

Используем

$$ \mathbf f_i=( x_i^{\mathrm{proj}}, y_i^{\mathrm{proj}}, v_i, a_i, \Delta\psi_i ). $$

Для каждой точки:

$$ s_i=\mathrm{IF}(\mathbf f_i). $$

Затем

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

Метод не требует явного построения модели нормального движения и поэтому хорошо подходит как baseline для большой выборки.

**Источник:** Liu F. T., Ting K. M., Zhou Z.-H. Isolation-Based Anomaly Detection // ACM Transactions on Knowledge Discovery from Data. 2012. Vol. 6, No. 1. Article 3. 39 p. DOI: [10.1145/2133360.2133363](https://doi.org/10.1145/2133360.2133363). [PDF](docs/methods/Isolation-Based%20Anomaly%20Detection.pdf).

# One-Class SVM #

На нормальных данных строится граница области типичных значений. Всё, что оказывается снаружи этой границы, считается аномалией.

## Общая математическая постановка ##

Решается задача

$$ \min_{\mathbf w,\rho,\xi} \frac12\|\mathbf w\|^2 + \frac{1}{\nu n} \sum_{i=1}^{n}\xi_i -\rho $$

при ограничениях

$$ \mathbf w^T\phi(x_i)\ge\rho-\xi_i, \qquad \xi_i\ge0. $$

В kernel formulation:

$$ f(x)=\sum_i\alpha_i K(x_i,x)-\rho. $$

Объект считается аномальным, если

$$ f(x)<0. $$

## Частный случай ##

Обучение выполняется только на нормальных точках:

$$ \mathcal D_N=\{\mathbf f_i:y_i>0\}. $$

Признак:

$$ \mathbf f_i=( x_i^{\mathrm{proj}}, y_i^{\mathrm{proj}}, v_i, a_i, \Delta\psi_i ). $$

Тогда

$$ s_i=-f(\mathbf f_i) $$

и

$$ \hat a_i=\mathbf1[s_i>0]. $$

Иными словами, модель учится описывать support нормального распределения, а всё, что оказывается за его границей, считается аномальным.

**Источник:** Schölkopf B., Platt J. C., Shawe-Taylor J., Smola A. J., Williamson R. C. Estimating the Support of a High-Dimensional Distribution // Neural Computation. 2001. Vol. 13, No. 7. P. 1443–1471. DOI: [10.1162/089976601750264965](https://doi.org/10.1162/089976601750264965). В `docs/methods/` лежит технический отчёт Microsoft Research MSR-TR-99-87 (27 November 1999; revised 18 September 2000): [PDF](docs/methods/Estimating%20the%20Support%20of%20a%20High-Dimensional%20Distribution.pdf).

# Robust PCA / Principal Component Pursuit #

Матрица наблюдений раскладывается на низкоранговую типичную часть и разреженные отклонения. Крупные элементы разреженной части — кандидаты в аномалии.

## Общая математическая постановка ##

Предполагается

$$ X=L+S, $$

где \(L\) — низкоранговая часть, а \(S\) — разреженная.

Решается

$$ \min_{L,S} \|L\|_* + \lambda\|S\|_1 $$

при

$$ X=L+S. $$

## Частный случай ##

Нельзя непосредственно подавать единственную последовательность

$$ (t_i,\mathrm{lat}_i,\mathrm{lon}_i) $$

в RPCA. Сначала строится матрица окон:

$$ X= \begin{pmatrix} \mathbf f_1^T\\ \mathbf f_2^T\\ \vdots\\ \mathbf f_m^T \end{pmatrix}, $$

где \(\mathbf f_j\) — вектор признаков некоторого временного окна траектории.

После разложения

$$ X=L+S $$

значение

$$ s_{ij}=|S_{ij}| $$

характеризует величину отклонения.

Для точки исходной траектории можно агрегировать значения нескольких перекрывающихся окон:

$$ s_i=\frac{1}{|W_i|}\sum_{W:\, i\in W} \|S_W(i)\|. $$

Тогда

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Candès E. J., Li X., Ma Y., Wright J. Robust Principal Component Analysis? // Journal of the ACM. 2011. Vol. 58, No. 3. Article 11. P. 1–37. DOI: [10.1145/1970392.1970395](https://doi.org/10.1145/1970392.1970395). В `docs/methods/` лежит препринт arXiv:0912.3599v1 (18 December 2009): [PDF](docs/methods/Robust%20Principal%20Component%20Analysis.pdf).

# LSTM Autoencoder / EncDec-AD #

Сеть учится сжимать и восстанавливать окна нормальной траектории. Аномалия — окно, которое восстанавливается плохо.

## Общая математическая постановка ##

Для временного окна

$$ X_i=(x_{i-L+1},\ldots,x_i) $$

энкодер строит латентное представление:

$$ h_i=E_\theta(X_i), $$

декодер:

$$ \hat X_i=D_\phi(h_i). $$

Обучение:

$$ \min_{\theta,\phi} \sum_i \|X_i-\hat X_i\|_2^2. $$

Anomaly score:

$$ s_i=\|X_i-\hat X_i\|_2^2. $$

## Частный случай ##

Для вашей траектории

$$ X_i=\left[ \begin{pmatrix} \mathrm{lat}_{i-L+1}\\ \mathrm{lon}_{i-L+1} \end{pmatrix}, \ldots, \begin{pmatrix} \mathrm{lat}_i\\ \mathrm{lon}_i \end{pmatrix} \right]. $$

Получаем

$$ \hat X_i=D(E(X_i)). $$

Для point-wise оценки можно использовать ошибку конечной точки окна:

$$ s_i=d_{\mathrm{geo}}\left( (\mathrm{lat}_i,\mathrm{lon}_i), (\widehat{\mathrm{lat}}_i,\widehat{\mathrm{lon}}_i) \right). $$

Тогда

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

Основная идея — модель учится восстанавливать нормальную динамику, поэтому длительная аномальная последовательность должна давать систематически повышенный reconstruction error.

**Источник:** Malhotra P., Ramakrishnan A., Anand G., Vig L., Agarwal P., Shroff G. LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection // ICML 2016 Anomaly Detection Workshop. New York, 2016. arXiv: [1607.00148](https://arxiv.org/abs/1607.00148). [PDF](docs/methods/LSTM-based%20Encoder-Decoder%20for%20Multi-sensor%20Anomaly%20Detection.pdf).

# Temporal Convolutional Network #

Свёрточная сеть по прошлым точкам предсказывает следующую. Аномалия — большое расхождение прогноза с фактической координатой.

## Общая математическая постановка ##

Для каузальной TCN:

$$ \hat x_{t+1}=f_\theta(x_{t-L+1},\ldots,x_t). $$

Модель обучается:

$$ \min_\theta \sum_t \|x_{t+1}-\hat x_{t+1}\|_2^2. $$

Остаток:

$$ r_{t+1}=x_{t+1}-\hat x_{t+1}. $$

При многомерном нормальном распределении остатков:

$$ r_t\sim\mathcal N(0,\Sigma). $$

Тогда

$$ s_t=r_t^T\Sigma^{-1}r_t. $$

## Частный случай ##

$$ \hat{\mathbf z}_{i}=f_\theta(\mathbf z_{i-L},\ldots,\mathbf z_{i-1}), $$

где

$$ \mathbf z_i=(\mathrm{lat}_i,\mathrm{lon}_i)^T. $$

Получаем

$$ \mathbf r_i=\mathbf z_i-\hat{\mathbf z}_i, \qquad s_i=\mathbf r_i^T\Sigma^{-1}\mathbf r_i, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

Это уже очень близко к вашей исходной идее «модель должна понять нормальную последовательность, а аномальная точка должна отличаться от прогноза».

**Источник:** He Y., Zhao J. Temporal Convolutional Networks for Anomaly Detection in Time Series // Journal of Physics: Conference Series. 2019. Vol. 1213. Article 042050. DOI: [10.1088/1742-6596/1213/4/042050](https://doi.org/10.1088/1742-6596/1213/4/042050). [PDF](docs/methods/Temporal_Convolutional_Networks_for_Anomaly_Detect.pdf).

# DONUT — VAE для anomaly detection #

Вариационный автоэнкодер учит распределение нормальных окон. Аномалия — окно с низкой вероятностью по этой модели.

## Общая математическая постановка ##

Для наблюдения \(x\):

$$ q_\phi(z|x)=\mathcal N\left( \mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)) \right). $$

Генеративная модель:

$$ p_\theta(x|z). $$

Обучение максимизирует ELBO:

$$ \mathcal L(x)=\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}\left( q_\phi(z|x)\|p(z) \right). $$

Anomaly score можно определить как отрицательную log-likelihood:

$$ s_i=-\log p_\theta(x_i). $$

На практике это связано с reconstruction/probabilistic loss.

## Частный случай ##

Для окон траектории:

$$ X_i=\bigl( (t,\mathrm{lat},\mathrm{lon}) \bigr)_{i-L+1:i}. $$

После обучения на нормальных окнах:

$$ s_i=-\log p_\theta(X_i). $$

Если \(s_i>\tau\), то \(\hat a_i=1\) (точка помечается как аномалия).

Для длительных аномалий такая постановка имеет смысл, поскольку аномалия может быть не одним экстремальным значением, а последовательностью, маловероятной для распределения нормальных окон.

**Источник:** Xu H., Chen W., Zhao N., Li Z., Bu J., Li Z., Liu Y., Zhao Y., Pei D., Feng Y., Chen J., Wang Z., Qiao H. Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications // Proceedings of The Web Conference 2018 (WWW '18). Lyon, France, 23–27 April 2018. ACM, 2018. 12 p. DOI: [10.1145/3178876.3185996](https://doi.org/10.1145/3178876.3185996). arXiv: [1802.03903](https://arxiv.org/abs/1802.03903). [PDF](docs/methods/Unsupervised%20Anomaly%20Detection%20via%20Variational%20Auto-Encoder.pdf).

# Anomaly Transformer #

Модель смотрит, с какими моментами времени точка связана через attention. У аномалии эти связи устроены иначе, чем у обычных точек ряда.

## Общая математическая постановка ##

Для окна

$$ X_i=(x_{i-L+1},\ldots,x_i) $$

self-attention формирует ассоциации:

$$ A_{ij}=\operatorname{softmax}\left( \frac{Q_iK_j^T}{\sqrt d} \right). $$

Anomaly Transformer дополнительно строит prior association \(P_{ij}\).

Для точки \(i\) anomaly score строится через association discrepancy:

$$ s_i=\frac{1}{L}\sum_{j=1}^{L}\left[ \mathrm{KL}(A_i\|P_i)+\mathrm{KL}(P_i\|A_i) \right]. $$

Аномальные точки должны иметь структуру ассоциаций, отличающуюся от типичной временной зависимости.

## Частный случай ##

Подаём в модель

$$ x_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i) $$

или, что для движения обычно информативнее,

$$ x_i=(\mathrm{lat}_i,\mathrm{lon}_i,v_i,\Delta\psi_i). $$

Для каждого временного окна получаем

$$ s_i=\operatorname{AD}(X_i), $$

после чего

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

То есть в данном случае модель не просто смотрит на расстояние до прогноза, а оценивает, насколько временные связи точки отличаются от характерных ассоциаций нормальной последовательности.

**Источник:** Xu J., Wu H., Wang J., Long M. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy // International Conference on Learning Representations (ICLR). 2022. arXiv: [2110.02642](https://arxiv.org/abs/2110.02642). [PDF](docs/methods/ANOMALY%20TRANSFORMER%20TIME%20SERIES%20ANOMALY.pdf).

# Spatio-Temporal GNN — STGVAD #

Траектории нескольких судов собираются в один граф по времени и пространственной близости. Сеть предсказывает нормальное продолжение, а большое отклонение от него считается аномалией.

## Общая математическая постановка ##

Траектории нескольких судов образуют граф

$$ G=(V,E). $$

Каждая вершина соответствует состоянию судна в конкретный момент:

$$ v_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i,\ldots). $$

Рёбра разделяются на временные и пространственные:

$$ E=E_{\mathrm{time}}\cup E_{\mathrm{space}}. $$

Общий message-passing шаг:

$$ h_i^{(l+1)}=\sigma\left( W_0h_i^{(l)}+\sum_{j\in N(i)}\alpha_{ij}W_1h_j^{(l)} \right). $$

Получаем представление

$$ H=\mathrm{GNN}_\theta(G,X). $$

Далее temporal module предсказывает нормальное продолжение:

$$ \hat x_i=g_\phi(H_{\le i}). $$

Anomaly score:

$$ s_i=d(x_i,\hat x_i). $$

## Частный случай ##

Для вашей задачи вершина:

$$ v_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i) $$

или расширенный

$$ v_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i,v_i,\mathrm{COG}_i). $$

Временные рёбра:

$$ (v_i,v_{i+1}) $$

для одного судна.

Пространственные рёбра:

$$ (v_i,v_j) \quad\text{при}\quad d_{\mathrm{geo}}(v_i,v_j)<\varepsilon. $$

После GNN/temporal блока:

$$ s_i=d_{\mathrm{geo}}\left( (\mathrm{lat}_i,\mathrm{lon}_i), (\widehat{\mathrm{lat}}_i,\widehat{\mathrm{lon}}_i) \right) $$

или аналогичная ошибка восстановления.

Затем

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

В STGVAD именно временные связи внутри траекторий и пространственные взаимодействия между судами объединяются в одном графовом представлении; для оценки авторы также используют искусственно инжектированные аномалии в AIS.

**Источник:** Kim J., Kim M., Hwang Y., Bae S., Cho D. J., Lee W., Park H. STGVAD: Spatio-Temporal Graph-based Vessel Behavior Anomaly Detection // IEEE Access. 2025. DOI: [10.1109/ACCESS.2025.3609783](https://doi.org/10.1109/ACCESS.2025.3609783). В `docs/methods/` лежит авторская версия, принятая к публикации. [PDF](docs/methods/STGVAD_Spatio-Temporal_Graph-Based_Vessel_Behavior.pdf).

# Optimal Speed-Bounded Trajectory #

Из траектории выбирается самая длинная подпоследовательность, которую судно физически могло пройти с допустимой скоростью. Остальные точки считаются выбросами.

## Общая математическая постановка ##

Пусть для объекта заданы допустимые скорости

$$ v_-\le v_i\le v_+. $$

Для двух последовательных измерений:

$$ v(p_i,p_j)=\frac{d(p_i,p_j)}{t_j-t_i}. $$

Пара точек физически согласована, если

$$ v_-\le\frac{d(p_i,p_j)}{t_j-t_i}\le v_+. $$

Ищется максимальная физически согласованная подпоследовательность:

$$ Q^\star=\arg\max_{Q\subseteq\{1,\ldots,N\}} |Q| $$

при условии, что все последовательные элементы \(Q\) удовлетворяют ограничению скорости.

Тогда точки, не вошедшие в \(Q^\star\), являются кандидатами на выбросы.

## Частный случай для \(P\) и \(G\) ##

Для

$$ p_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i) $$

получаем

$$ v_{ij}=\frac{d_{\mathrm{geo}}\left( (\mathrm{lat}_i,\mathrm{lon}_i), (\mathrm{lat}_j,\mathrm{lon}_j) \right)}{t_j-t_i}. $$

Выбирается

$$ Q^\star(P). $$

Тогда

$$ \hat a_i=\mathbf1[i\notin Q^\star(P)]. $$

Соответственно,

$$ \hat y_i= \begin{cases} 0,&i\notin Q^\star,\\ 1,&i\in Q^\star. \end{cases} $$

Это существенно сильнее простой проверки соседних точек: алгоритм ищет максимально длинную физически согласованную траекторию и рассматривает остальные измерения как выбросы. Custers et al. формально рассматривают именно физически согласованные траектории с ограничениями на скорость и ускорение.

**Источник:** Custers B., van de Kerkhof M., Meulemans W., Speckmann B., Staals F. Maximum Physically Consistent Trajectories // ACM Transactions on Spatial Algorithms and Systems. 2021. Vol. 7, No. 4. Article 17. 33 p. DOI: [10.1145/3452378](https://doi.org/10.1145/3452378). [PDF](docs/methods/Maximum%20Physically%20Consistent%20Trajectories.pdf).
