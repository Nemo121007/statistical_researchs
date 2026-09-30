# Общие тезисы #
Наблюдаемая траектория судна представляется как последовательность точек: 
X={x
i
	​

}
i=1
N
	​

,x
i
	​

=(t
i
	​

,lat
i
	​

,lon
i
	​

).

Модель \(M\), обученная на некотором наборе нормальных данных \(X_{\mathrm{train}}\), вычисляет для каждой точки anomaly score

$$ s_i^{(M)} = M(X_{\le i}), $$

после чего вводится порог \(\tau_M\):

$$ \hat a_i^{(M)} = \mathbf 1[s_i^{(M)}>\tau_M]. $$

Предсказание можно привести к исходной схеме меток:

$$ \hat y_i^{(M)} = \begin{cases} 0, & s_i^{(M)}>\tau_M,\\ 1, & s_i^{(M)}\le\tau_M. \end{cases} $$

При наличии нескольких нормальных классов \(y_i>0\) конкретный положительный класс уже является отдельной задачей классификации. Для оценки бинарного обнаружения используются

a
i
	​

=1[y
i
	​

=0],
a
^
i
	​

=1[
y
^
	​

i
	​

=0].

# Авторегрессионная модель ARIMA(2,1,1) #
(Добавить небольшое текстовое описание на чём работает метод)

Для одномерного временного ряда \(x_t\) модель ARIMA\((p1,d,q)\) задаётся как
$$ \phi(B)(1-B)^d x_t = c+\theta(B)\varepsilon_t, \qquad \varepsilon_t\sim\mathcal N(0,\sigma^2), $$
где \(B\) — оператор запаздывания,
$$ \phi(B)=1-\phi_1B-\ldots-\phi_pB^p, $$ $$ \theta(B)=1+\theta_1B+\ldots+\theta_qB^q. $$
В качестве конкретной модели выбираем ARIMA\((2,1,1)\):
$$ \Delta x_t = c+\phi_1\Delta x_{t-1} +\phi_2\Delta x_{t-2} +\varepsilon_t+\theta_1\varepsilon_{t-1}. $$
Аномальность измерения определяется через ошибку прогноза:
$$ r_t=x_t-\hat x_t. $$
При известной дисперсии прогноза
$$ s_t=\frac{|r_t|}{\sigma_t}. $$
Аномалия:
$$ s_t>\tau. $$
Частный случай для \(P\) и \(G\)\\
Для траектории судна строятся две модели:
$$ \mathrm{lat}_i \sim \mathrm{ARIMA}(2,1,1), \qquad \mathrm{lon}_i \sim \mathrm{ARIMA}(2,1,1). $$
Получаем
$$ \hat{\mathbf z}_i = \begin{pmatrix} \widehat{\mathrm{lat}}_i\\ \widehat{\mathrm{lon}}_i \end{pmatrix}, \qquad \mathbf z_i = \begin{pmatrix} \mathrm{lat}_i\\ \mathrm{lon}_i \end{pmatrix}. $$
Ошибка:
$$ r_i= \mathbf z_i-\hat{\mathbf z}_i. $$
Для географических координат предпочтительнее использовать геодезическую дистанцию
$$ s_i=d_{\mathrm{geo}}(\mathbf z_i,\hat{\mathbf z}_i) $$
либо локальную метрическую проекцию.
Итоговая оценка:
$$ \hat a_i=1[s_i>\tau]. $$
Для ARIMA фактическая аномальная метка \(y_i=0\) в процессе построения прогноза не нужна.
Публикация: Qin Yu, Jibin Lyu, Lirui Jiang, An Improved ARIMA-Based Traffic Anomaly Detection Algorithm for Wireless Sensor Networks, 2016. Статья опубликована в открытом доступе.
Открытый текст статьи

# Калмановский фильтр с моделью постоянной скорости # 
Здесь вместо общего KF выбираем конкретную модель состояния.

Общая математическая постановка

Состояние:

$$ \mathbf x_i= \begin{pmatrix} p_x\\ p_y\\ v_x\\ v_y \end{pmatrix}_i. $$

Модель движения:

$$ \mathbf x_i=A_i\mathbf x_{i-1}+\mathbf w_i, \qquad \mathbf w_i\sim\mathcal N(0,Q_i), $$

где

$$ A_i= \begin{pmatrix} 1&0&\Delta t_i&0\\ 0&1&0&\Delta t_i\\ 0&0&1&0\\ 0&0&0&1 \end{pmatrix}. $$

Измерение:

$$ \mathbf z_i=H\mathbf x_i+\mathbf v_i, \qquad \mathbf v_i\sim\mathcal N(0,R), $$ $$ H= \begin{pmatrix} 1&0&0&0\\ 0&1&0&0 \end{pmatrix}. $$

Инновация:

$$ \mathbf r_i=\mathbf z_i-H\hat{\mathbf x}_{i|i-1}. $$

Ковариация инновации:

$$ S_i=HP_{i|i-1}H^T+R. $$

Естественный статистический anomaly score — normalized innovation squared:

$$ s_i= \mathbf r_i^TS_i^{-1}\mathbf r_i. $$

При гауссовских предположениях

$$ s_i\sim\chi^2_2 $$

при отсутствии аномалии, поэтому

$$ s_i>\chi^2_{2,1-\alpha} $$

означает аномальное измерение.

Частный случай для \(P\) и \(G\)

Из

$$ x_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i) $$

формируем

$$ \mathbf z_i= \begin{pmatrix} \mathrm{lat}_i\\ \mathrm{lon}_i \end{pmatrix}. $$

Скорости \(v_x,v_y\) являются скрытой частью состояния и оцениваются фильтром.

Затем

$$ \hat a_i=\mathbf1[s_i>\tau], $$ $$ \hat y_i= \begin{cases} 0,&s_i>\tau,\\ 1,&s_i\le\tau. \end{cases} $$

Такой вариант непосредственно соответствует ситуации, когда физически правдоподобная траектория имеет небольшие локальные ошибки измерений, а настоящий скачок вызывает большую инновацию.

Публикация: R. E. Kalman, A New Approach to Linear Filtering and Prediction Problems, 1960

# Bootstrap Particle Filter / SIR #

Здесь вместо класса PF выбираем стандартный bootstrap particle filter, также известный как Sampling-Importance-Resampling.

Общая математическая постановка

Состояние:

$$ \mathbf x_i\sim p(\mathbf x_i\mid\mathbf x_{i-1}). $$

Представим апостериорное распределение частицами:

$$ p(\mathbf x_i\mid z_{1:i}) \approx \sum_{j=1}^{K}w_i^{(j)} \delta(\mathbf x_i-\mathbf x_i^{(j)}). $$

Предсказание:

$$ \mathbf x_i^{(j)} \sim p(\mathbf x_i\mid\mathbf x_{i-1}^{(j)}). $$

Веса:

$$ \tilde w_i^{(j)} = w_{i-1}^{(j)} p(\mathbf z_i\mid\mathbf x_i^{(j)}), $$ $$ w_i^{(j)} = \frac{\tilde w_i^{(j)}} {\sum_{k=1}^{K}\tilde w_i^{(k)}}. $$

Предсказательная вероятность измерения:

$$ p(\mathbf z_i\mid z_{1:i-1}) \approx \sum_{j=1}^{K} w_{i-1}^{(j)} p(\mathbf z_i\mid\mathbf x_i^{(j)}). $$

Поэтому естественный score:

$$ s_i= -\log p(\mathbf z_i\mid z_{1:i-1}). $$

Большое значение \(s_i\) соответствует маловероятному измерению.

Частный случай для \(P\) и \(G\)

Используем ту же модель движения постоянной скорости, что и в KF:

$$ \mathbf x_i= (\mathrm{lat}_i,\mathrm{lon}_i,v_{\mathrm{lat},i},v_{\mathrm{lon},i})^T, $$

но вместо единственного гауссовского состояния поддерживаем \(K\) гипотез движения.

Получаем

$$ s_i= -\log \left( \sum_{j=1}^{K} w_{i-1}^{(j)} p( (\mathrm{lat}_i,\mathrm{lon}_i) \mid \mathbf x_i^{(j)} ) \right). $$

После пороговой операции:

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

Преимущество по сравнению с KF заключается в том, что распределение состояний не обязано быть одним гауссовским облаком. Это потенциально полезно для поворотов, развилок и других мультимодальных движений.

Публикация: Arulampalam et al., A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking, 2002.

# Gaussian Process Regression с RBF-ядром #

Здесь выбираем конкретно GP-регрессию с RBF covariance kernel.

Общая математическая постановка

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

$$ s_i= (\mathbf z_i-\boldsymbol\mu_i)^T \Sigma_i^{-1} (\mathbf z_i-\boldsymbol\mu_i). $$

При нормальности:

$$ s_i\sim\chi^2_2. $$
Частный случай для \(P\) и \(G\)

Строятся два GP:

$$ f_{\mathrm{lat}}(t), \qquad f_{\mathrm{lon}}(t). $$

Для точки \(i\):

$$ \boldsymbol\mu_i= \begin{pmatrix} \mu_{\mathrm{lat}}(t_i)\\ \mu_{\mathrm{lon}}(t_i) \end{pmatrix}. $$

Из \(P\) берём наблюдаемую координату

$$ \mathbf z_i= \begin{pmatrix} \mathrm{lat}_i\\ \mathrm{lon}_i \end{pmatrix}. $$

Тогда

$$ \hat a_i = \mathbf1 \left[ (\mathbf z_i-\boldsymbol\mu_i)^T \Sigma_i^{-1} (\mathbf z_i-\boldsymbol\mu_i) >\tau \right]. $$

Этот подход особенно интересен для вашей задачи, потому что GP непосредственно моделирует неопределённость прогноза, а не только одну ожидаемую координату.

Публикации:\

Публикация, непосредственно посвящённая морским траекториям: Smith et al., Maritime abnormality detection using Gaussian processes, 2013/2014. Где текст?
Дополнительно существует более современная работа по GP anomaly detection в динамических системах, опубликованная в 2026 году в arXiv

# Hidden Markov Model #

Здесь лучше выбрать не абстрактный «HMM/HSMM», а конкретную HMM-схему для аномального поведения судов.

В качестве объекта наблюдения удобно использовать признаки движения:

$$ o_i= (v_i,\Delta v_i,\Delta\psi_i), $$

где

$$ v_i= \frac{d_{\mathrm{geo}}(x_{i-1},x_i)} {\Delta t_i}. $$
Общая математическая постановка

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
Частный случай

Для каждой траектории

$$ P= \{(t_i,\mathrm{lat}_i,\mathrm{lon}_i,\hat y_i)\} $$

строится последовательность

$$ O(P)=\{o_i\}_{i=1}^{N}. $$

Один из вариантов конкретной модели — набор HMM для классов поведения:

$$ \theta_N,\theta_{A_1},\ldots,\theta_{A_K}, $$

где \(\theta_N\) соответствует нормальному движению, а остальные модели — различным аномальным режимам.

Для point-wise оценки удобно использовать отношение предсказательных вероятностей:

$$ s_i = \max_{k} \log \frac {P(o_i\mid o_{1:i-1},\theta_{A_k})} {P(o_i\mid o_{1:i-1},\theta_N)}. $$

Тогда

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

Это point-wise специализация исходной trajectory-level HMM-схемы. Она позволяет привести метод к вашей разметке \(y_i=0\) / \(y_i>0\).

Для вашей задачи этот метод особенно интересен для аномалий типа «спираль», изменение режима движения, неожиданные остановки и т. п., поскольку HMM моделирует не только отдельные значения, но и переходы между состояниями.

Публикация: Toloue, Jahan, Anomalous Behavior Detection of Marine Vessels Based on Hidden Markov Model, 2018. Полный авторский текст доступен свободно. Работа непосредственно посвящена обнаружению аномального поведения морских судов по HMM.

# Hampel Filter #

Вместо общего «MAD/z-score» здесь разумно выбрать именно Hampel filter.

Общая математическая постановка

Для окна

$$ W_i=\{x_{i-h},\ldots,x_i,\ldots,x_{i+h}\} $$

медиана:

$$ m_i=\operatorname{median}(W_i). $$

Медианное абсолютное отклонение:

$$ MAD_i= \operatorname{median}_{x\in W_i}|x-m_i|. $$

Стандартизованное отклонение:

$$ s_i= \frac{|x_i-m_i|} {1.4826\,MAD_i+\varepsilon}. $$

Точка аномальна:

$$ s_i>\tau. $$
Частный случай

Применим фильтр к производным траектории, например к

$$ x_i=v_i $$

или

$$ x_i=a_i= \frac{v_i-v_{i-1}}{\Delta t_i}, $$

а не просто к широте и долготе.

Например,

$$ s_i^{(v)} = \frac{|v_i-\operatorname{med}(v_{i-h:i+h})|} {1.4826\,MAD(v_{i-h:i+h})+\varepsilon}. $$

Тогда

$$ \hat a_i = \mathbf1[s_i^{(v)}>\tau_v]. $$

В этом виде метод хорошо подходит для единичных выбросов, но не предназначен для длительных аномальных сегментов, которые сами образуют устойчивый локальный режим.

Публикация: Roos-Hoefgeest Toribio et al., A Novel Approach to Speed Up Hampel Filter for Outlier Detection, 2025, открытый доступ MDPI

# DBSCAN #

Здесь выбираем обычный DBSCAN, а не неопределённое множество clustering algorithms.

Общая математическая постановка

Для множества объектов \(X\) и метрики \(d\) определяется \(\varepsilon\)-окрестность:

$$ N_\varepsilon(x) = \{x'\in X:d(x,x')\le\varepsilon\}. $$

Точка является core point, если

$$ |N_\varepsilon(x)|\ge\mathrm{MinPts}. $$

Точка, не относящаяся ни к одному кластеру, получает статус noise:

$$ c_i=-1. $$
Частный случай

Для каждой точки формируется вектор движения, например

$$ \mathbf f_i= \left( x_i^{\mathrm{proj}}, y_i^{\mathrm{proj}}, v_i, \Delta\psi_i \right). $$

После нормализации признаков:

$$ \mathbf u_i=\operatorname{scale}(\mathbf f_i). $$

Запускаем

$$ DBSCAN(\mathbf u_i,\varepsilon,\mathrm{MinPts}). $$

Тогда

$$ \hat a_i = \mathbf1[c_i=-1]. $$

То есть

$$ \hat y_i= \begin{cases} 0,&c_i=-1,\\ 1,&c_i\ne-1. \end{cases} $$

Для морских траекторий есть непосредственно соответствующее исследование, где DBSCAN применяется к AIS с пространственными и динамическими признаками.

Публикация: Ester et al., A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise, 1996. Открытая копия доступна через AAAI.
Для морской специализации: Han, Armenakis, Jadidi, DBSCAN Optimization for Improving Marine Trajectory Clustering and Anomaly Detection, 2020 — открытый PDF и CC BY 4.0.

# Local Outlier Factor #
Общая математическая постановка

Пусть \(N_k(x)\) — множество \(k\) ближайших соседей.

Reachability distance:

$$ \operatorname{reachdist}_k(x,y) = \max\{k\text{-distance}(y),d(x,y)\}. $$

Локальная плотность:

$$ lrd_k(x) = \left( \frac{1}{|N_k(x)|} \sum_{y\in N_k(x)} \operatorname{reachdist}_k(x,y) \right)^{-1}. $$

LOF:

$$ LOF_k(x) = \frac{1}{|N_k(x)|} \sum_{y\in N_k(x)} \frac{lrd_k(y)}{lrd_k(x)}. $$

При

$$ LOF_k(x)\approx1 $$

плотность точки соответствует соседям; большие значения указывают на выброс.

Частный случай

Формируем

$$ \mathbf f_i= \left( \Delta x_i, \Delta y_i, v_i, a_i, \Delta\psi_i \right). $$

После нормализации:

$$ \mathbf u_i=\operatorname{scale}(\mathbf f_i). $$

Тогда

$$ s_i=LOF_k(\mathbf u_i) $$

и

$$ \hat a_i= \mathbf1[s_i>\tau_{\mathrm{LOF}}]. $$

LOF особенно полезен в ситуации, когда «аномальность» определяется не глобальной редкостью, а отличием от локального режима.

Публикация: Breunig et al., LOF: Identifying Density-Based Local Outliers, 2000. Полный PDF доступен через SIGMOD Record.

# Isolation Forest #
Общая математическая постановка

Строится множество случайных isolation trees.

Для точки \(x\) определяется средняя длина пути:

$$ E[h(x)]. $$

Нормировочный коэффициент:

$$ c(n) = 2H(n-1)-\frac{2(n-1)}{n}, $$

где \(H(\cdot)\) — гармоническое число.

Anomaly score:

$$ s(x) = 2^{-\frac{E[h(x)]}{c(n)}}. $$

Чем меньше путь, тем более изолирована точка и тем выше \(s(x)\).

Частный случай

Используем

$$ \mathbf f_i= ( x_i^{\mathrm{proj}}, y_i^{\mathrm{proj}}, v_i, a_i, \Delta\psi_i ). $$

Для каждой точки:

$$ s_i = IF(\mathbf f_i). $$

Затем

$$ \hat a_i = \mathbf1[s_i>\tau]. $$

Метод не требует явного построения модели нормального движения и поэтому хорошо подходит как baseline для большой выборки.

Публикация: Liu, Ting, Zhou, Isolation-Based Anomaly Detection, 2012. Это открытая версия работы об Isolation Forest в ACM.

# One-Class SVM #

Чтобы выбрать один конкретный метод, здесь я бы использовал kernel One-Class SVM.

Общая математическая постановка

Решается задача

$$ \min_{\mathbf w,\rho,\xi} \frac12\|\mathbf w\|^2 + \frac{1}{\nu n} \sum_{i=1}^{n}\xi_i -\rho $$

при ограничениях

$$ \mathbf w^T\phi(x_i)\ge\rho-\xi_i, \qquad \xi_i\ge0. $$

В kernel formulation:

$$ f(x) = \sum_i\alpha_iK(x_i,x)-\rho. $$

Объект считается аномальным, если

$$ f(x)<0. $$
Частный случай

Обучение выполняется только на нормальных точках:

$$ \mathcal D_N = \{\mathbf f_i:y_i>0\}. $$

Признак:

$$ \mathbf f_i= ( x_i^{\mathrm{proj}}, y_i^{\mathrm{proj}}, v_i, a_i, \Delta\psi_i ). $$

Тогда

$$ s_i=-f(\mathbf f_i) $$

и

$$ \hat a_i = \mathbf1[s_i>0]. $$

Иными словами, модель учится описывать support нормального распределения, а всё, что оказывается за его границей, считается аномальным.

Публикация: Schölkopf et al., Estimating the Support of a High-Dimensional Distribution, 2001. Свободная версия доступна в Microsoft Research.

# Robust PCA / Principal Component Pursuit #

Здесь конкретный алгоритм — Principal Component Pursuit.

Общая математическая постановка

Предполагается

$$ X=L+S, $$

где \(L\) — низкоранговая часть, а \(S\) — разреженная.

Решается

$$ \min_{L,S} \|L\|_* + \lambda\|S\|_1 $$

при

$$ X=L+S. $$
Частный случай

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

$$ s_i= \frac1{|W_i|} \sum_{W\ni i} \|S_W(i)\|. $$

Тогда

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

Публикация: Candès et al., Robust Principal Component Analysis?, 2009/2011. Полный текст доступен на arXiv.

# LSTM Autoencoder / EncDec-AD #
Общая математическая постановка

Для временного окна

$$ X_i=(x_{i-L+1},\ldots,x_i) $$

энкодер строит латентное представление:

$$ h_i=E_\theta(X_i), $$

декодер:

$$ \hat X_i=D_\phi(h_i). $$

Обучение:

$$ \min_{\theta,\phi} \sum_i \|X_i-\hat X_i\|_2^2. $$

Anomaly score:

$$ s_i = \|X_i-\hat X_i\|_2^2. $$
Частный случай

Для вашей траектории

$$ X_i= \left[ \begin{pmatrix} \mathrm{lat}_{i-L+1}\\ \mathrm{lon}_{i-L+1} \end{pmatrix}, \ldots, \begin{pmatrix} \mathrm{lat}_i\\ \mathrm{lon}_i \end{pmatrix} \right]. $$

Получаем

$$ \hat X_i=D(E(X_i)). $$

Для point-wise оценки можно использовать ошибку конечной точки окна:

$$ s_i= d_{\mathrm{geo}} \left( (\mathrm{lat}_i,\mathrm{lon}_i), (\widehat{\mathrm{lat}}_i,\widehat{\mathrm{lon}}_i) \right). $$

Тогда

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

Основная идея — модель учится восстанавливать нормальную динамику, поэтому длительная аномальная последовательность должна давать систематически повышенный reconstruction error.

Публикация: Malhotra et al., LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection, 2016. Полный текст доступен на arXiv.

# Temporal Convolutional Network #

Здесь вместо «CNN/TCN/RNN» фиксируем именно TCN predictor.

Общая математическая постановка

Для каузальной TCN:

$$ \hat x_{t+1} = f_\theta(x_{t-L+1},\ldots,x_t). $$

Модель обучается:

$$ \min_\theta \sum_t \|x_{t+1}-\hat x_{t+1}\|_2^2. $$

Остаток:

$$ r_{t+1} = x_{t+1}-\hat x_{t+1}. $$

При многомерном нормальном распределении остатков:

$$ r_t\sim\mathcal N(0,\Sigma). $$

Тогда

$$ s_t=r_t^T\Sigma^{-1}r_t. $$
Частный случай
$$ \hat{\mathbf z}_{i} = f_\theta (\mathbf z_{i-L},\ldots,\mathbf z_{i-1}), $$

где

$$ \mathbf z_i= (\mathrm{lat}_i,\mathrm{lon}_i)^T. $$

Получаем

$$ \mathbf r_i= \mathbf z_i-\hat{\mathbf z}_i, $$ $$ s_i= \mathbf r_i^T\Sigma^{-1}\mathbf r_i, $$ $$ \hat a_i=\mathbf1[s_i>\tau]. $$

Это уже очень близко к вашей исходной идее «модель должна понять нормальную последовательность, а аномальная точка должна отличаться от прогноза».

Публикация: He, Zhao, Temporal Convolutional Networks for Anomaly Detection in Time Series, 2019. Работа имеет открытый доступ по лицензии CC BY 3.0.

# DONUT — VAE для anomaly detection #

Вместо общего «VAE/GAN» выбираем конкретно DONUT.

Общая математическая постановка

Для наблюдения \(x\):

$$ q_\phi(z|x) = \mathcal N \left( \mu_\phi(x), \operatorname{diag} (\sigma_\phi^2(x)) \right). $$

Генеративная модель:

$$ p_\theta(x|z). $$

Обучение максимизирует ELBO:

$$ \mathcal L(x) = \mathbb E_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL} \left( q_\phi(z|x)\|p(z) \right). $$

Anomaly score можно определить как отрицательную log-likelihood:

$$ s_i=-\log p_\theta(x_i). $$

На практике это связано с reconstruction/probabilistic loss.

Частный случай

Для окон траектории:

$$ X_i= \bigl( (t,\mathrm{lat},\mathrm{lon}) \bigr)_{i-L+1:i}. $$

После обучения на нормальных окнах:

$$ s_i = -\log p_\theta(X_i). $$

Если

$$ s_i>\tau, $$

то

$$ \hat a_i=1. $$

Для длительных аномалий такая постановка имеет смысл, поскольку аномалия может быть не одним экстремальным значением, а последовательностью, маловероятной для распределения нормальных окон.

Публикация: Xu et al., Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications (Donut), 2018. Открытый текст доступен на arXiv

# Anomaly Transformer #

Здесь фиксируем конкретно Anomaly Transformer, а не абстрактный Transformer.

Общая математическая постановка

Для окна

$$ X_i= (x_{i-L+1},\ldots,x_i) $$

self-attention формирует ассоциации:

$$ A_{ij} = \operatorname{softmax} \left( \frac{Q_iK_j^T}{\sqrt d} \right). $$

Anomaly Transformer дополнительно строит prior association \(P_{ij}\).

Для точки \(i\) anomaly score строится через association discrepancy:

$$ s_i = \frac1L \sum_{j=1}^{L} \left[ KL(A_i\|P_i) + KL(P_i\|A_i) \right]. $$

Аномальные точки должны иметь структуру ассоциаций, отличающуюся от типичной временной зависимости.

Частный случай

Подаём в модель

$$ x_i= (t_i,\mathrm{lat}_i,\mathrm{lon}_i) $$

или, что для движения обычно информативнее,

$$ x_i= (\mathrm{lat}_i,\mathrm{lon}_i,v_i,\Delta\psi_i). $$

Для каждого временного окна получаем

$$ s_i=\operatorname{AD}(X_i), $$

после чего

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

То есть в данном случае модель не просто смотрит на расстояние до прогноза, а оценивает, насколько временные связи точки отличаются от характерных ассоциаций нормальной последовательности.

Публикация: Xu et al., Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy, 2021. Полный текст доступен на arXiv.

# Spatio-Temporal GNN — STGVAD #

Здесь вместо общего «GCN/GraphSAGE/ST-GNN» выбираем конкретный maritime-oriented алгоритм STGVAD.

Общая математическая постановка

Траектории нескольких судов образуют граф

$$ G=(V,E). $$

Каждая вершина соответствует состоянию судна в конкретный момент:

$$ v_i= (t_i,\mathrm{lat}_i,\mathrm{lon}_i,\ldots). $$

Рёбра разделяются на временные и пространственные:

$$ E=E_{\mathrm{time}}\cup E_{\mathrm{space}}. $$

Общий message-passing шаг:

$$ h_i^{(l+1)} = \sigma \left( W_0h_i^{(l)} + \sum_{j\in N(i)} \alpha_{ij}W_1h_j^{(l)} \right). $$

Получаем представление

$$ H=GNN_\theta(G,X). $$

Далее temporal module предсказывает нормальное продолжение:

$$ \hat x_i=g_\phi(H_{\le i}). $$

Anomaly score:

$$ s_i= d(x_i,\hat x_i). $$
Частный случай

Для вашей задачи вершина:

$$ v_i= (t_i,\mathrm{lat}_i,\mathrm{lon}_i) $$

или расширенный

$$ v_i= (t_i,\mathrm{lat}_i,\mathrm{lon}_i,v_i,\mathrm{COG}_i). $$

Временные рёбра:

$$ (v_i,v_{i+1}) $$

для одного судна.

Пространственные рёбра:

$$ (v_i,v_j) \quad\text{при}\quad d_{\mathrm{geo}}(v_i,v_j)<\varepsilon. $$

После GNN/temporal блока:

$$ s_i= d_{\mathrm{geo}} \left( (\mathrm{lat}_i,\mathrm{lon}_i), (\widehat{\mathrm{lat}}_i,\widehat{\mathrm{lon}}_i) \right) $$

или аналогичная ошибка восстановления.

Затем

$$ \hat a_i=\mathbf1[s_i>\tau]. $$

В STGVAD именно временные связи внутри траекторий и пространственные взаимодействия между судами объединяются в одном графовом представлении; для оценки авторы также используют искусственно инжектированные аномалии в AIS.

Публикация: Kim et al., STGVAD: Spatio-Temporal Graph-Based Vessel Behavior Anomaly Detection, IEEE Access, 2026. Работа находится в открытом доступе; доступна авторская версия, а также официальный код.

# Optimal Speed-Bounded Trajectory #

Для rule-based части я бы отказался от формулировки «если скорость больше максимальной» и использовал более формальный алгоритм Custers et al.: поиск максимальной физически согласованной подпоследовательности.

Общая математическая постановка

Пусть для объекта заданы допустимые скорости

$$ v_-\le v_i\le v_+. $$

Для двух последовательных измерений:

$$ v(p_i,p_j) = \frac{d(p_i,p_j)} {t_j-t_i}. $$

Пара точек физически согласована, если

$$ v_- \le \frac{d(p_i,p_j)} {t_j-t_i} \le v_+. $$

Ищется максимальная физически согласованная подпоследовательность:

$$ Q^\star = \arg\max_{Q\subseteq\{1,\ldots,N\}} |Q| $$

при условии, что все последовательные элементы \(Q\) удовлетворяют ограничению скорости.

Тогда точки, не вошедшие в \(Q^\star\), являются кандидатами на выбросы.

Частный случай для \(P\) и \(G\)

Для

$$ p_i= (t_i,\mathrm{lat}_i,\mathrm{lon}_i) $$

получаем

$$ v_{ij} = \frac{ d_{\mathrm{geo}} \left( (\mathrm{lat}_i,\mathrm{lon}_i), (\mathrm{lat}_j,\mathrm{lon}_j) \right) } {t_j-t_i}. $$

Выбирается

$$ Q^\star(P). $$

Тогда

$$ \hat a_i = \mathbf1[i\notin Q^\star(P)]. $$

Соответственно,

$$ \hat y_i= \begin{cases} 0,&i\notin Q^\star,\\ 1,&i\in Q^\star. \end{cases} $$

Это существенно сильнее простой проверки соседних точек: алгоритм ищет максимально длинную физически согласованную траекторию и рассматривает остальные измерения как выбросы. Custers et al. формально рассматривают именно физически согласованные траектории с ограничениями на скорость и ускорение. Статья открыта в ACM.

Публикация: Custers et al., Maximum Physically Consistent Trajectories, 2021