# Общие тезисы #

Наблюдаемая траектория судна:

$$ X=\{x_i\}_{i=1}^{N}, \qquad x_i=(t_i,\mathrm{lat}_i,\mathrm{lon}_i). $$

Предсказание и ground truth:

$$ P=\{(t_i,\mathrm{lat}_i,\mathrm{lon}_i,\hat y_i)\}_{i=1}^{N}, \qquad G=\{(t_i,\mathrm{lat}_i,\mathrm{lon}_i,y_i)\}_{i=1}^{N}. $$

Здесь \(y_i=0\) — аномалия, \(y_i>0\) — штатное состояние. Бинарная метка аномалии:

$$ a_i=\mathbf1[y_i=0]. $$

Ускорение далее обозначается \(a_i^{\mathrm{acc}}\), чтобы не смешивать его с \(a_i\).

Метод \(M\) сопоставляет точке anomaly score \(s^{(M)}\): большее значение — более аномальная точка. Формула \(s_i=M(X_{\le i})\) относится к online/causal-вариантам: ARIMA, фильтр Калмана, particle filter, GP-EKF, TCN и HMM. Остальные методы видят окно или весь оцениваемый набор.

На точках \(y_i>0\) учатся ARIMA, GP, GP-EKF, LSTM-автоэнкодер, TCN, DONUT, Anomaly Transformer, STGVAD и One-Class SVM. HMM учится на обоих классах. Isolation Forest строится по всей обучающей выборке. DBSCAN и LOF считают плотность по тому ряду, который размечают. Калман, particle filter, Hampel, RPCA и MPCT параметры модели по отдельной нормальной выборке не оценивают.

Порог и перевод в разметку. Здесь \(\hat a_i=1\) означает аномалию, тогда как в исходной схеме классов аномалия — это метка \(0\):

$$ \hat a_i^{(M)}=\mathbf1[s_i^{(M)}>\tau_M], \qquad \hat y_i^{(M)}=\begin{cases} 0, & s_i^{(M)}>\tau_M,\\ 1, & s_i^{(M)}\le\tau_M. \end{cases} $$

Без порога по score: DBSCAN (шум кластера), One-Class SVM (\(f<0\)), MPCT (скорость и форма окна). Для бинарной оценки \(\hat a_i=\mathbf1[\hat y_i=0]\).

Кинематика, кроме первой точки. \(\Delta t_i=t_i-t_{i-1}\), \(d_{\mathrm{geo}}\) — как в метриках, \(R=6371\,\mathrm{км}\):

$$ v_i=\frac{d_{\mathrm{geo}}(p_{i-1},p_i)}{\Delta t_i}, \qquad a_i^{\mathrm{acc}}=\frac{v_i-v_{i-1}}{\Delta t_i}, \qquad \Delta\psi_i=\mathrm{wrap}(\psi_i-\psi_{i-1}). $$

\(\mathrm{wrap}\) переводит угол в \((-\pi,\pi]\). \(\psi_i\) — курс шага \(p_{i-1}\to p_i\).

Шаг на восток и север, углы в радианах, \(\bar\varphi_i\) — средняя широта шага:

$$ \Delta x_i=R\cos\bar\varphi_i\,\Delta\lambda_i, \qquad \Delta y_i=R\,\Delta\varphi_i. $$

Форма окна \(w=6\): отношение длины пути к смещению. При \(i\le w\) форма равна \(1\), далее

$$ \mathrm{shape}_i=\frac{\sum_{k=i-w}^{i-1}d_{\mathrm{geo}}(p_k,p_{k+1})}{\max\bigl(d_{\mathrm{geo}}(p_{i-w},p_i),1\bigr)}. $$

Робастная шкала по точкам \(y>0\) обучения: вычесть медиану, разделить на межквартильный размах. Для HMM вместо размаха берётся \(1{.}4826\,\mathrm{MAD}\), результат обрезается в \([-25,25]\). Локальная проекция от медианы \((\varphi_0,\lambda_0)\) этих точек нужна Калману, фильтру частиц и GP-EKF:

$$ E_i=R\cos\varphi_0\,(\lambda_i-\lambda_0), \qquad N_i=R(\varphi_i-\varphi_0). $$

# ARIMA для прогнозирования и anomaly detection #

Модель прогнозирует следующее значение по предыдущим и помечает точку как аномальную при большом расхождении с прогнозом.

Qin et al. не фиксируют порядок ARIMA\((2,1,1)\). В статье: скользящее окно, порядок по AIC/BIC, переобучение после сдвига окна, short-step прогноз, экспоненциально взвешенное усреднение предсказаний, детекция по относительной ошибке (порог порядка 5–15% в зависимости от ряда).

Адаптация: выбран порядок \((2,1,1)\). Score опыта — геодезическая ошибка одношагового прогноза, не относительная ошибка статьи и не \(|r_t|/\sigma_t\).

Для одномерного ряда модель ARIMA\((p,d,q)\):

$$ \phi(B)(1-B)^d x_t = c+\theta(B)\varepsilon_t, \qquad \varepsilon_t\sim\mathcal N(0,\sigma^2), $$

$$ \phi(B)=1-\phi_1B-\ldots-\phi_pB^p, \qquad \theta(B)=1+\theta_1B+\ldots+\theta_qB^q. $$

Фиксированный вариант:

$$ \Delta x_t = c+\phi_1\Delta x_{t-1}+\phi_2\Delta x_{t-2}+\varepsilon_t+\theta_1\varepsilon_{t-1}. $$

$$ r_t=x_t-\hat x_t. $$

## Частный случай для \(P\) и \(G\) ##

Две независимые модели в градусах. Коэффициенты — метод Хеннана–Риссанена на самом длинном гладком участке \(y_i>0\):

$$ \mathrm{lat}_i \sim \mathrm{ARIMA}(2,1,1), \qquad \mathrm{lon}_i \sim \mathrm{ARIMA}(2,1,1). $$

Прогноз одношаговый и обновляется фактическим приращением. Метка на шаге прогноза не используется:

$$ s_i=d_{\mathrm{geo}}(\mathbf z_i,\hat{\mathbf z}_i), \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Qin Yu, Lyu Jibin, Lirui Jiang. An Improved ARIMA-Based Traffic Anomaly Detection Algorithm for Wireless Sensor Networks // International Journal of Distributed Sensor Networks. 2016. Vol. 2016. Article ID 9653230. 9 p. DOI: [10.1155/2016/9653230](https://doi.org/10.1155/2016/9653230). [PDF](docs/methods/An%20Improved%20ARIMA-Based%20Traffic%20Anomaly%20Detection%20Algorithm%20for%20Wireless%20Sensor%20Networks.pdf).

# Калмановский фильтр с моделью постоянной скорости #

Фильтр ведёт скрытое состояние «положение и скорость» и считает измерение аномальным, если оно плохо согласуется с предсказанным продолжением движения. Kalman (1960) задаёт фильтр, но не детектор аномалий: normalized innovation squared — статистическая адаптация.

## Общая математическая постановка ##

Состояние и модель постоянной скорости:

$$ \mathbf x_i= \begin{pmatrix} p_x\\ p_y\\ v_x\\ v_y \end{pmatrix}_i, \qquad \mathbf x_i=A_i\mathbf x_{i-1}+\mathbf w_i, \qquad \mathbf w_i\sim\mathcal N(0,Q_i), $$

$$ A_i= \begin{pmatrix} 1&0&\Delta t_i&0\\ 0&1&0&\Delta t_i\\ 0&0&1&0\\ 0&0&0&1 \end{pmatrix}. $$

Измерение:

$$ \mathbf z_i=H\mathbf x_i+\mathbf v_i, \qquad \mathbf v_i\sim\mathcal N(0,R), \qquad H= \begin{pmatrix} 1&0&0&0\\ 0&1&0&0 \end{pmatrix}. $$

Инновация и её ковариация:

$$ \mathbf r_i=\mathbf z_i-H\hat{\mathbf x}_{i|i-1}, \qquad S_i=HP_{i|i-1}H^T+R. $$

$$ s_i=\mathbf r_i^T S_i^{-1}\mathbf r_i. $$

При корректно заданной гауссовской модели \(s_i\sim\chi^2_2\). Квантиль \(\chi^2\) в опыт порогом не входит: \(\hat a_i=\mathbf1[s_i>\tau]\).

## Частный случай для \(P\) и \(G\) ##

Состояние ведётся в \((E,N)\). Шум процесса — дискретное белошумное ускорение, шаг в матрице \(A\) ограничен: \(\Delta t=\min(\Delta t_i,45\,\mathrm{с})\),

$$ Q(\Delta t)=\sigma_a^2\begin{pmatrix}\Delta t^4/4&0&\Delta t^3/2&0\\ 0&\Delta t^4/4&0&\Delta t^3/2\\ \Delta t^3/2&0&\Delta t^2&0\\ 0&\Delta t^3/2&0&\Delta t^2\end{pmatrix}. $$

Если \(\Delta t_i>180\,\mathrm{с}\), инновация этого шага всё равно входит в \(s_i\), после чего состояние заменяется измерением, а скорость обнуляется.

# Bootstrap Particle Filter / SIR #

Вместо одного прогноза держится набор гипотез о движении. Аномалия — измерение, маловероятное для всех гипотез сразу. Particle filter может представлять мультимодальное распределение состояния; развилки и резкие повороты требуют соответствующей transition model, а не только замены фильтра Калмана.

## Общая математическая постановка ##

$$ p(\mathbf x_i\mid z_{1:i}) \approx \sum_{j=1}^{K}w_i^{(j)} \delta(\mathbf x_i-\mathbf x_i^{(j)}). $$

Шаг SIR: resample по весам \(w_{i-1}\); propagate \(\mathbf x_i^{(j)}\sim p(\mathbf x_i\mid\mathbf x_{i-1}^{(j)})\); weight и normalize:

$$ \tilde w_i^{(j)} = p(\mathbf z_i\mid\mathbf x_i^{(j)}), \qquad w_i^{(j)} = \frac{\tilde w_i^{(j)}}{\sum_{k=1}^{K}\tilde w_i^{(k)}}. $$

Без resampling это SIS, а не SIR. После resample веса равны \(1/K\), поэтому

$$ s_i=-\log\left(\frac{1}{K}\sum_{j=1}^{K}p(\mathbf z_i\mid\mathbf x_i^{(j)})\right). $$

## Частный случай для \(P\) и \(G\) ##

Та же модель постоянной скорости и тот же предел \(\Delta t\le 45\,\mathrm{с}\), но с \(K\) гипотезами. \(\hat a_i=\mathbf1[s_i>\tau]\). Если \(\Delta t_i>180\,\mathrm{с}\) или \(s_i>40\), частицы после записи score переинициализируются в точке измерения.

**Источник:** Arulampalam M. S., Maskell S., Gordon N., Clapp T. A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking // IEEE Transactions on Signal Processing. 2002. Vol. 50, No. 2. P. 174–188. DOI: [10.1109/78.978374](https://doi.org/10.1109/78.978374). [PDF](docs/methods/A%20Tutorial%20on%20Particle%20Filters%20for%20Online.pdf).

# Gaussian Process Regression #

Нормальная траектория описывается функцией времени с оценкой неопределённости. Точка аномальна, если лежит далеко от этой функции относительно предсказанного разброса.

## Общая математическая постановка ##

$$ f(t)\sim GP(m(t),k(t,t')), \qquad k(t,t') = \sigma_f^2 \exp \left( -\frac{(t-t')^2}{2\ell^2} \right). $$

$$ z_i=f(t_i)+\varepsilon_i, \qquad \varepsilon_i\sim\mathcal N(0,\sigma_n^2), \qquad f(t_*)\mid\mathcal D \sim \mathcal N(\mu_*,\sigma_*^2). $$

## Частный случай для \(P\) и \(G\) ##

Два независимых GP, \(f_x(t)\) и \(f_y(t)\), описывают шаг \((\Delta x,\Delta y)\) в метрах, не положение на карте. Обучение — подвыборка точек \(y_i>0\), не весь ряд и не тестовые наблюдения. Вдали по времени от опорных точек прогноз сходится к среднему шагу, поэтому другой район сам по себе score не поднимает:

$$ s_i=\frac{(\Delta x_i-\mu_x(t_i))^2}{\sigma_x^2(t_i)}+\frac{(\Delta y_i-\mu_y(t_i))^2}{\sigma_y^2(t_i)}. $$

При независимых нормальных остатках \(s_i\sim\chi^2_2\). Ненулевая связь координат требует multi-output GP.

**Источники:**

- Smith M., Reece S., Roberts S., Psorakis I., Rezek I. Maritime Abnormality Detection Using Gaussian Processes // Knowledge and Information Systems. 2014. Vol. 38, No. 3. P. 717–741. DOI: [10.1007/s10115-013-0685-z](https://doi.org/10.1007/s10115-013-0685-z).
- Penacho Riveiros A., Bastianello N., Barreau M. Model-free Anomaly Detection for Dynamical Systems with Gaussian Processes. 2026. 6 p. [PDF](docs/methods/Model-free%20Anomaly%20Detection%20for.pdf).

# GP-EKF, GP-BayesFilters #

Ko и Fox не детектируют аномалии: они подставляют гауссовские процессы вместо параметрических моделей прогноза и наблюдения внутрь фильтра Байеса. В статье три фильтра: GP-PF, GP-EKF и GP-UKF. В опыт входит GP-EKF. Его отличие от обычного EKF — якобиан среднего GP, а не якобиан заранее заданной функции.

В статье нет входа управления, отдельного от состояния. У трека управления нет, поэтому аргумент процесса — скорость и шаг времени. Наблюдение ГНСС линейно по положению, второй GP для него не учится: в статье GP-наблюдение нужно, когда отображение состояния в измерение неизвестно.

## Общая математическая постановка ##

Фильтр Байеса:

$$ p(x_k\mid z_{1:k},u_{1:k-1}) \propto p(z_k\mid x_k)\int p(x_k\mid x_{k-1},u_{k-1})\,p(x_{k-1}\mid z_{1:k-1})\,dx_{k-1}. $$

Прогноз GP — гауссовское распределение перехода. Наблюдение в статье тоже GP. Для скалярного выхода и квадратично-экспоненциального ядра

$$ k(x,x')=\sigma_f^2\exp\Bigl(-\frac12(x-x')W(x-x')^T\Bigr)+\sigma_n^2\delta, $$

$$ \mathrm{GP}_\mu(x_*,D)=k_*^T K^{-1}y, \qquad \mathrm{GP}_\sigma(x_*,D)=k(x_*,x_*)-k_*^T K^{-1}k_*. $$

Якобиан среднего по входу, нужный EKF:

$$ \frac{\partial \mathrm{GP}_\mu(x_*,D)}{\partial x_*} = \frac{\partial k_*}{\partial x_*}^{\!T} K^{-1}y, \qquad \frac{\partial k(x_*,x)}{\partial x_*[i]}=-W_{ii}(x_*[i]-x[i])\,\sigma_f^2\exp\Bigl(-\frac12(x_*-x)W(x_*-x)^T\Bigr). $$

Шаг GP-EKF. Процесс учит приращение состояния, поэтому линеаризация содержит единичную матрицу:

$$ \bar\mu_k=\mu_{k-1}+\mathrm{GP}_\mu([\mu_{k-1},u_{k-1}],D_p), \qquad Q_k=\mathrm{GP}_\sigma([\mu_{k-1},u_{k-1}],D_p), $$

$$ G_k=I+\frac{\partial\mathrm{GP}_\mu}{\partial x_{k-1}}, \qquad \bar\Sigma_k=G_k\Sigma_{k-1}G_k^T+Q_k. $$

Дальше обычное обновление EKF: прогноз измерения, \(R_k\), якобиан наблюдения \(H_k\), коэффициент Калмана, новое среднее и ковариация. Выходы по координатам — независимые GP, поэтому \(Q_k\) диагональна.

Усиленный GP (enhanced GP) учит не само приращение, а остаток после параметрической модели \(f\):

$$ \Delta\tilde x_k=x_{k+1}-x_k-f(x_k,u_k). $$

## Частный случай для \(P\) и \(G\) ##

Состояние в \((E,N)\), как у фильтра Калмана: положение и скорость. Параметрическая модель — постоянная скорость, шаг в ней ограничен \(\Delta t=\min(\Delta t_i,45\,\mathrm{с})\):

$$ \mathbf x_i=\begin{pmatrix}E\\ N\\ v^E\\ v^N\end{pmatrix}_i, \qquad f(\mathbf x,\Delta t)=\begin{pmatrix}v^E\Delta t\\ v^N\Delta t\\ 0\\ 0\end{pmatrix}. $$

Вход GP — \((v^E,v^N,\Delta t)\), не координаты на карте: другой район сам по себе не выглядит как отсутствие обучающих данных. Четыре независимых GP учатся на переходах \(y_i>0\to y_{i+1}>0\). Опорных переходов не больше 160, они берутся равномерно. Длина корреляции и шум ядра выбираются сеткой по логарифму маргинального правдоподобия. Наблюдение

$$ \mathbf z_i=H\mathbf x_i+\mathbf v_i, \qquad H=\begin{pmatrix}1&0&0&0\\ 0&1&0&0\end{pmatrix}, \qquad R=\sigma_z^2 I_2, \quad \sigma_z=40\,\mathrm{м}. $$

$$ s_i=\mathbf r_i^T S_i^{-1}\mathbf r_i, \qquad \mathbf r_i=\mathbf z_i-H\bar{\mathbf x}_i, \qquad S_i=H\bar P_i H^T+R, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

При согласованной гауссовской модели \(s_i\sim\chi^2_2\). Квантиль \(\chi^2\) в порог не входит. Если \(\Delta t_i>180\,\mathrm{с}\), инновация шага сохраняется, затем состояние заменяется измерением, а скорость обнуляется.

**Источник:** Ko J., Fox D. GP-BayesFilters: Bayesian Filtering Using Gaussian Process Prediction and Observation Models // Autonomous Robots. 2009. Vol. 27, No. 1. P. 75–90. DOI: [10.1007/s10514-009-9119-x](https://doi.org/10.1007/s10514-009-9119-x). [PDF](docs/GP-BayesFilters_Bayesian_Filtering_Using_Gaussian_.pdf).

# Hidden Markov Model #

Toloue и Jahan строят пять HMM с дискретными наблюдениями \((S,\Delta S,\mathrm{Drift},\Delta C)\): unexpected stop, risky speed, drift, spiral, normal. Модели обучаются Baum–Welch и сравниваются алгоритмом Forward на всей траектории.

Point-wise адаптация той же идеи, а не формула статьи: гауссовские HMM и отношение локальных predictive likelihood нормального и аномальных режимов.

$$ o_i=\mathrm{scale}_{\mathrm{MAD}}(v_i,a_i^{\mathrm{acc}},\Delta\psi_i). $$

## Общая математическая постановка ##

$$ S_i\in\{1,\ldots,K\}, \qquad \pi_k=P(S_1=k), \qquad A_{jk}=P(S_i=k\mid S_{i-1}=j). $$

$$ o_i\mid S_i=k \sim \mathcal N(\mu_k,\mathrm{diag}(\sigma_k^2)). $$

$$ p(o_{1:N}\mid\theta) = \sum_{s_{1:N}} \pi_{s_1} \prod_{i=2}^{N}A_{s_{i-1}s_i} \prod_{i=1}^{N} p(o_i\mid S_i=s_i). $$

## Частный случай ##

Три диагональные модели: \(\theta_N\) на штатных отрезках, \(\theta_A\) на аномальных, \(\theta_F\) на аномальных отрезках с \(v\ge 8\,\mathrm{м/с}\), если такие есть. Point-wise score — максимум разности предсказательных логарифмов:

$$ s_i=\max_{k\in\{A,F\}} \log P(o_i\mid o_{1:i-1},\theta_k)-\log P(o_i\mid o_{1:i-1},\theta_N), \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Toloue K. F., Jahan M. V. Anomalous Behavior Detection of Marine Vessels Based on Hidden Markov Model // 2018 6th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2018. P. 10–12. [PDF](docs/methods/AnomalousBehaviorDetectionofMarineVessels.pdf).

# Hampel Filter #

В скользящем окне точка сравнивается с медианой соседей. Выброс — сильное отклонение относительно типичного разброса окна.

Roos-Hoefgeest Toribio et al. ускоряют фильтр, заменяя MAD оценкой mMAD. Ниже — стандартный Hampel, не эта замена. Центрированное окно \(\{x_{i-h},\ldots,x_{i+h}\}\) использует будущие точки. Для online-потока нужно хвостовое окно либо задержка решения на \(h\) отсчётов.

## Общая математическая постановка ##

$$ W_i=\{x_{i-h},\ldots,x_i,\ldots,x_{i+h}\}, \qquad m_i=\operatorname{median}(W_i), $$

$$ MAD_i=\operatorname{median}_{x\in W_i}|x-m_i|, \qquad s_i=\frac{|x_i-m_i|}{1.4826\,MAD_i+\varepsilon}. $$

Точка аномальна при \(s_i>\tau\).

## Частный случай ##

Фильтр применяется к скорости, не к широте и долготе. Полуокно \(h=10\), у краёв ряд дополняется крайним значением:

$$ s_i=\frac{|v_i-\operatorname{med}(v_{i-h:i+h})|}{1.4826\,MAD(v_{i-h:i+h})+\varepsilon}, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

Метод рассчитан на единичные выбросы, не на длинный аномальный режим.

**Источник:** Roos-Hoefgeest Toribio M., Garnung Menéndez A., Roos-Hoefgeest Toribio S., Álvarez García I. A Novel Approach to Speed Up Hampel Filter for Outlier Detection // Sensors. 2025. Vol. 25. Article 3319. DOI: [10.3390/s25113319](https://doi.org/10.3390/s25113319). [PDF](docs/methods/A%20Novel%20Approach%20to%20Speed%20Up%20Hampel%20Filter%20for%20Outlier%20Detection.pdf).

# DBSCAN #

Точки группируются по плотности соседей. Аномалии — точки вне плотных кластеров. Это batch-метод: окрестность считается по всему набору, не по префиксу траектории.

## Общая математическая постановка ##

$$ N_\varepsilon(x)=\{x'\in X:d(x,x')\le\varepsilon\}. $$

Точка — core point при \(|N_\varepsilon(x)|\ge\mathrm{MinPts}\). Шум: \(c_i=-1\).

## Частный случай ##

Han et al. кластеризуют AIS по \((\mathrm{lat},\mathrm{lon},\mathrm{SOG},\mathrm{COG},\mathrm{Heading})\). В опыте абсолютные координаты не используются. Кластеры строятся по всему размечаемому ряду, на тесте — по тестовому облаку:

$$ \mathbf f_i=\bigl(\Delta x_i,\Delta y_i,v_i,a_i^{\mathrm{acc}},\Delta\psi_i,\log\max(\mathrm{shape}_i,1)\bigr), \qquad \mathbf u_i=\mathrm{robust}(\mathbf f_i). $$

$$ \hat a_i=\mathbf1[c_i=-1]. $$

Непрерывный score для PR-AUC — адаптация, не формула Ester et al. \(d_k(\mathbf u_i)\) — расстояние до \(\mathrm{MinPts}\)-го соседа, считая саму точку; \(d_{\mathrm{core}}\) — расстояние до ближайшей core-точки (если core-точек нет, берётся \(d_k\)):

$$ s_i=\max\bigl(d_k(\mathbf u_i),\, d_{\mathrm{core}}(\mathbf u_i)\bigr). $$

Больший \(s_i\) — точка сильнее изолирована от плотного режима \((\Delta x,\Delta y,v,a^{\mathrm{acc}},\Delta\psi,\mathrm{shape})\). Бинарная разметка по-прежнему \(\hat a_i=\mathbf1[c_i=-1]\).

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

$$ \mathbf f_i=\bigl( \Delta x_i, \Delta y_i, v_i, a_i^{\mathrm{acc}}, \Delta\psi_i \bigr), \qquad s_i=\mathrm{LOF}_k(\mathrm{robust}(\mathbf f_i)), \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

LOF пересчитывается на размечаемом ряду. Порог переносится с обучающей выборки, где счёт выполнен отдельно.

**Источник:** Breunig M. M., Kriegel H.-P., Ng R. T., Sander J. LOF: Identifying Density-Based Local Outliers // Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. Dallas, USA. ACM, 2000. P. 93–104. DOI: [10.1145/342009.335388](https://doi.org/10.1145/342009.335388). [PDF](<docs/methods/LOF_ Identifying Density-Based Local Outliers.pdf>).

# Isolation Forest #

Случайные деревья отделяют точки разрезами по признакам. Аномалия отделяется за меньшее число разрезов. Метод смотрит на весь набор и не требует отдельной выборки только нормальных точек.

## Общая математическая постановка ##

Средняя длина пути \(E[h(x)]\), нормировка через гармоническое число \(H\):

$$ c(n)=2H(n-1)-\frac{2(n-1)}{n}, \qquad s(x)=2^{-\frac{E[h(x)]}{c(n)}}. $$

## Частный случай ##

Лес строится по всей обучающей выборке и тем же лесом оценивается тест. Абсолютные координаты не входят:

$$ \mathbf f_i=\bigl(\Delta x_i,\Delta y_i,v_i,a_i^{\mathrm{acc}},\Delta\psi_i,\log\max(\mathrm{shape}_i,1)\bigr). $$

В sklearn при \(\mathrm{contamination}=\mathrm{auto}\) в порог попадает score статьи, сдвинутый на \(1/2\). Здесь \(n\) — размер подвыборки дерева:

$$ s_i=2^{-E[h(\mathrm{robust}(\mathbf f_i))]/c(n)}-\frac12, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Liu F. T., Ting K. M., Zhou Z.-H. Isolation-Based Anomaly Detection // ACM Transactions on Knowledge Discovery from Data. 2012. Vol. 6, No. 1. Article 3. 39 p. DOI: [10.1145/2133360.2133363](https://doi.org/10.1145/2133360.2133363). [PDF](docs/methods/Isolation-Based%20Anomaly%20Detection.pdf).

# One-Class SVM #

На нормальных точках оценивается область высокой концентрации обучающего распределения. Объект вне этой области считается аномалией.

## Общая математическая постановка ##

$$ \min_{\mathbf w,\rho,\xi} \frac12\|\mathbf w\|^2 + \frac{1}{\nu n} \sum_{i=1}^{n}\xi_i -\rho, \qquad \mathbf w^T\phi(x_i)\ge\rho-\xi_i, \quad \xi_i\ge0. $$

Параметр \(\nu\in(0,1]\) — верхняя граница доли ошибок на обучении и нижняя граница доли support vectors.

$$ f(x)=\sum_i\alpha_i K(x_i,x)-\rho. $$

Аномалия при \(f(x)<0\).

## Частный случай ##

Обучение только на подвыборке \(\{\mathbf f_i:y_i>0\}\), не больше \(4000\) точек. Абсолютные координаты не входят. Ядро RBF. Порог score фиксирован нулём, \(\nu\) — гиперпараметр:

$$ \mathbf f_i=\bigl(\Delta x_i,\Delta y_i,v_i,a_i^{\mathrm{acc}},\Delta\psi_i,\log\max(\mathrm{shape}_i,1)\bigr), \qquad s_i=-f(\mathrm{robust}(\mathbf f_i)), \qquad \hat a_i=\mathbf1[s_i>0]. $$

**Источник:** Schölkopf B., Platt J. C., Shawe-Taylor J., Smola A. J., Williamson R. C. Estimating the Support of a High-Dimensional Distribution // Neural Computation. 2001. Vol. 13, No. 7. P. 1443–1471. DOI: [10.1162/089976601750264965](https://doi.org/10.1162/089976601750264965). В `docs/methods/` лежит технический отчёт Microsoft Research MSR-TR-99-87 (27 November 1999; revised 18 September 2000): [PDF](docs/methods/Estimating%20the%20Support%20of%20a%20High-Dimensional%20Distribution.pdf).

# Robust PCA / Principal Component Pursuit #

Матрица раскладывается на низкоранговую часть и разреженные отклонения. Крупные элементы разреженной части — кандидаты в аномалии. Это batch-разложение матрицы, не causal-прогноз точки.

## Общая математическая постановка ##

$$ X=L+S, \qquad \min_{L,S} \|L\|_* + \lambda\|S\|_1. $$

В классической постановке Candès et al. \(\lambda=1/\sqrt{\max(n_1,n_2)}\).

## Частный случай ##

Одна последовательность \((t_i,\mathrm{lat}_i,\mathrm{lon}_i)\) в PCP не подаётся. Матрица собирается из окон длины \(16\) с шагом \(8\); окно не пересекает разрыв \(\Delta t>180\,\mathrm{с}\). Признаки строки — \((\Delta x/1000,\Delta y/1000,v,\Delta\psi)\). Ряд, который размечается, раскладывается сам; шкала — робастная по этим строкам.

\(\|S_W(i)\|\) — евклидова норма разреженных компонент точки \(i\) внутри окна. Точке присваивается максимум, не среднее. Точка вне окон получает \(0\):

$$ s_i=\max_{W:\, i\in W} \|S_W(i)\|, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Candès E. J., Li X., Ma Y., Wright J. Robust Principal Component Analysis? // Journal of the ACM. 2011. Vol. 58, No. 3. Article 11. P. 1–37. DOI: [10.1145/1970392.1970395](https://doi.org/10.1145/1970392.1970395). В `docs/methods/` лежит препринт arXiv:0912.3599v1 (18 December 2009): [PDF](docs/methods/Robust%20Principal%20Component%20Analysis.pdf).

# LSTM Autoencoder / EncDec-AD #

Сеть учится восстанавливать окна нормальной траектории. Аномалия — окно с нетипичной ошибкой восстановления. Окно не является causal-прогнозом одной точки.

## Общая математическая постановка ##

$$ X_i=(x_{i-L+1},\ldots,x_i), \qquad h_i=E_\theta(X_i), \qquad \hat X_i=D_\phi(h_i). $$

Обучение: \(\min_{\theta,\phi}\sum_i \|X_i-\hat X_i\|_2^2\).

В EncDec-AD Malhotra et al. вектор ошибки \(e=X-\hat X\) на нормальной проверочной выборке описывается средним \(\mu\) и ковариацией \(\Sigma\), score — расстояние Махаланобиса. Сырой \(\|e\|_2^2\) — упрощение без этой нормировки.

## Частный случай ##

Окно — положение в километрах относительно первой точки окна. Декодер читает нулевой вход и финальное состояние энкодера. \(\mu_j\) и диагональные \(\sigma_j^2\) оцениваются на штатных окнах, не входивших в обучение:

$$ s_W=\sum_j\frac{(e_j-\mu_j)^2}{\sigma_j^2}, \qquad s_i=\max_{W\ni i}s_W, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Malhotra P., Ramakrishnan A., Anand G., Vig L., Agarwal P., Shroff G. LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection // ICML 2016 Anomaly Detection Workshop. New York, 2016. arXiv: [1607.00148](https://arxiv.org/abs/1607.00148). [PDF](docs/methods/LSTM-based%20Encoder-Decoder%20for%20Multi-sensor%20Anomaly%20Detection.pdf).

# Temporal Convolutional Network #

Каузальная свёрточная сеть предсказывает продолжение нормального ряда. Аномалия — остаток, редкий для распределения ошибок на нормальных данных.

He и Zhao прогнозируют несколько шагов вперёд и смешивают признаки нескольких масштабов. Один шаг — частный случай.

## Общая математическая постановка ##

$$ \hat x_{t+1:t+H}=f_\theta(x_{t-L+1},\ldots,x_t), \qquad r_{t+h}=x_{t+h}-\hat x_{t+h}. $$

Остатки на нормальных данных описываются диагональным гауссовским законом. Нулевое среднее не предполагается: \(\mu\) и \(\sigma^2\) оцениваются.

## Частный случай ##

Окно — положение в километрах относительно его первой точки. Сеть предсказывает следующую точку в той же системе. Score пишется только в неё; если окно не собралось, \(s_i=0\):

$$ s_i=\sum_{c=1}^{2}\frac{(r_{i,c}-\mu_c)^2}{\sigma_c^2}, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** He Y., Zhao J. Temporal Convolutional Networks for Anomaly Detection in Time Series // Journal of Physics: Conference Series. 2019. Vol. 1213. Article 042050. DOI: [10.1088/1742-6596/1213/4/042050](https://doi.org/10.1088/1742-6596/1213/4/042050). [PDF](docs/methods/Temporal_Convolutional_Networks_for_Anomaly_Detect.pdf).

# DONUT — VAE для anomaly detection #

Вариационный автоэнкодер задаёт распределение окон. Окно с низкой вероятностью считается аномальным. DONUT сделан для сезонных KPI веб-приложений: modified ELBO, учёт пропусков и KDE-интерпретация реконструкции. Формулы ниже — общая VAE-постановка; перенос на окна траектории — адаптация, не алгоритм статьи.

## Общая математическая постановка ##

$$ q_\phi(z|x)=\mathcal N\left( \mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)) \right), \qquad p_\theta(x|z). $$

$$ \mathcal L(x)=\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}\left( q_\phi(z|x)\|p(z) \right). $$

Общий probabilistic score \(s=-\log p_\theta(x)\) не является формулой опыта.

## Частный случай ##

Окно положения в километрах относительно первой точки записывается одним вектором и нормируется по окнам \(y_i>0\). Реконструкция берётся в среднем энкодера, без выборки \(z\). Точке присваивается максимум по покрывающим окнам:

$$ s_W=\|x-\hat x\|_2^2+D_{\mathrm{KL}}\bigl(q_\phi(z|x)\,\|\,p(z)\bigr), \qquad s_i=\max_{W\ni i}s_W, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Xu H., Chen W., Zhao N., Li Z., Bu J., Li Z., Liu Y., Zhao Y., Pei D., Feng Y., Chen J., Wang Z., Qiao H. Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications // Proceedings of The Web Conference 2018 (WWW '18). Lyon, France, 23–27 April 2018. ACM, 2018. 12 p. DOI: [10.1145/3178876.3185996](https://doi.org/10.1145/3178876.3185996). arXiv: [1802.03903](https://arxiv.org/abs/1802.03903). [PDF](docs/methods/Unsupervised%20Anomaly%20Detection%20via%20Variational%20Auto-Encoder.pdf).

# Anomaly Transformer #

Для окна последовательности модель совмещает ошибку восстановления и расхождение ассоциаций attention с prior-ассоциацией. Из-за adjacent-concentration bias аномалия часто имеет меньшую association discrepancy, поэтому в итоговом score стоит \(\mathrm{Softmax}(-\mathrm{AssDis})\), а не само расхождение.

## Общая математическая постановка ##

Окно \(X=(x_{1},\ldots,x_{L_{\mathrm{win}}})\). В слое \(\ell\) ассоциации

$$ A^{(\ell)}_{ij}=\operatorname{softmax}\left( \frac{Q_iK_j^T}{\sqrt d} \right) $$

\(L\) — число слоёв, не длина окна. Prior — гауссово ядро по расстоянию индексов, \(\sigma_\ell=\mathrm{softplus}(\rho_\ell)+1\):

$$ P^{(\ell)}_{ij}=\operatorname{softmax}_j\Bigl(-\frac{(i-j)^2}{2\sigma_\ell^2}\Bigr). $$

$$ \mathrm{AssDis}_i=\frac{1}{L}\sum_{\ell=1}^{L}\frac12\Bigl[\mathrm{KL}(A^{(\ell)}_i\|P^{(\ell)}_i)+\mathrm{KL}(P^{(\ell)}_i\|A^{(\ell)}_i)\Bigr]. $$

Итоговый критерий позиции внутри окна:

$$ \mathrm{AnomalyScore}_i(X)=\mathrm{Softmax}(-\mathrm{AssDis})_i\,\|x_i-\hat x_i\|_2^2. $$

Обучение минимизирует \(\|X-\hat X\|_2^2\). Параметр \(\sigma_\ell\) в функцию потерь не входит.

## Частный случай ##

$$ x_i=\bigl(\Delta x_i/1000,\ \Delta y_i/1000,\ v_i,\ \Delta\psi_i\bigr). $$

Признаки нормируются по окнам \(y_i>0\). Точке присваивается максимум \(\mathrm{AnomalyScore}\) по покрывающим окнам: \(s_i=\max_{W\ni i}\mathrm{AnomalyScore}_i(W)\), \(\hat a_i=\mathbf1[s_i>\tau]\).

**Источник:** Xu J., Wu H., Wang J., Long M. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy // International Conference on Learning Representations (ICLR). 2022. arXiv: [2110.02642](https://arxiv.org/abs/2110.02642). [PDF](docs/methods/ANOMALY%20TRANSFORMER%20TIME%20SERIES%20ANOMALY.pdf).

# Spatio-Temporal GNN — STGVAD #

В статье состояния нескольких судов — вершины одного графа: пространственные связи задаёт OPTICS, временные соединяют состояния внутри окна не только с соседом по времени. Идентификатора судна в ряде нет, поэтому опыт строится на окне одной последовательности.

## Постановка опыта ##

Окно длины \(L\). Вершина несёт \(u_i=(v_i/10,\Delta\psi_i)\). Признак соседа относительно начала окна: \(n_j=(\Delta x_j/1000,\Delta y_j/1000,v_j/10,\Delta\psi_j)\). Ребро \(j\to i\) есть при \(j<i\) и расстоянии меньше \(5\,\mathrm{км}\); строки \(A\) нормированы на единицу. Собственные координаты в \(u_i\) не входят.

$$ h^{(1)}=\mathrm{ReLU}\bigl(W_0^{(1)}u+W_1^{(1)}AN\bigr), \qquad h^{(2)}=\mathrm{ReLU}\bigl(W_0^{(2)}h^{(1)}+W_1^{(2)}Ah^{(1)}\bigr), \qquad \hat r=W_2h^{(2)}. $$

\(\hat r\) и \(r\) — положение в километрах от начала окна. Ошибка первой точки окна равна \(0\). Точке присваивается максимум по покрывающим окнам:

$$ s_i=\max_{W\ni i}1000\,\|\hat r_i-r_i\|, \qquad \hat a_i=\mathbf1[s_i>\tau]. $$

**Источник:** Kim J., Kim M., Hwang Y., Bae S., Cho D. J., Lee W., Park H. STGVAD: Spatio-Temporal Graph-based Vessel Behavior Anomaly Detection // IEEE Access. 2026. Vol. 14. P. 2152–2165. DOI: [10.1109/ACCESS.2025.3609783](https://doi.org/10.1109/ACCESS.2025.3609783) (принята в 2025). [PDF](docs/methods/STGVAD_Spatio-Temporal_Graph-Based_Vessel_Behavior.pdf).

# Maximum Physically Consistent Trajectory #

Custers et al. ищут самую длинную подпоследовательность со скоростью не выше \(v_+\). Опыт — не этот глобальный максимум, а жадный якорь: отклонённая точка якорь не сдвигает, дальнее облако новым штатным участком не становится. Дополнительно ограничена форма окна.

Якорь \(j\) — последняя точка с \(\hat a_j=0\). Для \(i>j\)

$$ v_{ij}=\frac{d_{\mathrm{geo}}(p_j,p_i)}{\max(t_i-t_j,\,1)}, \qquad \hat a_i=\mathbf1\bigl[v_{ij}>v_+\ \lor\ \mathrm{shape}_i>\tau_{\mathrm{shape}}\bigr]. $$

При \(\hat a_i=0\) якорь переходит в \(i\). Иначе остаётся. Непрерывный score для PR-AUC у принятой точки равен \(0\), иначе

$$ s_i=\max(0,\,v_{ij}-v_+)+\max(0,\,\mathrm{shape}_i-\tau_{\mathrm{shape}})+\frac{d_{\mathrm{geo}}(p_j,p_i)}{1000}. $$

Слагаемые — превышение скорости в м/с, превышение формы и смещение от якоря в километрах.

**Источник:** Custers B., van de Kerkhof M., Meulemans W., Speckmann B., Staals F. Maximum Physically Consistent Trajectories // ACM Transactions on Spatial Algorithms and Systems. 2021. Vol. 7, No. 4. Article 17. 33 p. DOI: [10.1145/3452378](https://doi.org/10.1145/3452378). [PDF](docs/methods/Maximum%20Physically%20Consistent%20Trajectories.pdf).
