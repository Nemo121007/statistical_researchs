# Обозначения #

$$ P=\{(t_i,\mathrm{lat}_i,\mathrm{lon}_i,\hat y_i)\}_{i=1}^{N}, \qquad G=\{(t_i,\mathrm{lat}_i,\mathrm{lon}_i,y_i)\}_{i=1}^{N}. $$

\(P\) — prediction, \(G\) — ground truth. \(y_i=0\) и \(\hat y_i=0\) — аномалия, положительная метка — штатное состояние.

$$ a_i=\mathbf1[y_i=0], \qquad \hat a_i=\mathbf1[\hat y_i=0]. $$

\(\hat a_i=1\) — предсказанная аномалия. В исходных метках аномалия — класс \(0\).

\(d_{\mathrm{geo}}\) и длина участка считаются по парам \((\mathrm{lat},\mathrm{lon})\). Время в аргумент расстояния не входит. Пространственная интерполяция выполняется в локальной метрической проекции \(p=\mathrm{Project}(\mathrm{lat},\mathrm{lon})\), не по сырым градусам.

Метрики считаются по готовой паре \((P,G)\). Порядок обнаружения интерпретирует только Detection Delay.

# Point-wise F1 #

Баланс полноты и точности бинарной классификации точек. Время и координаты в расчёт не входят.

$$ \mathrm{Precision}=\frac{TP}{TP+FP}, \qquad \mathrm{Recall}=\frac{TP}{TP+FN}, \qquad F_1=\frac{2TP}{2TP+FP+FN}. $$

$$ TP=\sum_i\mathbf1[a_i=1\land\hat a_i=1], \quad FP=\sum_i\mathbf1[a_i=0\land\hat a_i=1], \quad FN=\sum_i\mathbf1[a_i=1\land\hat a_i=0]. $$

При \(TP+FP=0\) или \(TP+FN=0\) соответствующая доля равна \(0\).

**Источники:** Powers D. M. W. Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation // Journal of Machine Learning Technologies. 2011. Vol. 2, No. 1. P. 37–63. Schmidl S., Wenig P., Papenbrock T. Anomaly Detection in Time Series: A Comprehensive Evaluation // Proc. VLDB Endowment. 2022. Vol. 15, No. 9. P. 1779–1797. DOI: [10.14778/3538598.3538602](https://doi.org/10.14778/3538598.3538602). Sørbø S., Ruocco M. Navigating the Metric Maze // Data Mining and Knowledge Discovery. 2024. Vol. 38. P. 1027–1068. DOI: [10.1007/s10618-023-00988-8](https://doi.org/10.1007/s10618-023-00988-8).

# PR-AUC #

Площадь под кривой Precision–Recall по непрерывному anomaly score \(s_i\): большее значение означает более аномальную точку. Бинарной маски \(\hat y_i\) недостаточно.

Для порога \(\tau\), \(\hat a_i(\tau)=\mathbf1[s_i\ge\tau]\):

$$ \mathrm{Precision}(\tau)=\frac{TP(\tau)}{TP(\tau)+FP(\tau)}, \qquad \mathrm{Recall}(\tau)=\frac{TP(\tau)}{TP(\tau)+FN(\tau)}, $$

$$ PR\text{-}AUC=\int_0^1\mathrm{Precision}(r)\,dr. $$

Интеграл считается численно по кривой, построенной по всем достижимым порогам. Это площадь под PR-кривой. Average precision — другой распространённый оценщик того же графика, и его не следует подставлять под знак интеграла без оговорки.

**Источники:** Saito T., Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets // PLOS ONE. 2015. Vol. 10, No. 3. e0118432. DOI: [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432). Schmidl et al., 2022, DOI: [10.14778/3538598.3538602](https://doi.org/10.14778/3538598.3538602). Sørbø, Ruocco, 2024, DOI: [10.1007/s10618-023-00988-8](https://doi.org/10.1007/s10618-023-00988-8).

# Affiliation F1 #

Event-level метрика Huet et al. Сравниваются истинные и предсказанные аномальные интервалы по времени. Географическое расстояние не используется.

Ось времени делится на зоны \(I_j\): каждый момент относится к ближайшему истинному событию \(gt_j\), \(j=1,\ldots,m\). Пусть \(pred\) — объединение предсказанных аномальных интервалов. На зоне считаются средние направленные расстояния по времени:

$$ D^{\mathrm{prec}}_j=\mathrm{dist}(pred\cap I_j,\ gt_j), \qquad D^{\mathrm{rec}}_j=\mathrm{dist}(gt_j,\ pred\cap I_j). $$

Расстояние переводится в вероятность сравнением с равномерной случайной точкой в той же зоне (survival function; замкнутый вид — в приложении Huet et al.). Precision усредняется только по событиям \(S\), к которым попало хотя бы одно предсказание. Recall усредняется по всем \(m\) истинным событиям:

$$ \mathrm{Precision}=\frac{1}{|S|}\sum_{j\in S}p_j, \qquad \mathrm{Recall}=\frac{1}{m}\sum_{j=1}^{m}r_j, $$

$$ F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}. $$

При \(S=\emptyset\) precision не определяется. Усреднение обеих величин по одному и тому же \(m\) этой метрике не соответствует.

Поскольку \(t_i\) у \(P\) и \(G\) совпадают, зоны строятся по индексам непрерывных интервалов \(\{i:y_i=0\}\) и \(\{i:\hat y_i=0\}\).

**Источник:** Huet A., Navarro J. M., Rossi D. Local Evaluation of Time Series Anomaly Detection Algorithms // Proc. 28th ACM SIGKDD (KDD ’22). 2022. P. 3873–3883. DOI: [10.1145/3534678.3539339](https://doi.org/10.1145/3534678.3539339).

# Detection Delay #

Задержка от начала истинного эпизода до первой детекции внутри него. NAB (Lavin, Ahmad) эту величину не определяет: там раннее обнаружение входит в гладкий score внутри anomaly window. Ниже — отдельное определение для этого benchmark.

Для непрерывного интервала \(S_k=\{i:y_i=0\}\):

$$ t_k^{\mathrm{start}}=\min_{i\in S_k}t_i, \qquad t_k^{\mathrm{detect}}=\min\{t_i:i\in S_k,\ \hat y_i=0\}, $$

$$ \mathrm{Delay}_k=t_k^{\mathrm{detect}}-t_k^{\mathrm{start}}. $$

Если внутри \(S_k\) аномалия не предсказана, событие undetected. Нулевой задержкой оно не заменяется. По обнаруженным событиям считаются медиана, среднее и 95-й перцентиль.

**Источники:** Lavin A., Ahmad S. Evaluating Real-Time Anomaly Detection Algorithms – The Numenta Anomaly Benchmark // ICMLA. 2015. P. 38–44. DOI: [10.1109/ICMLA.2015.141](https://doi.org/10.1109/ICMLA.2015.141). Scharwächter E., Müller E. Statistical Evaluation of Anomaly Detectors for Sequences // KDD MiLeTS. 2020. arXiv: [2008.05788](https://arxiv.org/abs/2008.05788).

# Geodesic Distance RMSE #

ATE RMSE (Sturm et al.) — корень из среднего квадрата расстояний между синхронными позами двух траекторий. Здесь среднее берётся по пропускам, а опорная поза в разрыве строится интерполяцией.

Ближайшие штатные точки слева и справа от разрыва, \(y_l>0\), \(y_r>0\), переводятся в локальную проекцию. Для \(t_i\) внутри разрыва:

$$ \alpha_i=\frac{t_i-t_l}{t_r-t_l}, \qquad p_i^*=p_l+\alpha_i(p_r-p_l), \qquad d_i=\|p_i-p_i^*\|. $$

$$ S_{FN}=\{i:y_i=0\land\hat y_i\neq 0\}, \qquad RMSE_d=\sqrt{\frac{1}{|S_{FN}|}\sum_{i\in S_{FN}}d_i^2}. $$

В сумму входят аномальные точки, оставленные штатными. При пустом \(S_{FN}\) или при разрыве без пары штатных соседей \(RMSE_d\) не определяется.

**Источники:** Sturm J. et al. A Benchmark for the Evaluation of RGB-D SLAM Systems // IROS. 2012. P. 573–580. DOI: [10.1109/IROS.2012.6385773](https://doi.org/10.1109/IROS.2012.6385773). Sørbø, Ruocco, 2024, DOI: [10.1007/s10618-023-00988-8](https://doi.org/10.1007/s10618-023-00988-8).

# Hausdorff Distance #

Классическое расстояние между двумя множествами точек траектории, в том виде, в каком его считают Olesen et al. и Yu et al.:

$$ h(A,B)=\max_{a\in A}\min_{b\in B}d(a,b), \qquad H(A,B)=\max\{h(A,B),h(B,A)\}. $$

Оно не использует временную синхронизацию и определяется самой дальней парой.

Отдельно, по тем же пропускам и опорным точкам, что и \(RMSE_d\):

$$ H_{obs\to ref}=\max_{i\in S_{FN}}d_i. $$

Это направленный максимум ошибки пропуска, а не \(H(A,B)\).

**Источники:** Olesen K. V. et al. A Contextually Supported Abnormality Detector for Maritime Trajectories // Journal of Marine Science and Engineering. 2023. Vol. 11, No. 11. Article 2085. DOI: [10.3390/jmse11112085](https://doi.org/10.3390/jmse11112085). Yu C. et al. A Novel Trajectory Repairing Model Based on the Artificial Potential Field-Enhanced A* Algorithm for Small Coastal Vessels // Journal of Marine Science and Engineering. 2025. Vol. 13, No. 7. Article 1200. DOI: [10.3390/jmse13071200](https://doi.org/10.3390/jmse13071200).

# Distance Loss #

У Yu et al. это модуль разности полных длин двух траекторий:

$$ L(T)=\sum d(p_i,p_{i+1}), \qquad DL=|L(T)-L(T^*)|, \qquad DLR=\frac{DL}{L(T^*)}. $$

В этом benchmark длины считаются в локальной проекции. \(L_{ref}\) — по штатным точкам и интерполяции разрывов, \(L_{pred}\) — по последовательным точкам с \(\hat y_i\neq 0\):

$$ DL=|L_{pred}-L_{ref}|, \qquad DLR=\frac{DL}{L_{ref}}. $$

Петля среди точек, принятых за штатные, увеличивает \(L_{pred}\) относительно \(L_{ref}\). Метрика измеряет разность длин. При \(L_{ref}=0\) отношение \(DLR\) не определяется.

**Источник:** Yu C. et al., 2025, DOI: [10.3390/jmse13071200](https://doi.org/10.3390/jmse13071200).
