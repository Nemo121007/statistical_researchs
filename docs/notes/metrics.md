# Обозначения #

$$ P=\{(t_i,\mathrm{lon}_i,\mathrm{lat}_i,\hat y_i)\}_{i=1}^{N}, \qquad G=\{(t_i,\mathrm{lon}_i,\mathrm{lat}_i,y_i)\}_{i=1}^{N}. $$

$ P $ — prediction, $G$ — ground truth. $ y_i=0$ и $\hat y_i=0$ — аномалия, положительная метка — штатное состояние.

$$ a_i=\mathbf1[y_i=0], \qquad \hat a_i=\mathbf1[\hat y_i=0]. $$

$\hat a_i=1$ — предсказанная аномалия. В исходных метках аномалия — класс $0$.

$d_{\mathrm{geo}}$ — расстояние по большому кругу, $R=6371\,\mathrm{км}$. Время в аргумент не входит. В формуле углы в радианах, координаты ряда хранятся в градусах:

$$ d_{\mathrm{geo}}(p,q)=2R\arcsin\sqrt{\sin^2\frac{\Delta\varphi}{2}+\cos\varphi_1\cos\varphi_2\sin^2\frac{\Delta\lambda}{2}}. $$

Опорная точка разрыва строится линейно по широте и по кратчайшей дуге долготы. Для $\alpha\in[0,1]$ и $\mathrm{wrap}(\delta)=((\delta+180^\circ)\bmod 360^\circ)-180^\circ$

$$ \varphi^*=\varphi_l+\alpha(\varphi_r-\varphi_l), \qquad \lambda^*=\mathrm{wrap}\bigl(\lambda_l+\alpha\,\mathrm{wrap}(\lambda_r-\lambda_l)\bigr). $$

Метрики считаются по готовой паре $(P,G)$. Порядок обнаружения интерпретирует только Detection Delay.

# Point-wise F1 #

Баланс полноты и точности бинарной классификации точек. Время и координаты в расчёт не входят.

$$ \mathrm{Precision}=\frac{TP}{TP+FP}, \qquad \mathrm{Recall}=\frac{TP}{TP+FN}, \qquad F_1=\frac{2TP}{2TP+FP+FN}. $$

$$ TP=\sum_i\mathbf1[a_i=1\land\hat a_i=1], \quad FP=\sum_i\mathbf1[a_i=0\land\hat a_i=1], \quad FN=\sum_i\mathbf1[a_i=1\land\hat a_i=0]. $$

При нулевом знаменателе соответствующая доля не определяется. $F_1$ не определяется только при $TP=FP=FN=0$; при $TP=0$ и $FP+FN>0$ значение $F_1$ равно $0$.

**Источники**:

1. Powers D. M. W. *Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation* // Journal of Machine Learning Technologies. 2011. Vol. 2, No. 1. P. 37–63. DOI: [10.9735/2229-3981.2.1.37](https://doi.org/10.9735/2229-3981.2.1.37).
2. Schmidl S., Wenig P., Papenbrock T. *Anomaly Detection in Time Series: A Comprehensive Evaluation* // Proceedings of the VLDB Endowment. 2022. Vol. 15, No. 9. P. 1779–1797. DOI: [10.14778/3538598.3538602](https://doi.org/10.14778/3538598.3538602).
3. Sørbø S., Ruocco M. *Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series* // Data Mining and Knowledge Discovery. 2024. Vol. 38. P. 1027–1068. DOI: [10.1007/s10618-023-00988-8](https://doi.org/10.1007/s10618-023-00988-8).

# PR-AUC #

Площадь под кривой Precision–Recall по непрерывному anomaly score $s_i$: большее значение означает более аномальную точку. Бинарной маски $\hat y_i$ недостаточно.

Для порога $\tau$, $\hat a_i(\tau)=\mathbf1[s_i\ge\tau]$:

$$ \mathrm{Precision}(\tau)=\frac{TP(\tau)}{TP(\tau)+FP(\tau)}, \qquad \mathrm{Recall}(\tau)=\frac{TP(\tau)}{TP(\tau)+FN(\tau)}, $$

$$ PR\text{-}AUC=\int_0^1\mathrm{Precision}(r)\,dr. $$

Узлы кривой — различные значения $s_i$ по убыванию; кривая дополняется точкой $(0,1)$, интеграл считается трапециями. Это площадь под PR-кривой, не average precision. Если в $G$ нет аномалий или score не задан, $PR\text{-}AUC$ не определяется.

**Источники**:

1. Saito T., Rehmsmeier M. *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets* // PLOS ONE. 2015. Vol. 10, No. 3. Article e0118432. DOI: [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432).
2. Schmidl S., Wenig P., Papenbrock T. *Anomaly Detection in Time Series: A Comprehensive Evaluation* // Proceedings of the VLDB Endowment. 2022. Vol. 15, No. 9. P. 1779–1797. DOI: [10.14778/3538598.3538602](https://doi.org/10.14778/3538598.3538602).
3. Sørbø S., Ruocco M. *Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series* // Data Mining and Knowledge Discovery. 2024. Vol. 38. P. 1027–1068. DOI: [10.1007/s10618-023-00988-8](https://doi.org/10.1007/s10618-023-00988-8).

# Affiliation F1 #

Event-level метрика Huet et al. Сравниваются истинные и предсказанные аномальные интервалы по времени. Географическое расстояние не используется.

Ось времени делится на зоны $I_j$: каждый момент относится к ближайшему истинному событию $gt_j$, $j=1,\ldots,m$. Граница между $I_j$ и $I_{j+1}$ — середина паузы между соседними истинными интервалами; у краёв ряда зона тянется до $t_0$ и $t_N$ (правая граница ряда — с последним шагом дискретизации, как у событий). Ширина $|I_j|$ задаёт масштаб штрафа в локальных $p_j$, $r_j$. Пусть $pred$ — объединение предсказанных аномальных интервалов, обрезанных по $I_j$. На зоне считаются средние направленные расстояния по времени:

$$ D^{\mathrm{prec}}_j=\mathrm{dist}(pred\cap I_j,\ gt_j), \qquad D^{\mathrm{rec}}_j=\mathrm{dist}(gt_j,\ pred\cap I_j). $$

Расстояние переводится в вероятность сравнением с равномерной случайной точкой в той же зоне. Precision усредняется только по событиям $S$, к которым попало хотя бы одно предсказание. Recall усредняется по всем $m$ истинным событиям:

$$ \mathrm{Precision}=\frac{1}{|S|}\sum_{j\in S}p_j, \qquad \mathrm{Recall}=\frac{1}{m}\sum_{j=1}^{m}r_j, $$

$$ F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}. $$

При $S=\emptyset$ precision и $F_1$ не определяются. Если истинных событий нет, не определяются все три величины. Усреднение precision и recall по одному и тому же $m$ этой метрике не соответствует.

Поскольку $t_i$ у $P$ и $G$ совпадают, непрерывный участок $\{i:y_i=0\}$ становится полуинтервалом $[t_{\mathrm{start}},t_{\mathrm{end}+1})$. Если участок упирается в конец ряда, правая граница продолжается на последний шаг дискретизации.

**Источник**:
Huet A., Navarro J. M., Rossi D. *Local Evaluation of Time Series Anomaly Detection Algorithms* // Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22). Washington, DC, USA, 2022. P. 3873–3883. DOI: [10.1145/3534678.3539339](https://doi.org/10.1145/3534678.3539339).

# Detection Delay #

Задержка от начала истинного эпизода до первой детекции внутри него. NAB (Lavin, Ahmad) эту величину не определяет: там раннее обнаружение входит в гладкий score внутри anomaly window. Ниже — отдельное определение для этого benchmark.

Для непрерывного интервала $S_k=\{i:y_i=0\}$:

$$ t_k^{\mathrm{start}}=\min_{i\in S_k}t_i, \qquad t_k^{\mathrm{detect}}=\min\{t_i:i\in S_k,\ \hat y_i=0\}, $$

$$ \mathrm{Delay}_k=t_k^{\mathrm{detect}}-t_k^{\mathrm{start}}. $$

Если внутри $S_k$ аномалия не предсказана ($\hat y_i\neq 0$ на всём участке), событие undetected; положим $\mathrm{Delay}_k=t_k^{\mathrm{end}}-t_k^{\mathrm{start}}$, где $t_k^{\mathrm{end}}=\max_{i\in S_k}t_i$. Нулевой задержкой пропуск не заменяется.

Среднее, медиана, 95-й перцентиль и максимум считаются по всем эпизодам с $\mathrm{Delay}_k>0$: и по опозданиям (детекция есть, но не в $t_k^{\mathrm{start}}$), и по undetected (штраф — длина $S_k$). Эпизоды с детекцией в первой точке ($\mathrm{Delay}_k=0$) в эти агрегаты не входят. Дополнительно: $N_{\mathrm{on\,time}}$ — число эпизодов с $\mathrm{Delay}_k=0$ при детекции; $N_{\mathrm{late\,or\,missed}}$ — число эпизодов с опозданием или без детекции. Если нет ни одного эпизода с $\mathrm{Delay}_k>0$ (только своевременные детекции или нет истинных эпизодов), агрегаты задержки не определяются.

**Источники**:
1. Lavin A., Ahmad S. *Evaluating Real-Time Anomaly Detection Algorithms – The Numenta Anomaly Benchmark* // 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA). 2015. P. 38–44. DOI: [10.1109/ICMLA.2015.141](https://doi.org/10.1109/ICMLA.2015.141).
2. Scharwächter E., Müller E. *Statistical Evaluation of Anomaly Detectors for Sequences* // Proceedings of the 6th ACM SIGKDD International Workshop on Mining and Learning from Time Series (KDD MiLeTS 2020). 2020. arXiv: [2008.05788](https://arxiv.org/abs/2008.05788). DOI: [10.48550/arXiv.2008.05788](https://doi.org/10.48550/arXiv.2008.05788). 

# RMSE (заполненные траектории) #

После линейного заполнения координат в аномальных точках (см. Distance Loss) для каждого индекса $i$ считается отклонение $d_i$ между заполненными траекториями $G$ и $P$ в локальной ENU (метры). Как и у Detection Delay, в агрегат входят не все точки: нули в сумму не попадают.

$$ I=\{i : d_i > 0\}, \qquad RMSE=\sqrt{\frac{1}{|I|}\sum_{i\in I} d_i^2}. $$

Точки с $d_i = 0$ (совпадение заполненных координат) в $RMSE$ **не входят**, чтобы не занижать ошибку там, где траектории совпадают. Если $I=\varnothing$, $RMSE$ не определяется. $d_i$ и $RMSE$ — в **метрах**.

*Справочно (литература, в `GeospatialMetrics` не считается):* корень из среднего квадрата отклонения пропущенной аномалии от хорды штатного хода по $S_{FN}=\{i:y_i=0\land\hat y_i\neq 0\}$ и $V=\{i:\text{есть оба штатных соседа}\}$.

**Источники**:
1. Sturm J., Engelhard N., Endres F., Burgard W., Cremers D. *A Benchmark for the Evaluation of RGB-D SLAM Systems* // 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2012. P. 573–580. DOI: [10.1109/IROS.2012.6385773](https://doi.org/10.1109/IROS.2012.6385773).
2. Sørbø S., Ruocco M. *Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series* // Data Mining and Knowledge Discovery. 2024. Vol. 38. P. 1027–1068. DOI: [10.1007/s10618-023-00988-8](https://doi.org/10.1007/s10618-023-00988-8).

# Hausdorff Distance #

По пропускам $S_{FN}\cap V$ (см. справку в разделе RMSE):

$$ H_{FN}=\max_{i\in S_{FN}\cap V}d_i. $$

При пустом $S_{FN}$ значение равно $0$. Если $S_{FN}\cap V=\emptyset$ при непустом $S_{FN}$, $H_{FN}$ не определяется. В `GeospatialMetrics` вместо Hausdorff считается $\max_i d_i$ по всем индексам заполненных траекторий (в **метрах**).

Это направленный максимум ошибки пропуска. Симметричное расстояние множеств $H(A,B)=\max\{h(A,B),h(B,A)\}$, $h(A,B)=\max_{a\in A}\min_{b\in B}d(a,b)$, здесь не считается: оно не использует ни время, ни маску пропуска.

**Источники**:
1. Olesen K. V., Boubekki A., Kampffmeyer M. C., Jenssen R., Christensen A. N., Hørlück S., Clemmensen L. H. *A Contextually Supported Abnormality Detector for Maritime Trajectories* // Journal of Marine Science and Engineering. 2023. Vol. 11, No. 11. Article 2085. DOI: [10.3390/jmse11112085](https://doi.org/10.3390/jmse11112085).
2. Yu C., Jiang Z., Zhang X., He W., Zhong C. *A Novel Trajectory Repairing Model Based on the Artificial Potential Field-Enhanced A\* Algorithm for Small Coastal Vessels* // Journal of Marine Science and Engineering. 2025. Vol. 13, No. 7. Article 1200. DOI: [10.3390/jmse13071200](https://doi.org/10.3390/jmse13071200).
# Distance Loss #

У Yu et al. это модуль разности полных длин двух траекторий. Здесь длина — сумма $d_{\mathrm{geo}}$ по звеньям ломаной:

$$ L(T)=\sum d_{\mathrm{geo}}(p_i,p_{i+1}), \qquad DL=|L_{\mathrm{pred}}-L_{\mathrm{ref}}|, \qquad DLR=\frac{DL}{L_{\mathrm{ref}}}. $$

$L_{\mathrm{ref}}$ считается по координатам $G$: штатные точки и интерполяция внутренних разрывов. Краевой аномальный участок без пары штатных соседей в $L_{\mathrm{ref}}$ не входит. $L_{\mathrm{pred}}$ считается по координатам $P$ в точках с $\hat y_i\neq 0$; исключённые точки в ломаную не входят, соседние принятые соединяются напрямую.

Петля среди точек, принятых за штатные, увеличивает $L_{\mathrm{pred}}$ относительно $L_{\mathrm{ref}}$. Если штатных точек нет, $L_{\mathrm{ref}}=0$ и $DL=L_{\mathrm{pred}}$. При $L_{\mathrm{ref}}=0$ отношение $DLR$ не определяется. $DL$ и длины $L$ — в **метрах**.

**Источник**:

Yu C., Jiang Z., Zhang X., He W., Zhong C. *A Novel Trajectory Repairing Model Based on the Artificial Potential Field-Enhanced A\* Algorithm for Small Coastal Vessels* // Journal of Marine Science and Engineering. 2025. Vol. 13, No. 7. Article 1200. DOI: [10.3390/jmse13071200](https://doi.org/10.3390/jmse13071200).
