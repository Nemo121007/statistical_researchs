#  Обозначения #
Для формул обозначим:
$$ P=\{(t_i,\varphi_i,\lambda_i,\hat y_i)\}_{i=1}^{N} $$
— prediction, а
$$ G=\{(t_i,\varphi_i,\lambda_i,y_i)\}_{i=1}^{N} $$
— ground truth.\
Здесь \(y_i=0\) означает аномалию, \(y_i>0\) — штатное состояние. Для бинарных метрик:
$$ a_i=\mathbf1[y_i=0],\quad \hat a_i=\mathbf1[\hat y_i=0]. $$

# Point-wise F1 #
Классическая точечная метрика классификации, характеризующая баланс между полнотой обнаружения аномальных точек и 
точностью предсказаний. Используется как основная метрика непосредственного качества бинарной классификации каждой 
точки.

Для бинарной классификации:
$$ Precision=\frac{TP}{TP+FP}, $$ $$ Recall=\frac{TP}{TP+FN}, $$ $$ F_1= 2\frac{Precision\cdot Recall} {Precision+Recall} = \frac{2TP}{2TP+FP+FN}. $$
В частном случае:\
Аномалией считается точка с меткой \(0\):
$$ TP=\sum_i \mathbf1[a_i=1\land\hat a_i=1], $$ $$ FP=\sum_i \mathbf1[a_i=0\land\hat a_i=1], $$ $$ FN=\sum_i \mathbf1[a_i=1\land\hat a_i=0]. $$
После этого F1 рассчитывается стандартным образом.

Местоположение и время точек непосредственно в расчёте не участвуют.

Источники
1. Powers D. M. W. Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation // Journal of Machine Learning Technologies. 2011. Vol. 2, No. 1. P. 37–63. DOI: 10.9735/2229-3981.2.1.\ — формализация Precision, Recall и F-score.\
2. Schmidl S., Wenig P., Papenbrock T. Anomaly Detection in Time Series: A Comprehensive Evaluation // Proceedings of the VLDB Endowment. 2022. Vol. 15, No. 9. P. 1779–1797. DOI: 10.14778/3538598.3538602. — применение F1 в систематической оценке TSAD.\
3. Sørbø S., Ruocco M. Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series // Data Mining and Knowledge Discovery. 2024. Vol. 38. P. 1027–1068. DOI: 10.1007/s10618-023-00988-8. — систематический анализ point-wise метрик TSAD.

# PR-AUC #
Threshold-independent метрика качества anomaly score, оценивающая способность модели ранжировать аномальные точки выше 
штатных при всех возможных порогах. Особенно информативна при сильном дисбалансе классов.\
Для каждого порога \(\tau\) вычисляются:
$$ Precision(\tau) = \frac{TP(\tau)} {TP(\tau)+FP(\tau)}, $$ $$ Recall(\tau) = \frac{TP(\tau)} {TP(\tau)+FN(\tau)}. $$
PR-кривая задаётся функцией:
$$ Precision=f(Recall). $$
Площадь под ней:
$$ PR\text{-}AUC = \int_0^1 Precision(r)\,dr. $$
На практике интеграл рассчитывается численно.

Используемого бинарного prediction недостаточно для расчёта PR-AUC. Нужен непрерывный anomaly score:
$$ s_i\in\mathbb R, $$
где большее значение означает большую вероятность/степень аномальности.
Тогда:
$$ a_i=\mathbf1[y_i=0], $$
а пороговая классификация:
$$ \hat a_i(\tau)=\mathbf1[s_i\ge\tau]. $$
PR-AUC рассчитывается по всем $$\tau $$
Следовательно, для этой метрики фактический prediction должен содержать либо anomaly score, либо вероятностную оценку аномальности. Если модель выдаёт только конечную маску 0 / normal, PR-AUC вычислить невозможно.

Источники
1. Saito T., Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets // PLOS ONE. 2015. Vol. 10, No. 3. e0118432. DOI: 10.1371/journal.pone.0118432. — подробно исследуют Precision–Recall и PR-кривые при дисбалансе классов.
2. Schmidl S., Wenig P., Papenbrock T. Anomaly Detection in Time Series: A Comprehensive Evaluation // Proceedings of the VLDB Endowment. 2022. Vol. 15, No. 9. P. 1779–1797. DOI: 10.14778/3538598.3538602. — систематическая оценка TSAD с использованием AUC-PR.
3. Sørbø S., Ruocco M. Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series // Data Mining and Knowledge Discovery. 2024. Vol. 38. P. 1027–1068. DOI: 10.1007/s10618-023-00988-8. — классификация и анализ threshold-independent метрик TSAD.

# Affiliation F1 #
Метрика event-level оценки, предназначенная для сравнения истинных и предсказанных аномальных интервалов. В отличие от 
point-wise F1, учитывает временную близость предсказанного и истинного события, поэтому частичное смещение детекции не приравнивается автоматически к полностью независимой ошибке.\
Важно: здесь «расстояние» является временным, а не географическим.

Пусть:
$$ R=\{R_1,\ldots,R_m\} $$
— истинные anomaly events, а
$$ P=\{P_1,\ldots,P_k\} $$
— предсказанные events.\
Для каждого истинного события определяется локальная affiliation precision/recall на основе расстояний между точками/интервалами предсказания и истинного события.\
В общем виде:
$$ AffiliationPrecision = \frac{1}{m} \sum_{j=1}^{m}p_j, $$ $$ AffiliationRecall = \frac{1}{m} \sum_{j=1}^{m}r_j, $$
после чего:
$$ AffiliationF_1= 2\frac{ AffiliationPrecision\cdot AffiliationRecall }{ AffiliationPrecision+AffiliationRecall }. $$
Точная formulation \(p_j,r_j\) определяется через локальные временные расстояния между prediction и ground truth; преимуществом подхода является отсутствие необходимости вручную задавать набор эвристических параметров.

Используемая реализация:\
Из последовательностей выделяются непрерывные интервалы:
$$ GT_{anom} = \{i:y_i=0\}, $$ $$ Pred_{anom} = \{i:\hat y_i=0\}. $$
Поскольку \(t_i\) в prediction и ground truth одинаковы, метрика работает только с временными индексами/временными координатами этих двух множеств.\
Координаты \((\varphi_i,\lambda_i)\) в расчёте не используются.

Источники
1. Huet A., Navarro J. M., Rossi D. Local Evaluation of Time Series Anomaly Detection Algorithms // Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22). 2022. P. 3873–3883. DOI: 10.1145/3534678.3539339. — исходная работа, вводящая Affiliation Precision/Recall.
2. Sørbø S., Ruocco M. Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series // Data Mining and Knowledge Discovery. 2024. Vol. 38. P. 1027–1068. DOI: 10.1007/s10618-023-00988-8. — подробно рассматривают Affiliation F-score и его свойства.

# Detection Delay #
Detection Delay / Time-to-Detection\
Временная задержка между фактическим началом аномального эпизода и первой его детекцией моделью. Метрика характеризует скорость реакции, а не полноту или точность классификации.

Для \(k\)-го anomaly event:
$$ Delay_k= t_k^{detect}-t_k^{start}, $$
где \(t_k^{start}\) — начало истинного события, а \(t_k^{detect}\) — момент первой корректной детекции.\
Для множества событий можно использовать:
$$ MedianDelay, \quad MeanDelay, \quad P95Delay. $$
Для online-систем медианное и высокие квантили обычно информативнее одного среднего значения.\
Используемая реализация:\
Пусть \(S_k\) — непрерывный интервал индексов, для которого:
$$ y_i=0. $$
Начало события:
$$ t_k^{start}=\min_{i\in S_k}t_i. $$
Первой детекцией считается:
$$ t_k^{detect} = \min \{t_i: i\ge \min S_k,\ \hat y_i=0\}. $$
Тогда:
$$ Delay_k=t_k^{detect}-t_k^{start}. $$
Если до конца события предсказания аномалии не произошло, событие считается undetected и не должно молча преобразовываться в нулевую задержку.

Источники
1. Lavin A., Ahmad S. Evaluating Real-Time Anomaly Detection Algorithms – The Numenta Anomaly Benchmark // 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA). 2015. P. 38–44. DOI: 10.1109/ICMLA.2015.141. — NAB вводит scoring, учитывающий момент обнаружения и предпочтительность ранней реакции в online anomaly detection.
2. Scharwächter E., Müller E. Statistical Evaluation of Anomaly Detectors for Sequences // Proceedings of the 6th ACM SIGKDD Workshop on Mining and Learning from Time Series (KDD MiLeTS 2020). 2020. DOI: 10.48550/arXiv.2008.05788. — рассматривают временную толерантность и временную близость детекций в последовательностях.

# Geodesic Distance RMSE #
Среднеквадратичная пространственная ошибка между фактической координатой точки и её эталонной координатой. За счёт 
квадратичного штрафа значительно сильнее реагирует на крупные отклонения, чем средняя абсолютная ошибка.

Пусть \(P_i\) — оценённая позиция, а \(P_i^*\) — эталонная:
$$ d_i=d(P_i,P_i^*). $$
Тогда:
$$ RMSE_d = \sqrt{ \frac{1}{N} \sum_{i=1}^{N}d_i^2 } $$
где \(d(\cdot,\cdot)\) — пространственное расстояние.\
Для географических координат целесообразно использовать геодезическое расстояние на поверхности Земли.\

Используемая реализация:\
Для каждого аномального участка ground truth находятся соседние штатные точки:
$$ P_l=(t_l,\varphi_l,\lambda_l), \qquad P_r=(t_r,\varphi_r,\lambda_r), $$
где \(P_l\) и \(P_r\) — ближайшие слева и справа точки с \(y>0\).\
Для каждого времени \(t_i\) внутри разрыва строится reference position:
$$ \alpha_i= \frac{t_i-t_l}{t_r-t_l}. $$
При линейной интерполяции:
$$ \varphi_i^*= \varphi_l+ \alpha_i(\varphi_r-\varphi_l), $$ $$ \lambda_i^*= \lambda_l+ \alpha_i(\lambda_r-\lambda_l). $$
После этого:
$$ d_i= d_{\rm geo} \left( (\varphi_i,\lambda_i), (\varphi_i^*,\lambda_i^*) \right). $$
В расчёт spatial distortion должны попадать прежде всего false negative points:
$$ S_{FN} = \{i:y_i=0\land\hat y_i\neq0\}. $$
И:
$$ RMSE_d= \sqrt{ \frac{1}{|S_{FN}|} \sum_{i\in S_{FN}}d_i^2 } $$
Таким образом, модель получает большой штраф именно за те аномальные координаты, которые она ошибочно оставила штатными.

Источники
1. Sturm J., Engelhard N., Endres F., Burgard W., Cremers D. A Benchmark for the Evaluation of RGB-D SLAM Systems // 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2012. P. 573–580. DOI: 10.1109/IROS.2012.6385773. — используют RMSE абсолютной trajectory error относительно ground-truth trajectory.
2. Sørbø S., Ruocco M. Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series // Data Mining and Knowledge Discovery. 2024. Vol. 38. P. 1027–1068. DOI: 10.1007/s10618-023-00988-8. — как общий источник по структуре оценки TSAD; непосредственно географический RMSE не является их основной метрикой.

# Hausdorff Distance #
Метрика максимального пространственного расхождения двух геометрических объектов. В контексте траекторий показывает, насколько далеко друг от друга находятся наиболее удалённые части фактической и эталонной траекторий.

Для множеств точек \(A\) и \(B\):
$$ h(A,B) = \max_{a\in A}\min_{b\in B}d(a,b), $$
а симметричное расстояние Хаусдорфа:
$$ H(A,B)= \max \{h(A,B),h(B,A)\} $$
Метрика чувствительна к наиболее удалённому отклонению.

В текущей задаче времени и пространственные точки уже синхронизированы, наиболее естественна направленная 
временно-сопоставленная форма:
$$ H_{obs\rightarrow ref} = \max_{i\in S_{FN}} d_{\rm geo}(P_i,P_i^*) $$
То есть фактически определяется наиболее сильное пространственное отклонение среди ошибочно пропущенных 
аномальных точек.\
Если требуется именно классическая Hausdorff Distance, можно рассматривать множества prediction/reference траекторий и 
рассчитывать симметричное \(H(A,B)\). Однако для вашего benchmark directed-вариант лучше использует имеющуюся временную привязку.

Источники
1. Olesen K. V., Boubekki A., Kampffmeyer M. C., Jenssen R., Christensen A. N., Hørlück S., Clemmensen L. H. A Contextually Supported Abnormality Detector for Maritime Trajectories // Journal of Marine Science and Engineering. 2023. Vol. 11, No. 11. Article 2085. DOI: 10.3390/jmse11112085. — непосредственно рассматривают и используют Hausdorff Distance для maritime trajectories, включая обсуждение его ограничений.
2. Yu C., Jiang Z., Zhang X., He W., Zhong C. A Novel Trajectory Repairing Model Based on the Artificial Potential Field-Enhanced A* Algorithm for Small Coastal Vessels // Journal of Marine Science and Engineering. 2025. Vol. 13, No. 7. Article 1200. DOI: 10.3390/jmse13071200. — используют Hausdorff Distance для оценки качества восстановления судовой траектории вместе с DTW и Distance Loss.

# Distance Loss #
Разность полной длины сравниваемой и эталонной траекторий. Метрика характеризует изменение пройденного расстояния вследствие пространственного искажения траектории и особенно чувствительна к лишним петлям, резким отклонениям и другим изменениям геометрии пути.

Длина траектории:
$$ L(P)= \sum_{i=1}^{N-1} d(P_i,P_{i+1}). $$
Distance Loss:
$$ DL= |L(P)-L(P^*)| $$
где \(P\) — оцениваемая траектория, \(P^*\) — эталонная.\
Для сравнения траекторий различного масштаба удобно использовать нормированную форму:
$$ DLR= \frac{|L(P)-L(P^*)|} {L(P^*)}. $$

Эталонная длина рассчитывается по исходным штатным точкам и их интерполяции:
$$ L_{ref} = \sum_i d_{\rm geo}(P_i^*,P_{i+1}^*). $$
Фактическая длина trajectory, принятой классификатором как штатная:
$$ L_{pred} = \sum_{i} d_{\rm geo}(P_i,P_{i+1}), $$
где учитываются точки:
$$ \hat y_i\neq0. $$
Тогда:
$$ DL= |L_{pred}-L_{ref}| $$
или для сравнения различных рейсов:
$$ DLR= \frac{|L_{pred}-L_{ref}|} {L_{ref}}$$
При этом DL особенно хорошо выявляет ситуацию, когда ошибочно принятые за штатные точки образуют большую петлю:
$$ A\rightarrow B\rightarrow C\rightarrow B\rightarrow D, $$
поскольку длина фактической траектории становится значительно больше эталонной.

Источники
1. Yu C., Jiang Z., Zhang X., He W., Zhong C. A Novel Trajectory Repairing Model Based on the Artificial Potential Field-Enhanced A* Algorithm for Small Coastal Vessels // Journal of Marine Science and Engineering. 2025. Vol. 13, No. 7. Article 1200. DOI: 10.3390/jmse13071200. — непосредственно используют Distance Loss для оценки качества восстановления судовых траекторий; метрика рассматривается совместно с Hausdorff Distance и DTW.
