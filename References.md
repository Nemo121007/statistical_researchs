# Метрики #
1. Huet A., Navarro J. M., Rossi D. Local Evaluation of Time Series Anomaly Detection Algorithms // Proceedings of the 
28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22). 2022. P. 3873–3883. 
DOI: 10.1145/3534678.3539339.\
Работа вводит Affiliation Precision/Recall/F1, формализует способ их расчёта и исследует их применимость для 
point-based и range-based аномалий. Авторы также подробно рассматривают ограничения классических Precision/Recall 
и существующих event-based метрик.
DOI 10.1145/3534678.3539339
2. Tatbul N., Lee T. J., Zdonik S., Alam M., Gottschlich J. Precision and Recall for Time Series // 
Advances in Neural Information Processing Systems (NeurIPS). 2018. Vol. 31. P. 1920–1930.\
Работа по range-based Precision/Recall для временных рядов; возможно использование также как источник 
для обсуждения различий между point-wise и event-level оценкой.
3. Saito T., Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating 
Binary Classifiers on Imbalanced Datasets // PLOS ONE. 2015. Vol. 10, No. 3. e0118432. 
DOI: 10.1371/journal.pone.0118432.\
Работа обосновывает применение Precision–Recall curve и PR-AUC при сильном дисбалансе классов и 
показывает ограничения ROC-подхода в таких условиях.
DOI 10.1371/journal.pone.0118432 !!!
4. Powers D. M. W. Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation // 
Journal of Machine Learning Technologies. 2011. Vol. 2, No. 1. P. 37–63. DOI: 10.9735/2229-3981.2.1.\
Проверить
5. Lavin A., Ahmad S. Evaluating Real-Time Anomaly Detection Algorithms – 
The Numenta Anomaly Benchmark // 2015 IEEE 14th International Conference on Machine Learning and Applications 
(ICMLA). 2015. P. 38–44. DOI: 10.1109/ICMLA.2015.141.\
Основная работа по оценке времени реакции на аномалию в online/streaming detection. NAB учитывает момент обнаружения: 
раннее обнаружение получает больший положительный вклад, а задержка и false alarms штрафуются. Это один из основных 
источников для обоснования Detection Delay как отдельной характеристики.
DOI 10.1109/ICMLA.2015.141
6. Scharwächter E., Müller E. Statistical Evaluation of Anomaly Detectors for Sequences // Proceedings of the 6th 
ACM SIGKDD Workshop on Mining and Learning from Time Series (KDD MiLeTS 2020). 2020. DOI: 10.48550/arXiv.2008.05788.\
Работа рассматривает временную близость детекций, time-tolerant Precision/Recall и показывает, 
что временная толерантность может завышать оценку качества, если применять её без статистического контроля. 
Полезна для методологического обоснования того, почему Detection Delay лучше рассматривать отдельно, 
а не скрывать внутри модифицированного F1.
DOI / arXiv 2008.05788
7. Sturm J., Engelhard N., Endres F., Burgard W., Cremers D. A Benchmark for the Evaluation of RGB-D SLAM Systems 
// 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2012. P. 573–580. 
DOI: 10.1109/IROS.2012.6385773.\
Работа по оценке ошибок траекторий, вводящая Absolute Trajectory Error (ATE) и агрегирование абсолютной 
пространственной ошибки посредством RMSE относительно ground truth trajectory.
DOI 10.1109/IROS.2012.6385773
8. Olesen K. V., Boubekki A., Kampffmeyer M. C., Jenssen R., Christensen A. N., Hørlück S., 
Clemmensen L. H. A Contextually Supported Abnormality Detector for Maritime Trajectories // 
Journal of Marine Science and Engineering. 2023. Vol. 11, No. 11. Article 2085. DOI: 10.3390/jmse11112085.\
Авторы непосредственно рассматривают Hausdorff Distance как меру сходства/отклонения морских траекторий, 
приводят её математическое определение и обсуждают ограничения, в частности игнорирование временной компоненты.
9. Yu C., Jiang Z., Zhang X., He W., Zhong C. A Novel Trajectory Repairing Model Based on the Artificial Potential 
Field-Enhanced A* Algorithm for Small Coastal Vessels // Journal of Marine Science and Engineering. 2025. Vol. 13, 
No. 7. Article 1200. DOI: 10.3390/jmse13071200.\
Авторы оценивают восстановленные морские траектории одновременно посредством Hausdorff Distance, DTW и Distance Loss. 
Для Distance Loss дана формула как абсолютная разница суммарных длин двух траекторий:
$$ DL=\left|L_1-L_2\right|. $$
Таким образом, работа даёт прямое академическое обоснование использования Distance Loss при сравнении эталонной и 
искажённой судовой траектории. DOI 10.3390/jmse13071200
10. Schmidl S., Wenig P., Papenbrock T. Anomaly Detection in Time Series: A Comprehensive Evaluation // Proceedings 
of the VLDB Endowment. 2022. Vol. 15, No. 9. P. 1779–1797. DOI: 10.14778/3538598.3538602.\
Исследование оценки TSAD, в котором используются F1, PR-AUC и другие threshold-agnostic метрики и 
систематически сравниваются способы оценки большого числа алгоритмов на временных рядах.
11. Sørbø S., Ruocco M. Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in 
Time Series // Data Mining and Knowledge Discovery. 2024. Vol. 38. P. 1027–1068. DOI: 10.1007/s10618-023-00988-8.\
Обзорная работа, систематизирующая point-wise, range-based, time-aware и affiliation-based метрики и сопоставляющая 
их достоинства и ограничения. Для вашей методики она особенно полезна как единый источник, связывающий Point-wise F1, 
PR-AUC и Affiliation F1 в общей системе оценки TSAD.
DOI 10.1007/s10618-023-00988-8


# Алгоритмы #