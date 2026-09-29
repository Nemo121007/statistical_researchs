# Метрики #
1. Huet A., Navarro J. M., Rossi D. Local Evaluation of Time Series Anomaly Detection Algorithms // Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22). 2022. P. 3873–3883. DOI: 10.1145/3534678.3539339.\
Работа вводит Affiliation Precision/Recall/F1, формализует способ их расчёта и исследует их применимость для point-based и range-based аномалий. Авторы также подробно рассматривают ограничения классических Precision/Recall и существующих event-based метрик.
DOI 10.1145/3534678.3539339
2. Tatbul N., Lee T. J., Zdonik S., Alam M., Gottschlich J. Precision and Recall for Time Series // Advances in Neural Information Processing Systems (NeurIPS). 2018. Vol. 31. P. 1920–1930.\
Классическая работа по range-based Precision/Recall для временных рядов; возможно использование также как источник для обсуждения различий между point-wise и event-level оценкой.
NeurIPS / полный текст
3. Saito T., Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets // PLOS ONE. 2015. Vol. 10, No. 3. e0118432. DOI: 10.1371/journal.pone.0118432.\
Работа обосновывает применение Precision–Recall curve и PR-AUC при сильном дисбалансе классов и показывает ограничения ROC-подхода в таких условиях.
DOI 10.1371/journal.pone.0118432

6. Scharwächter E., Müller E. Statistical Evaluation of Anomaly Detectors for Sequences // Proceedings of the 6th ACM SIGKDD Workshop on Mining and Learning from Time Series (KDD MiLeTS 2020). 2020. DOI: 10.48550/arXiv.2008.05788.\
Работа рассматривает временную близость детекций, time-tolerant Precision/Recall и показывает, что временная толерантность может завышать оценку качества, если применять её без статистического контроля. Полезна для методологического обоснования того, почему Detection Delay лучше рассматривать отдельно, а не скрывать внутри модифицированного F1.
DOI / arXiv 2008.05788


# Алгоритмы #