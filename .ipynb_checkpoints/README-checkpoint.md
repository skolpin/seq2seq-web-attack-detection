# Seq2seq Web Attack Detection
Макет обнаружения аномальных HTTP-запросов к веб-серверу.

Макет представляет из себя интерактивный блокнот Jupyter Noteboook seq2seq-web-attack-detection.ipynb с программной реализацией на Python 3.6.8
Для работы с ним необходимо установить Jupyter:

> pip install jupyter

после установки необходимо запустить сам Jupyter

> jupyter notebook

и открыть в нем файл блокнота seq2seq-web-attack-detection.ipynb

Зависимости от сторонних библиотек и их версий перечислены в requirements.txt.
Для установки библиотек необходимо воспользоваться pip:

> pip install -r requirements.txt


К макету также прилагаются следующие файлы:

	- utils.py - библиотека собственных функций
    
	- train.txt, anomaly.txt - файлы с обучающими (нормальными) и тестовыми (аномальными) HTTP-запросами к 
        	веб-серверу
    
    
Референсы макета:
https://github.com/skolpin/seq2seq-web-attack-detection/detecting_web_attacks_rnn.pdf
https://habr.com/ru/company/pt/blog/439202/



Сам макет seq2seq-web-attack-detection.ipynb при работе загружает из файлов train.txt и anomaly.txt нормальные и аномальные запросы, разбивает нормальные 
запросы на обучающие и валидационные (при этом количество валидационных запросов равно количеству аномальных), затем переводит все данные в 
вид трехмерных One-hot encoding массивов, строит и обучает модель рекуррентного seq2seq-автокодировщика и сохраняет ее в папку 
model. Изначально в этой папке уже есть обученная на этих данных модель.

Затем макет позволяет загрузить обученную модель с диска и обработать с помощью нее валидационные (=нормальные) и тестовые (=аномальные)
запросы. По результатам обработки валидационных запросов вычисляется порог ошибки реконструкции запроса, превышение которого признается 
аномалией.

Затем макет позволяет построить гистограмму распределения значений ошибки для нормальных и аномальных запросов относительно порога (сохраняя 
ее в файл "Histogram.png"), вычислить метрики качества обнаружения аномальных запросов  (сохраняя отчет в файл "test report.txt") и вывести 
ложные (FP и FN) срабатывания, а также несколько TP-срабатываний (истинно аномальных обнаруженных запросов) с выделением аномальных участков 
запроса.

Есть возможность пропустить этап обучения, если это уже было сделано ранее, 
и сразу перейти к загрузке модели с диска и их обработке. (все ячейки до обучения все равно необходимо выполнить)