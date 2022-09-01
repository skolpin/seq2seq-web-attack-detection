# Seq2seq Web Attack Detection
Макет обнаружения аномальных HTTP-запросов к веб-серверу.

## Гипотеза

Признаки веб-атак (SQL Injection, XSS, CSRF и др.) могут быть выявлены через обнаружение аномалий в HTTP-запросах.

## Способ

Рекуррентный нейросетевой автокодировщик типа «Sequence-to-Sequence» с LSTM, обученный на множестве нормальных HTTP-запросов, детектирует аномалии. Автокодировщик обучается воспроизводить с малой ошибкой входной запрос, промежуточно переводя его в скрытое представление меньшей размерности. При подаче аномального запроса на вход обученного автокодировщика ожидается большая ошибка воспроизведения, которая и позволит детектировать аномалию. В результате анализа аномального образца детектор также выделяет наиболее аномальные участки запроса.

## Данные

Набор нормальных и аномальных HTTP-запросов тестового веб-приложения [__train.txt__](/train.txt), [__anomaly.txt__](/anomaly.txt).

## Описание макета

HTTP-запрос в макете представляется как последовательность символов. Для кодирования символов создается словарь уникальных соответствий «символ – числовой код». Затем каждый символ в запросе кодируется по принципу «one-hot encoding», то есть представляется в виде вектора, длина которого равна объему словаря символов, и все элементы которого равны нулю, за исключением одного элемента, равного единице, стоящего на позиции кода этого символа. Запрос представляется в виде последовательности векторов, кодирующих символы.

Для обучения автокодировщика обучающие данные представляются в виде массива с размерностью [количество запросов в обучающей выборке] x [длина самого длинного запроса в обучающей выборке] x [объем словаря символов]. Так как запросы имеют разную длину, они дополняются до длины самого длинного запроса при помощи специального символа `<PAD>`, который также вносится в словарь и кодируется как прочие символы. Также вносятся в словарь и кодируются спецсимволы `<UNK>` (для будущего кодирования символов, не вошедших в словарь), `<GO>` (для кодирования начала запроса) и `<EOR>` (для кодирования конца запроса).

Модель Sequence-to-Sequence состоит из двух рекуррентных LSTM сетей — кодера и декодера. Кодер отображает входную последовательность в вектор скрытых состояний фиксированной длины. Декодер разворачивает скрытое представление в целевой вектор. При обучении автокодировщик представляет собой модель, в которой целевые значения устанавливаются такими же, как входные значения. 

Идея состоит в том, чтобы обучить автокодировщик декодировать нормальные запросы, или, другими словами, приближать их тождественное отображение. Если обученному автокодировщику подать на вход аномальный запрос, он, вероятно, воссоздает его с высокой степенью ошибки, просто потому, что никогда его не видел.

Схема автокодировщика Sequence-to-Sequence:

![seq2seq](/slides/seq2seq.png)

LSTM-блоки в модели отображают входной запрос в вектор скрытых состояний длиной 256. На выходе декодера расположен Softmax-слой с количеством нейронов, равным объему словаря символов. Особенность функции активации Softmax заключается в том, что сумма выходных значений всех нейронов в слое равна единице, а выход каждого нейрона можно интерпретировать как вероятность или уверенность модели в данном символе, что показано на рисунке как P(∙).

В результате обработки одного запроса на выходе модели получаем вектор вероятностей для каждого символа. На основе этого вектора вычисляется мера ошибки декодирования данного запроса:

$error=1-(\frac{1}{len(request)}\sum_{char\in request}P(char))$

Мера ошибки оперирует усреднением вероятностей по всем символам в запросе. Мотивация введения такой меры ошибки состоит в том, что, если модель качественно декодировала запрос, средняя вероятность по всем символам будет близка к единице, а значит мера ошибки будет близка к нулю. И наоборот, при подаче модели аномального запроса, она будет давать низкую вероятность для тех участков символов, которые не встречались в обучающих запросах и ошибка будет высокой.

После обучения автокодировщика на множестве нормальных запросов определяется пороговое значение ошибки для обнаружения аномальных запросов. Для этого из множества нормальных запросов выделяются валидационные нормальные запросы, не участвующие в обучении. После обучения модели на вход подаются валидационные запросы, для них вычисляется ошибка и определяется порог

$threshold=\widehat{m}(e)+k\widehat{\sigma}(e); k=5...6$

где $\widehat{m}(e)$ и $\widehat{\sigma}(e)$ – оценки математического ожидания и дисперсии ошибки для валидационных нормальных запросов.

После вычисления порога на вход модели подаются аномальные запросы, для них также вычисляется ошибка, которая сравнивается с порогом. При превышении порога запрос признается аномальным. Затем вычисленные значения ошибок для валидационных и аномальных данных, а также порог используются для построения гистограммы и вывода отчета о качестве обнаружения аномалий. Схема работы макета обнаружения аномальных HTTP-запросов представлена на рисунке:

<img src="/slides/scheme.png" width="600"/>

Стоит отметить особенность подхода. Применение автокодировщика Sequence-to-Sequene позволяет не только выявлять аномальные запросы и отличать их от нормальных, но и, за счет вывода вероятности для каждого символа в запросе, дает возможность выделять наиболее аномальные участки запроса. То, что модель воспроизводит символы или последовательности символов с низкими вероятностями, может говорить о том, что модель не встречалась с такой информацией при обучении на нормальных запросах и в таких участках запроса могут содержаться признаки атаки. Для выделения аномальных символов вероятность символа сравнивается с пороговым значением 0.5. Схема обработки одного запроса и его вывода с выделением аномальных символов представлена на рисунке:

<img src="/slides/method.png" width="600"/>

## Методика тестирования

Для тестирования макета использовался набор нормальных и аномальных HTTP-запросов к тестовому веб-приложению. Набор включает в себя 21991 нормальный запрос и 1097 аномальных запросов, содержащих веб-атаки SQL injection, XSS, CSRF и другие. Набор данных представляет собой текстовые файлы [__train.txt__](/train.txt) (содержит нормальные запросы) и [__anomaly.txt__](/anomaly.txt) (содержит аномальные запросы).

Для целей валидации и тестирования макетом из нормальных запросов выделяется множество валидационных запросов, таким образом, чтобы объем валидационных и аномальных данных совпадал. В данном случае объем обучающей выборки составил 20894 запроса, тестовой – 2194 запроса (по 1097 каждого класса).

После запуска макета, загрузки данных и их разделения на обучающие, валидационные и аномальные, происходит обучение и тестирование модели согласно схеме работы макета. В результате работы в файл Histogram.png сохраняется гистограмма распределения ошибки на аномальных и нормальных запросах относительно порога, в файл test report.txt сохраняется отчет о качестве обнаружения аномалий, а на экран выводятся неверно проклассифицированные запросы, а также несколько верно обнаруженных аномальных запросов.

## Результаты

На риcунке представлена гистограмма распределений ошибки для нормальных и аномальных запросов относительно порога ошибки. 

<img src="/Histogram.png" width="800"/>

А так выглаядит отчет о тестировании макета, включающий в себя метрики качества классификации: Accuracy, Precision, Recall, F1-меру, а также матрицу ошибок.

```
Test report:
____________
Accuracy score  : 0.9990884229717412
Precision score : 1.0
Recall score    : 0.9981768459434822
F1 score        : 0.9990875912408759

Confusion matrix:
             pred:anomal  pred:normal
true:anomal         1095            2
true:normal            0         1097
```

Пример вывода макета при обработке аномального запроса с выделением аномальных участков. Видно, что в теле запроса содержится инъекция кода и тело запроса практически целиком выделено моделью как аномалия.

![example](/slides/example.png)

## Вывод 

Макет, созданный на основе предложенного способа, пригоден для обнаружения аномалий в HTTP-запросах при обнаружении веб-атак. Результаты тестирования показывают высокое качество обнаружения аномалий. К недостаткам способа можно отнести медленную обработку запросов нейронной сетью (при реализации на Python).

## Применение

Макет представляет из себя интерактивный блокнот Jupyter Noteboook [__seq2seq-web-attack-detection.ipynb__](/seq2seq-web-attack-detection.ipynb) с программной реализацией на Python 3.6.8.

Для работы с ним необходимо установить Jupyter:

` > pip install jupyter `

после установки необходимо запустить сам Jupyter

` > jupyter notebook `

и открыть файл блокнота seq2seq-web-attack-detection.ipynb

Зависимости от сторонних библиотек и их версий перечислены в requirements.txt.
Для установки библиотек необходимо воспользоваться pip:

` > pip install -r requirements.txt `

К макету также прилагаются следующие файлы:

* `utils.py` - библиотека собственных функций
    
* `train.txt`, `anomaly.txt` - файлы с обучающими (нормальными) и тестовыми (аномальными) HTTP-запросами к веб-серверу
  
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

## Референсы макета и источник данных

- https://github.com/skolpin/seq2seq-web-attack-detection/blob/main/slides/detecting_web_attacks_rnn.pdf
- https://habr.com/ru/company/pt/blog/439202/
