# HAHA: Проект по детектированию угла расхождения строп при погрузке/разгрузке контейнеров

## Краткое описание

Программа позволяет по фотографии погрузки/разгрузки контейнера определить угол расхождения строп, а также примерное
положение контейнера относительно камеры.

## Технические требования

1. OS MacOS либо Linux
2. Процессор не хуже 5600X (много параллельных вычислений на процессоре, может занять много времени, чем больше ядер -
   тем лучше)
3. ОЗУ не меньше 8 гб (лучше хотя бы 16)
4. Python 3

## Подготовка к запуску

1. Установить Wolfram Script (Прим.: потребуется регистрация на Wolfram User Portal)
    1. По гайду https://support.wolfram.com/46072 установить Wolfram Kernel
        2. Установочный файл здесь https://www.wolfram.com/engine/
    2. Установить wolframscript https://www.wolfram.com/wolframscript/
    3. Проверить, что установка прошла корректно, введя в консоль комманду
   ```wolfram
   wolframscript -code 2+2 
   ```
   ```
   4
   ```
2. Установить зависимости python

```bash
pip install -r requirements.txt
```

## Краткий принцип работы

Алгоритм работает следующим образом:

1. Обученная на детектирование модель YOLOv8m находит контейнеры на изображении
2. Из всех найденных контейнеров выбирается ближайший к центру снимка
3. Вырезается изображение контейнера, продолженное до верхней границы исходного изображения
4. Обученная на нахождение ключевых точек модель YOLOv8m-pose находит скелет контейнера
5. По построенному скелету запускается алгоритм оптимизации, находящий конфигурацию груза и строп, при проектировании
   которой на матрицу камеры получается наилучшим образом приближенный скелет. Более подробно про алгоритм будет на
   защите.
6. По конфигурации вычисляется угол между противоположными стропами

## Приближения в которых работает алгоритм

1. На фотографии вместе с поднимаемым грузом **обязательно** дожен быть виден крюк, за который подвешен контейнер
2. Поднимаемый груз - стандартный контейнер 8x8.5x10 футов (малый). Размеры контейнера можно изменить.
3. Длины строп одинаковы (ака контейнер не перекошен)

## Запуск программы

1. Обновить права скрипта

```bash
chmod 777 script
```

2. Положить изображение в директорию проекта
3. Запустить программу и указать относительный путь до изображения
```bash
python main.py test.jpg
```
4. В результате появится файл result.jpg

## Примечание
1. Cкорость работы зависит от процессора, на моем компьютере это примерно 2 минуты
2. В процессе работы на этапе вычисления углов в консоль может писаться много мусора, это ни на что не влияет и может
   быть проигнорировано.
3. На текущем этапе модель нестабильна и может выдавать неправдоподобные результаты



## Участники

* Колесников Иван
* Полоник Иван
* Алтухова Анна
* Самигуллин Линар
* Гербер Маргарита