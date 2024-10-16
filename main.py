import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


### Шаг 1: Генерация данных
def generate_data():
    x = np.linspace(-np.pi, np.pi, num=100000)  # создаем массив значений от -π до π с шагом 2*π/100000
    y = np.cos(x)  # вычисляем значения cos(x) для каждого x
    return x[:, np.newaxis], y[:, np.newaxis]  # Расширяем размерности данных до (N, 1)


def main():
    ######################### Шаг 2: Разделение на обучающий и проверочный наборы #####################################

    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Метод train_test_split используется для разделения данных на обучающую и тестовую выборки.
    # Аргументы метода:
    # X и y — исходные данные и метки соответственно.
    # test_size=0.2 — доля данных, которая будет использоваться для тестирования (в данном случае 20%).
    # random_state=42 — фиксирует случайный начальный номер для воспроизводимости результата.

    ######################### Шаг 2.1: Построение графика функции #####################################################
    ### Построить график функции $y=\cos(x)$ для визуализации данных.

    plt.figure(figsize=(10, 6)) # Создание фигуры с указанными размерами (10x6)
    plt.title("Функция cos(x)") # Заголовок
    plt.xlabel('x') # метка оси
    plt.ylabel('y') # метка оси
    plt.grid(True) # добавление сетки
    plt.plot(X, y, color='b', linewidth=2) #; Рисует линию между точками X и y,используя синий цвет и толщину линии 2px.
    plt.savefig('График функции') # сохранение графиков

    ######################### Шаг 3: Составление минимально возможной нейронной сети ##################################

    model = Sequential([
        Dense(units=1),  # первый скрытый слой с одним нейроном
        Activation('tanh'),  # нелинейная функция активации tanh
    ])
    ######################### Шаг 4: Определение функции потерь #######################################################
    ### Задать функцию потерь для регрессии.

    loss_function = MeanSquaredError()

    ######################### Шаг 5: Определение метода оптимизации ###################################################

    learning_rate = 0.001 # устанавливает скорость обучения для оптимизатора.
    opt = Adam(learning_rate=learning_rate) # создает экземпляр оптимизатора Adam с заданной скоростью обучения.
    model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])
    # loss определяет, как будут вычисляться потери (ошибки) модели во время обучения.
    # metrics задаёт список метрик, которые будут отслеживаться во время обучения.
    # optimizer устанавливает метод оптимизации, который будет использоваться для минимизации функции потерь.

    ########################## Шаг 6: Цикл обучения ###################################################################
    ### Настроить цикл обучения на обучающем наборе данных и сохранить историю обучения.

    batch_size = 100 # размер мини-пакетов
    epochs = 100 # число эпох для обучения
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(X_test, y_test),
                        callbacks=[tf.keras.callbacks.CSVLogger('training_log.csv')], shuffle=True)

    # Метод fit выполняет цикл обучения модели на обучающих данных. Параметры метода:
    # X_train и y_train — обучающие данные и метки соответственно.
    # batch_size - batch_size
    # batch_size — число эпох
    # verbose - уровень детализации вывода
    # validation_data = (X_test, y_test) — данные для валидации
    # callbacks — список обратных вызовов для записи истории обучения в файл
    # shuffle - включить перемешивание данных перед каждым циклом обучения

    ########################### Шаг 7: Проверка модели ################################################################
    ### Запустить проверку на проверочном наборе данных и сохранить историю обучения.

    predictions = model.evaluate(X_test, y_test, batch_size=batch_size)
    print("Оценка модели: {}".format(predictions[0]))

    ########################### Шаг 8: Графическое представление результатов ##########################################
    ### Построить графики ошибок на обучающем и проверочном наборах данных.

    df = pd.read_csv('training_log.csv')
    plt.figure(figsize=(10, 6))
    plt.title("Ошибка (MSE) против эпохи")
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.grid(True)
    sns.lineplot(x="epoch", y="val_loss", data=df)
    plt.savefig('График результатов')


if __name__ == '__main__':
    main()
