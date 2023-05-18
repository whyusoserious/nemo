import pandas as pd


def make_df(filepath):
    # чтение данных из файла
    with open(f'{filepath}.txt', 'r') as f:
        lines = f.readlines()

    # преобразование строковых записей в объект datetime и числовые значения
    dates = []
    values = []
    for line in lines:
        date_str, value_str = line.strip().split(';')
        date = pd.to_datetime(date_str)
        dates.append(date)
        values.append(float(value_str))

    # создание DataFrame с индексом на основе объектов datetime
    data = pd.DataFrame({'value': values}, index=dates)

    # создание временного ряда на основе столбца 'value' в DataFrame
    time_series = pd.Series(data['value'])

    data.reset_index(level=0, inplace=True)
    data.rename(columns={'index': 'date'}, inplace=True)

    return data
