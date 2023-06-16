import pandas as pd
import numpy as np
from random import choice


def pre_filter_items(data, item_features, purchases_weeks=52, take_n_popular=5000):
    """Пред-фильтрация товаров
    Input
    -----
    data: pd.DataFrame
        Датафрейм с информацией о покупках
    item_features: pd.DataFrame
        Датафрейм с информацией о товарах
    """

    # Уберем товары с нулевыми продажами
    data = data[data['quantity'] != 0]

    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 4 месяца
    purchases_last_week = data.groupby('item_id')['week_no'].max().reset_index()
    weeks = purchases_last_week[
        purchases_last_week['week_no'] > data['week_no'].max() - purchases_weeks].item_id.tolist()
    data = data[data['item_id'].isin(weeks)]

    # Уберем не интересные для рекоммендаций категории (department)
    department_size = pd.DataFrame(item_features.groupby('department')['item_id'].nunique(). \
                                   sort_values(ascending=False)).reset_index()

    department_size.columns = ['department', 'n_items']

    rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
    items_in_rare_departments = item_features[
        item_features['department'].isin(rare_departments)].item_id.unique().tolist()

    data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] >= 0.7]

    # Уберем слишком дорогие товары
    data = data[data['price'] < 50]

    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер не покупил товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


# Генерация датафрейма с фичами юзеров
def get_users_features(user_data, train_data, users_embeddings):
    
    # Преобразовываем возраст - берем средний по группе
    fill_dict = {'65+': 70, '45-54': 50, '25-34': 30, '35-44': 40, '19-24': 22, '55-64': 60}
    user_data['age_desc_int'] = user_data['age_desc'].apply(lambda x: fill_dict[x])
    
    # Преобразовываем доход - берем средний по группе
    fill_dict = {'35-49K': 42, '50-74K': 61, '25-34K': 29, '75-99K': 87, 'Under 15K': 12,\
             '100-124K': 112, '15-24K': 20, '125-149K': 137, '150-174K': 162, '250K+': 270,\
             '175-199K': 187, '200-249K': 225}
    user_data['income_desc_int'] = user_data['income_desc'].apply(lambda x: fill_dict[x])
    
    # Преобразовываем размер дома
    fill_dict = {'2': 2, '3': 3, '4': 4, '1': 1, '5+': 5}
    user_data['household_size_desc_int'] = user_data['household_size_desc'].apply(lambda x: fill_dict[x])
    
    # Преобразовываем кол-во детей
    fill_dict = {'None/Unknown': 0, '1': 1, '2': 2, '3+': 3}
    user_data['kid_category_desc_int'] = user_data['kid_category_desc'].apply(lambda x: fill_dict[x])
    
    # Убираем ненужные столбцы
    user_data = user_data[['user_id', 'marital_status_code', 'homeowner_desc', 'hh_comp_desc', \
                              'age_desc_int', 'income_desc_int', 'household_size_desc_int', 'kid_category_desc_int']]
    
    # Считаем максимальный, минимальный(с весом!) номер дня в данных
    max_d, min_d = train_data['day'].max(), train_data['day'].min() // 2 # минимум с весом 0.5
    # число для заполнения Nan-ов 
    nan_d = max_d - min_d
    
    # кол-во дней с последней покупки
    days_from_last_purchase = train_data.groupby('user_id')['day'].max().reset_index()
    user_data = user_data.merge(days_from_last_purchase, on='user_id', how='left')
    user_data.fillna(nan_d, inplace=True)
    user_data['days_from_last_purchase'] = (max_d - user_data['day']).astype('int32')
    user_data.drop(columns='day', axis=1, inplace=True)
    
    # Средний чек
    av_check = train_data.groupby(['user_id', 'basket_id'])['sales_value'].sum().reset_index()
    av_check = av_check.groupby(['user_id'])['sales_value'].mean().reset_index()
    user_data = user_data.merge(av_check, on='user_id', how='left')
    user_data.rename(columns={'sales_value': 'av_check'}, inplace=True)
    
    # среднее кол-во товаров в заказе
    av_items_num = train_data.groupby(['user_id', 'basket_id'])['quantity'].sum().reset_index()
    av_items_num = av_items_num.groupby(['user_id'])['quantity'].mean().reset_index()
    user_data = user_data.merge(av_items_num, on='user_id', how='left')
    user_data.rename(columns={'quantity': 'av_items_num'}, inplace=True)
    
    # Эмбеддинги
    user_data = user_data.merge(users_embeddings, on='user_id', how='left')
    
    # Делаем фичи типа object категориальными
    cat_feats = ['marital_status_code', 'homeowner_desc', 'hh_comp_desc']
    user_data[cat_feats] = user_data[cat_feats].astype('category')
    
    return user_data

# Генерация датафрейма с фичами товаров
def get_items_features(item_data, train_data, items_embeddings):
    
    # Начальные фичи преобразуем в категориальные
    cat_feats = item_data.columns[1:]
    item_data[cat_feats] = item_data[cat_feats].astype('category')
    
    # Считаем максимальный, минимальный(с весом!) номер дня в данных
    max_d, min_d = train_data['day'].max(), train_data['day'].min() // 2 # минимум с весом 0.5
    # число для заполнения Nan-ов 
    nan_d = max_d - min_d
    
    # дней с последней продажи
    days = train_data[['item_id','day']].groupby('item_id').max().reset_index()
    days['days_from_last_purchase_item'] = max_d - days['day']
    item_data = item_data.merge(days, on='item_id', how='left')
    item_data['days_from_last_purchase_item'].fillna(nan_d, inplace=True)
    item_data.drop(columns='day', axis=1, inplace=True)
    
    # Среднее колво в корзине
    av_items_num = train_data.groupby(['item_id', 'basket_id'])['quantity'].sum().reset_index()
    av_items_num = av_items_num.groupby(['item_id'])['quantity'].mean().reset_index()
    item_data = item_data.merge(av_items_num, on='item_id', how='left')
    item_data.rename(columns={'quantity': 'av_item_num_per_basket'}, inplace=True)
    item_data['av_item_num_per_basket'].fillna(0, inplace=True)
    
    # Накопительная выручка по товарам
    item_value = train_data.groupby('item_id')['sales_value'].sum().reset_index()
    item_data = item_data.merge(item_value, on='item_id', how='left')
    item_data.rename(columns={'sales_value': 'item_value'}, inplace=True)
    item_data['item_value'].fillna(0, inplace=True)
    
    # Колво товаров в категории
    cat_num = item_data.groupby('sub_commodity_desc')['item_id'].count().reset_index()
    item_data = item_data.merge(cat_num, on='sub_commodity_desc', how='left')
    item_data.rename(columns={'item_id_y': 'cat_num', 'item_id_x': 'item_id'}, inplace=True)
    item_data['cat_num'].fillna(0, inplace=True)

    # Эмбеддинги
    item_data = item_data.merge(items_embeddings, on='item_id', how='left')
    
    return item_data

# Функция возвращает для юзера 5 рекомендованных товаров
# df - датафрейм с рекомендациями, dt - с товарами(вынимать категории), k - кол-во товаров в итоговом списке
# top_valued_items - список id товаров, дороже $7, отсортированный по убыванию цены - рассчитывается заранее для оптимизации
# top_items - список популярных товаров - получаем от рекоммендера 1 уровня
# справедливости ради  код надо рефакторить, чтобы избавиться от констант в условиях и циклах!

def get_recommendation_5(user, df, dt, top_popular_items, top_valued_items, k=5):
    
    actual_list = list(df[df['user_id'] == user]['actual'].tolist()[0])
    model_rec_list = df[df['user_id'] == user]['recomendations'].tolist()[0]
    res = [] # Возвращаемый список рекомендаций - 5шт
    
    # Мн-во ранее купленных товаров и мн-во категорий, из который уже взяты рекомендации
    filtered_items = actual_list.copy()
    filtered_items.append(999999) # Уберем непопулрные товары - это если была предфильтрация
    filtered_categories = [] # для контроля за уникальностью категорий товаров
    
    # ищем первый товар из рекомендованных стоимостью >$7
    flag = False
    for i in model_rec_list:
        if i in top_valued_items:
            item = i
            flag = True
            break

    # Если не нашли - добавляем дорогой из популярных.
    if not flag:
        for i in top_popular_items:
            if i in top_valued_items:
                item = i
                flag = True
            break
    if not flag: # если ничего не найдено, то берем случайный дорогой товар
        item = choice(top_valued_items)
        #item = top_valued_items[-1]
        flag = True
        
    if flag: # добавляем найденный товар в список рекмомендаций
        res.append(item)
        filtered_items.append(item)
        cat = dt[dt['item_id'] == item]['sub_commodity_desc'].tolist()[0]
        filtered_categories.append(cat)
    assert (len(res) == 1), 'Нет дорогого товара!' 

    # Добавляем 2 новых товара
    for i, item in enumerate(model_rec_list):
        cat = dt[dt['item_id'] == item]['sub_commodity_desc'].tolist()[0] # получаем категорию товара
        if not ((item in filtered_items) or (cat in filtered_categories)): # проверяем что товар не нужно отфильтровать
            res.append(item)
            filtered_items.append(item)
            filtered_categories.append(cat)
            if len(res) > 2:
                break
    if len(res) < 3: # Если не нашли нужное в рекомендованных, то ищем в top
        for i, item in enumerate(top_popular_items):
            cat = dt[dt['item_id'] == item]['sub_commodity_desc'].tolist()[0] # получаем категорию товара
            if not ((item in filtered_items) or (cat in filtered_categories)): # проверяем что товар не нужно отфильтровать
                res.append(item)
                filtered_items.append(item)
                filtered_categories.append(cat)
                if len(res) > 2:
                    break
    assert (len(res) == 3), 'Нет новых товаров!' 
        
     
    # добавляем оставашиеся 2 товара
    for i, item in enumerate(model_rec_list):
        cat = dt[dt['item_id'] == item]['sub_commodity_desc'].tolist()[0] # получаем категорию товара
        if not (cat in filtered_categories): # проверяем что товар не нужно отфильтровать
            res.append(item)
            filtered_items.append(item)
            filtered_categories.append(cat)
            if len(res) > 4:
                break
    if len(res) < 5: # Если не нашли нужное в own_recommended, то ищем в top
        for i, item in enumerate(top_popular_items):
            cat = dt[dt['item_id'] == item]['sub_commodity_desc'].tolist()[0] # получаем категорию товара
            if not (cat in filtered_categories): # проверяем что товар не нужно отфильтровать
                res.append(item)
                filtered_items.append(item)
                filtered_categories.append(cat)
                if len(res) > 4:
                    break

    assert len(res) == 5, 'Слишком короткий результат!'

    return res, filtered_categories

    
