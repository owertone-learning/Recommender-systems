import pandas as pd
import numpy as np

class MainRecommender:
    """Рекомендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True, fake_id=999999):
        self.weighting = weighting
        self.fake_id = fake_id

        # Топ покупок каждого пользователя
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != fake_id]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != fake_id]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        # Подготовка user-item матриц
        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self._prepare_dicts(
            self.user_item_matrix)
        self.user_item_matrix_for_pred = self.user_item_matrix
        if weighting == 'bm25':
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T.tocsr()
        elif weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T.tocsr()

        # Обучение моделей
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data):
        """Готовит user-item матрицу"""

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)

        model.fit(csr_matrix(user_item_matrix).tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новый user, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""

        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[0][1]  # Берем 2-ой (не товар)
        return self.id_to_itemid[top_rec] # возвращаем индексы

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            top_popular = [rec for rec in self.overall_top_purchases[:N] if rec not in recommendations]
            recommendations.extend(top_popular)
            recommendations = recommendations[:N]

        return recommendations

    def get_recommendations(self, user, model, N=5):
        """Рекомендации через стандартные библиотеки implicit"""

        sparse_user_item = csr_matrix(self.user_item_matrix_for_pred).tocsr()
        self._update_dict(user_id=user)
        filter_items = [] if self.fake_id is not None else [self.itemid_to_id[self.fake_id]]
        res = model.recommend(userid=self.userid_to_id[user],
                              user_items=sparse_user_item[self.userid_to_id[user]],
                              N=N,
                              filter_already_liked_items=False,
                              filter_items=filter_items,
                              recalculate_user=True)
        mask = res[1].argsort()[::-1]
        res = [self.id_to_itemid[rec] for rec in res[0][mask]]
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стандартные библиотеки implicit"""

        self._update_dict(user_id=user)
        filter_items = [] if self.fake_id is not None else [self.itemid_to_id[self.fake_id]]
        res = model.recommend(userid=self.userid_to_id[user],
                              user_items=csr_matrix(self.user_item_matrix_for_pred).tocsr(),
                              N=N,
                              filter_already_liked_items=False,
                              filter_items=filter_items,
                              recalculate_user=True)
        mask = res[1].argsort()[::-1]
        res = [self.id_to_itemid[rec] for rec in res[0][mask]]
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, n=5):
        """Рекомендации через стандартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self.get_recommendations(user, model=self.model, N=n)

    def get_own_recommendations(self, user, n=5):
        """Рекомендуем товары среди тех, которые пользвотель уже купил (ItemItemRecommender)"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=n)

    def get_similar_users_recommendation(self, user, n=5):
        """Рекомендуем топ-N товаров, среди купленных похожими пользователями"""

        res = []
        # Находим похожих юзеров
        similar_users = self.model.similar_users(self.userid_to_id[user], N=n+1)
        similar_users = [rec for rec in similar_users[0]][1:]
        # Находим купленные топ товары, купленные этими юзерами
        for user in similar_users:
            # user_rec = self._get_recommendations(user, model=self.own_recommender, N=1)
            user_rec = self._get_recommendations(user, model=self.own_recommender, N=n)
            # выберем одну рекомендацию из списка рекомендаций юзера
            for r in user_rec:
                if r not in res:
                    res.append(r)
                    break
        res = np.array(res).flatten()

        assert len(res) == n, 'Количество рекомендаций != {}'.format(n)
        return res

    def get_similar_items_recommendation(self, user, n=5):
        """Рекомендуем товары, похожие на топ-N купленных пользователем товаров"""

        # выясняем топ-N купленных юзером товаров
        top_users_purchases = self.user_item_matrix.iloc[self.userid_to_id[user]].sort_values(ascending=False).head(
            n).reset_index()
        # находим по одной рекомендации для каждого топ-N товара.
        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = np.array(res)

        assert len(res) == n, 'Количество рекомендаций != {}'.format(n)
        return res
