#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

res_ok = 0
res_ng = 0

#
# 学習データ作成
#
train_data = []
train_result = []

f = open('train.csv')
reader = csv.reader(f)
tmp_data = []
prev_data = 0.0
cur_data = 0.0

for item in reader:
    cur_data = float(item[0])

    # 過去５日分のデータを元に学習データを作成する
    if 5 <= len(tmp_data):
        # 過去５日分のデータ
        train_data.append(tmp_data[:])

        # 翌日上がったか、下がったかのデータ
        if prev_data < cur_data:
            train_result.append(1)
        else:
            train_result.append(0)

        tmp_data.pop(0)

    # データの更新
    tmp_data.append(cur_data)
    prev_data = cur_data

#
# 学習
# 　
clf = tree.DecisionTreeClassifier()
#clf = RandomForestClassifier()
clf.fit(train_data, train_result)

#
# 予測
#
f = open('predict.csv')
reader = csv.reader(f)
tmp_data = []
prev_data = 0.0
cur_data = 0.0

for item in reader:
    cur_data = float(item[0])

    # 過去５日分のデータを元に翌日の予測を行う
    if 5 <= len(tmp_data):
        # 予測するためのデータ作成
        predict_data = np.array(tmp_data)
        predict_data = predict_data.reshape(1, -1)

        # 予測
        result = clf.predict(predict_data)

        # 実際に上がったのか下がったのか判定
        if prev_data < cur_data:
            res = 1
        else:
            res = 0

        # 予測の比較
        if result == res:
            res_ok = res_ok + 1
        else:
            res_ng = res_ng + 1

        tmp_data.pop(0)

    tmp_data.append(cur_data)
    prev_data = cur_data

print "予測結果"
print "正解 　: %d回" % res_ok
print "不正解 : %d回" % res_ng
