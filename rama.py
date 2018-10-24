# coding=utf-8
#!/usr/bin/python
'''
Created on 2018年10月23日
@author: lichaoxing
基于RAIMA模型时间序列预测，补齐平稳序列中缺失值（非平稳可在时间序列构造时采用平稳处理）
'''
import pandas as pd
import numpy as np
import urllib.request
from statsmodels.tsa.arima_model import ARMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings('ignore')
#从url下载数据
def get_data_from_url(url_str):
    result = urllib.request.urlopen(url_str)
    content = result.read().decode()
    return content
#绘制自相关与偏自相关图像
def plot_acf_pacf(timeseries, maxLag):
    f = plt.figure(facecolor='white', figsize = (12, 9))
    ax1 = f.add_subplot(211)
    plot_acf(timeseries, lags = maxLag, ax = ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(timeseries, lags = maxLag, ax = ax2)
    plt.show()
    return;
#基于BIC（残差）准测训练模型
def proper_model(timeseries, maxLag):
    init_bic = 1000000000
    model_return = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(timeseries, order=(p, q))
            try:
                results_ARMA = model.fit(disp = 0, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                model_return = results_ARMA
                init_bic = bic
    return model_return
#基于业务需求预测缺失值
def give_me_value(result_only_lost, result_list, time_list_seq, temp_list_seq, humi_list_seq, missed_value_num, det_time):
    if len(time_list_seq) < 100:
        time_list_seq = time_list_seq[: len(time_list_seq)]
        temp_list_seq = temp_list_seq[: len(temp_list_seq)]
        humi_list_seq = humi_list_seq[: len(humi_list_seq)]
    else:
        time_list_seq = time_list_seq[len(time_list_seq) - 100: len(time_list_seq)]
        temp_list_seq = temp_list_seq[len(temp_list_seq) - 100: len(temp_list_seq)]
        humi_list_seq = humi_list_seq[len(humi_list_seq) - 100: len(humi_list_seq)]  

    tidx = pd.DatetimeIndex(time_list_seq, freq = None)
    #构造时间序列
    dta_temp = pd.Series(temp_list_seq, index = tidx)
    dta_humi = pd.Series(humi_list_seq, index = tidx)
    model_temp = proper_model(dta_temp, 9)
    model_humi = proper_model(dta_humi, 9)
    result_temp = [float(-999.00) for _ in range(missed_value_num)]
    result_humi = [float(-999.00) for _ in range(missed_value_num)]
    if model_temp is not None:
        predict_temp = model_temp.forecast(missed_value_num)
        result_temp = predict_temp[0].tolist()
    if model_humi is not None:
        predict_humi = model_humi.forecast(missed_value_num)
        result_humi = predict_humi[0].tolist()
    for num_i in range(missed_value_num):
        time_list_seq.append(time_list_seq[-1] + det_time)
        temp_list_seq.append(float(result_temp[num_i]))
        humi_list_seq.append(float(result_humi[num_i]))
        
        result_list.append({"id": result_list[-1]["id"] + 1, "time": time_list_seq[-1], "temp": temp_list_seq[-1], "humi": humi_list_seq[-1]})
        result_only_lost.append({"time": time_list_seq[-1], "temp": temp_list_seq[-1], "humi": humi_list_seq[-1]})
    return
#预测温度与湿度
def predict_temp_humi(data_str, det_time):
    data_list = data_str.split("\n")
    del data_list[-1]
    if len(data_list) < 2:
        return [], ["内容为空"]
    result_list = []
    result_only_lost = []
    time_list_seq = []
    temp_list_seq = []
    humi_list_seq = []
    missed_value_num = 0
    total_missed_value_num = 0
    for i in range(1, len(data_list)):
        tmp_dict = {}
        data_list_split = data_list[i].split(",")
        #构建字典
        tmp_dict["id"] = int(data_list_split[0])
        tmp_dict["time"] = int(data_list_split[1])
        tmp_dict["temp"] = float(data_list_split[2])
        tmp_dict["humi"] = float(data_list_split[3])
        if i == 1:
            result_list.append(tmp_dict)
            #时间 温度 湿度  list
            time_list_seq.append(int(data_list_split[1]))
            temp_list_seq.append(float(data_list_split[2]))
            humi_list_seq.append(float(data_list_split[3]))
        else:
            #允许测试时间点在标准等差情况下有50（单位）的浮动
            diff_tim = int(data_list_split[1]) - int(data_list[i - 1].split(",")[1])
            if int(diff_tim % det_time) < 50 and int(diff_tim / det_time) > 1:
                missed_value_num = int(diff_tim / det_time) - 1
            elif int(diff_tim % det_time) > (det_time - 50) and int(diff_tim / det_time) > 0:
                missed_value_num = int(diff_tim / det_time)
            if missed_value_num:
                give_me_value(result_only_lost, result_list, time_list_seq, temp_list_seq, humi_list_seq, missed_value_num, det_time)
                total_missed_value_num += missed_value_num
                missed_value_num = 0
            tmp_dict["id"] = result_list[-1]["id"] + 1
            result_list.append(tmp_dict)
            time_list_seq.append(int(data_list_split[1]))
            temp_list_seq.append(float(data_list_split[2]))
            humi_list_seq.append(float(data_list_split[3]))

    return result_only_lost, result_list, total_missed_value_num
#测试方法
def predict(url, det_time):
    
    file = open("./testdata_1.csv", "r")
    content = file.read()
#     print(content)
#     content = get_data_from_url(url)
#     result_only_lost, result, total_missed_value_num = predict_temp_humi(content, det_time)
    try:
        result_only_lost, result, total_missed_value_num = predict_temp_humi(content, det_time)
    except:
        result_only_lost = []
        result = ["缺失值前训练数据不足"]
        total_missed_value_num = 0
    file.close()
    return result_only_lost, result, total_missed_value_num

if __name__ == "__main__":
#     url = "http://dataapi.coldwang.com/api/collector/history?access=855&collector_sn=14400000418&start=2018-09-17 14:46:47&end=2018-09-17 18:42:22"

    det_time = 300
    result_only_lost, result, total_missed_value_num = predict("", det_time)
    print("result_only_lost：", result_only_lost)
    print("result：", result)
    print("total_missed_value_num：%d" % total_missed_value_num)