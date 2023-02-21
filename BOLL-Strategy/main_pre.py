#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-02-22 4:11
# @Author : lichangheng

#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-02-21 15:52
# @Author : lichangheng

# coding=utf-8
from __future__ import print_function, absolute_import

import copy

import pandas as pd
from gm.api import *
from transformer import *
import talib as ta
import json
import numpy as np
import sys


# 策略中必须有init方法
def init(context):
    # 设置布林线的三个参数
    context.maPeriod = 26  # 计算BOLL布林线中轨的参数
    context.stdPeriod = 26  # 计算BOLL 标准差的参数
    context.stdRange = 1  # 计算BOLL 上下轨和中轨距离的参数
    # 设置要进行回测的合约
    context.symbol = 'SHSE.600004'  # 订阅&交易标的, 此处订阅的是600004
    context.period = max(219,context.maPeriod, context.stdPeriod, context.stdRange) + 1  # 订阅数据滑窗长度
    # 订阅行情
    subscribe(symbols=context.symbol, frequency='1d', count=context.period)


def on_bar(context, bars):  # 策略改进-Transformer时序
    # 获取数据滑窗，只要在init里面有订阅，在这里就可以取的到，返回值是pandas.DataFrame
    # 取到长度为220的历史数据
    data_his = context.data(symbol=context.symbol, frequency='1d', count=220, fields='close')
    # 获取长度为27的data数据
    data = data_his[-27:]
    # 历史股价序列传给Transformer模型预测未来3天的股价
    p1,p2,p3 = get_predict(np.array(data_his))
    p = pd.DataFrame([p1,p2,p3])
    # p和data拼接得到预测序列
    data_pre = pd.concat([copy.deepcopy(data),p],ignore_index=True)

    # 计算boll的上下界
    bollAvg = data['close'].rolling(context.maPeriod).mean()
    bollUpper = data['close'].rolling(context.maPeriod).mean() \
                + context.stdRange * data['close'].rolling(context.stdPeriod).std()
    bollBottom = data['close'].rolling(context.maPeriod).mean() \
                 - context.stdRange * data['close'].rolling(context.stdPeriod).std()

    # 计算预测序列的boll上下界
    bollAvg_pre = data_pre['close'].rolling(context.maPeriod).mean()
    bollUpper_pre = data_pre['close'].rolling(context.maPeriod).mean() \
                + context.stdRange * data_pre['close'].rolling(context.stdPeriod).std()
    bollBottom_pre = data_pre['close'].rolling(context.maPeriod).mean() \
                 - context.stdRange * data_pre['close'].rolling(context.stdPeriod).std()

    # 获取现有持仓
    pos = context.account().position(symbol=context.symbol, side=PositionSide_Long)

    # 交易逻辑与下单
    # 当有持仓，且股价穿过BOLL上界的时候卖出股票。
    if data.close.values[-1] > bollUpper.values[-1] and data.close.values[-2] < bollUpper.values[-2]:
        if pos:  # 有持仓就市价卖出股票。
            order_percent(symbol=context.symbol, percent=0.3, side=OrderSide_Sell,
                          order_type=OrderType_Market, position_effect=PositionEffect_Close)
            print('以市价单卖出一手')
    # 当股价穿过BOLL下界的时候买入股票。
    elif data.close.values[-1] < bollBottom.values[-1] and data.close.values[-2] > bollBottom.values[-2]:
        order_percent(symbol=context.symbol, percent=0.3, side=OrderSide_Buy,
                      order_type=OrderType_Market, position_effect=PositionEffect_Open)
        print('以市价单买入一手')
    # 当股价从下往上穿过均值线时买入
    elif p1 > bollAvg_pre.values[-1] and p2 < bollAvg_pre.values[-2]:
        order_percent(symbol=context.symbol, percent=0.3, side=OrderSide_Buy,
                      order_type=OrderType_Market, position_effect=PositionEffect_Open)
        print('以市价单买入一手')
    # 当股价从上往下穿过均值线时卖出
    elif p1 < bollAvg_pre.values[-1] and p2 > bollAvg_pre.values[-2]:
        if pos:  # 有持仓就市价卖出股票。
            order_percent(symbol=context.symbol, percent=0.3, side=OrderSide_Sell,
                          order_type=OrderType_Market, position_effect=PositionEffect_Close)
            print('以市价单卖出一手')


def on_order_status(context, order):
    # 标的代码
    symbol = order['symbol']
    # 委托价格
    price = order['price']
    # 委托数量
    volume = order['volume']
    # 目标仓位
    target_percent = order['target_percent']
    # 查看下单后的委托状态，等于3代表委托全部成交
    status = order['status']
    # 买卖方向，1为买入，2为卖出
    side = order['side']
    # 开平仓类型，1为开仓，2为平仓
    effect = order['position_effect']
    # 委托类型，1为限价委托，2为市价委托
    order_type = order['order_type']
    if status == 3:
        if effect == 1:
            if side == 1:
                side_effect = '开多仓'
            elif side == 2:
                side_effect = '开空仓'
        else:
            if side == 1:
                side_effect = '平空仓'
            elif side == 2:
                side_effect = '平多仓'
        order_type_word = '限价' if order_type == 1 else '市价'
        print('{}:标的：{}，操作：以{}{}，委托价格：{}，委托数量：{}'.format(context.now, symbol, order_type_word, side_effect, price,
                                                         volume))


def on_backtest_finished(context, indicator):
    print('*' * 50)
    print('回测已完成，请通过右上角“回测历史”功能查询详情。')


if __name__ == '__main__':
    '''
        strategy_id策略ID,由系统生成
        filename文件名,请与本文件名保持一致
        mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID,可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        '''

    run(strategy_id='8916e512-b1bf-11ed-9345-00fff1ca1445',
        filename='main_pre.py',
        mode=MODE_BACKTEST,
        token='82ecee4230a8bb420ae916009ed890fa568338e7',
        backtest_start_time='2009-09-17 13:00:00',
        backtest_end_time='2020-03-21 15:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)
