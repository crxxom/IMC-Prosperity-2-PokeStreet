from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np
import statistics

INF = int(1e9)

import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

logger = Logger()

class Trader:


    SHORT_TERM_MOVING_AVERAGE = 5
    LONG_TERM_MOVING_AVERAGE = 40
    LR_PERIOD = 10

    # starfruit directional bet hyperparameters
    starfruit_slope = None
    starfruit_slope_threshold = 0.14
    bidask_tolerance = 2
    max_directional_inventory = 5
    # inventory_control = 18
    num_directional_bet = 0
    

    
    POSITION_LIMIT = {'AMETHYSTS' : 20, 
                      'STARFRUIT': 20}

    time_series = {'AMETHYSTS' : [],
                   'STARFRUIT_short_ma' : [],
                   'STARFRUIT_long_ma':[], 
                  }

    def get_midprice(self, order_depth):
        
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        return (best_sell_pr + best_buy_pr) / 2
    
    def get_MA(self, ts, tdy_px, period):
        if len(ts) == period:
            ts.pop(0)
            
        ts.append(tdy_px)
        moving_average = statistics.mean(ts)
        
        return moving_average, ts

    def get_regression(self, ts, tdy_px, period):
        if len(ts) == period:
            ts.pop(0)

        ts.append(tdy_px)

        if len(ts) < period:
            prediction = tdy_px
            self.starfruit_slope = 0
        else:
            x = range(len(ts))
            y = ts
            slope, intercept = statistics.linear_regression(x, y)

            polynomial_coef = np.polyfit(x, y, 2)
            polynomial = np.poly1d(polynomial_coef)
            polynomial_derivative = np.polyder(polynomial)

            prediction = polynomial(x[-1])
            # self.starfruit_slope = polynomial_derivative(x[-1])

            # prediction = intercept + slope * (x[-1] + 1)
            self.starfruit_slope = slope
            
        return prediction, ts

    def MLR_prediction(self, ts, tdy_px, period):
        if len(ts) != period:
            ts = [tdy_px] * (period)

        ts.pop(0)
        ts.append(tdy_px)
        
        coef = [0.09452187,  0.19067047 ,  0.34303531,  0.37049416]
        intercept = 6.4507169901571615
        nxt_price = intercept
        for i, val in enumerate(ts):
            nxt_price += val * coef[i]

        return nxt_price

        
    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def market_making(self, product, order_depth, acc_bid, acc_ask, position):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = position[product]

        mx_with_buy = -1

        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid+acc_ask)/2

        logger.print(f'Slope of regression: {self.starfruit_slope}')

        # i am buying
        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT[product]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
                continue

            # if cpos < -self.inventory_control and ask < acc_bid + self.bidask_tolerance:
            #     order_for = min(-vol, 5)
            #     cpos += order_for
            #     assert(order_for >= 0)
            #     orders.append(Order(product, ask, order_for))
            #     logger.print(f"Buy Inventory control, traded at ${ask}:{order_for}")
            #     continue

            # directional bet
            if (product=='STARFRUIT') and self.starfruit_slope>self.starfruit_slope_threshold and (cpos < 0) and ask < acc_bid+self.bidask_tolerance:
                order_for = min(-vol, self.max_directional_inventory - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
                logger.print(f"Buy Traded at ${ask}: {order_for}, midprice is {mprice_actual} where prediction is {mprice_ours} with cpos {cpos-order_for}")
                
                self.num_directional_bet += 1
                



        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        # directional bets
        if (product=='STARFRUIT') and self.starfruit_slope>self.starfruit_slope_threshold and (cpos < 0):
            
            # mx_with_buy = max(mx_with_buy, ask)
            order_for = min(40, self.max_directional_inventory - cpos)
            cpos += order_for
            assert(order_for >= 0)
            orders.append(Order(product, acc_ask+self.bidask_tolerance, order_for))
            logger.print(f"Buy BOT Traded at ${ask}: {order_for}, midprice is {mprice_actual} where prediction is {mprice_ours} with cpos {cpos-order_for}")
            
            self.num_directional_bet += 1
            


        if (cpos < self.POSITION_LIMIT[product]) and (cpos > 15):
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            cpos += num
            
        if (cpos < self.POSITION_LIMIT[product]) and (position[product] < 0):
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            cpos += num
        
        if cpos < self.POSITION_LIMIT[product]:
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = position[product]

        # i am sell 
        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT[product]:
                order_for = max(-vol, -self.POSITION_LIMIT[product]-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
                continue

            # if cpos > self.inventory_control and bid > acc_ask - self.bidask_tolerance:
            #     order_for = max(-vol, -5)
            #     cpos += order_for
            #     assert(order_for <= 0)
            #     orders.append(Order(product, bid, order_for))
            #     logger.print(f"Sell: Inventory control, traded at ${bid}: {order_for}")
            #     continue

            # directional bets
            if product=="STARFRUIT" and self.starfruit_slope < -self.starfruit_slope_threshold and (cpos > 0) and bid > acc_ask-self.bidask_tolerance:

                order_for = max(-vol, -(self.max_directional_inventory + cpos))
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
                logger.print(f"Sell Traded at ${bid}: {order_for}, midprice is {mprice_actual} where prediction is {mprice_ours} with cpos {cpos-order_for}")
                
                self.num_directional_bet += 1


        # directional bets
        if product=="STARFRUIT" and self.starfruit_slope < -self.starfruit_slope_threshold and (cpos > 0):

            order_for = max(-40, -(self.max_directional_inventory + cpos))
            cpos += order_for
            assert(order_for <= 0)
            orders.append(Order(product, acc_bid-self.bidask_tolerance, order_for))
            logger.print(f"Sell BOT Traded at ${bid}: {order_for}, midprice is {mprice_actual} where prediction is {mprice_ours} with cpos {cpos-order_for}")
            
            self.num_directional_bet += 1

        if (cpos > -self.POSITION_LIMIT[product]) and (cpos < -15):
            num = max(-40, -self.POSITION_LIMIT[product]-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num
            
        if (cpos > -self.POSITION_LIMIT[product]) and (position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT[product]-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT[product]:
            num = max(-40, -self.POSITION_LIMIT[product]-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

        
    def compute_orders(self, product, order_depth, acc_bid, acc_ask, position):

        if product == "AMETHYSTS":
            return self.market_making(product, order_depth, acc_bid, acc_ask, position)

        elif product == "STARFRUIT":
            return self.market_making(product, order_depth, acc_bid, acc_ask, position)
            # return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product], position)
        
        
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        if state.traderData == "":
            # initialize tracking variables
            trader_data = {'AMETHYSTS' : [],
                           'STARFRUIT_ts' : [],
                           'STARFRUIT_short_ma': [],
                           'STARFRUIT_long_ma': [],
                           'num_directional_bet': 0}
            
            state.traderData = trader_data            
        else:
            state.traderData = json.loads(state.traderData)
        
        position = {'AMETHYSTS' : 0, 
                'STARFRUIT' : 0}
        
        result = {'AMETHYSTS' : [],
                  'STARFRUIT' : []}

        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            position[key] = val

        for key, val in position.items():
            logger.print(f'{key} position: {val}')

    
        timestamp = state.timestamp


        # rolling linear regression - 3308 profit
        starfruit_fp, starfruit_ts_sma = self.get_regression(state.traderData['STARFRUIT_short_ma'], self.get_midprice(state.order_depths['STARFRUIT']), self.SHORT_TERM_MOVING_AVERAGE)
        
        state.traderData['STARFRUIT_short_ma'] = starfruit_ts_sma
    
        starfruit_fp_lma, starfruit_ts_lma = self.get_regression(state.traderData['STARFRUIT_long_ma'], self.get_midprice(state.order_depths['STARFRUIT']), self.LONG_TERM_MOVING_AVERAGE)
            
        state.traderData['STARFRUIT_long_ma'] = starfruit_ts_lma

        # if starfruit_fp_sma > starfruit_fp_lma:
        #     starfruit_fp = math.floor(starfruit_fp_sma)
        #     starfruit_lb = math.floor(starfruit_fp_sma)
        #     starfruit_up = math.floor(starfruit_fp_sma)
        # else:
        #     starfruit_fp = math.ceil(starfruit_fp_sma)
        #     starfruit_lb = math.ceil(starfruit_fp_sma)
        #     starfruit_up = math.ceil(starfruit_fp_sma)
        

        
        # # ML Model Regression
        # starfruit_fp = self.MLR_prediction(state.traderData['STARFRUIT_ts'], self.get_midprice(state.order_depths['STARFRUIT']), self.LR_PERIOD)

        # LTMA_starfruit, starfruit_ts_lma = self.get_MA(state.traderData['STARFRUIT_long_ma'], self.get_midprice(state.order_depths['STARFRUIT']),self.LONG_TERM_MOVING_AVERAGE)

        # if starfruit_fp > LTMA_starfruit:
        #     starfruit_lb = math.floor(starfruit_fp)
        #     starfruit_ub = math.ceil(starfruit_fp)
        # else:
        #     starfruit_lb = math.floor(starfruit_fp)
        #     starfruit_ub = math.ceil(starfruit_fp)

        
        acc_bid = {'AMETHYSTS' : 10000,
                   'STARFRUIT' : round(starfruit_fp)}
        acc_ask = {'AMETHYSTS' : 10000,
                   'STARFRUIT' : round(starfruit_fp)}

        for product in ['AMETHYSTS','STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product], position)
            result[product] += orders

        state.traderData['num_directional_bet'] = self.num_directional_bet
        logger.print(f"Number of directional bets: {self.num_directional_bet}")
        # totpnl = 0
        conversions = 0
        false_state = copy.deepcopy(state)
        false_state.traderData = str(round(starfruit_fp))
        logger.flush(false_state, result, conversions, "")
        return result, conversions, json.dumps(state.traderData)
