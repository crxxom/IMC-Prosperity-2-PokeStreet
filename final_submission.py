from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np
import statistics
from statistics import NormalDist
import time


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

    # starfruit directional bet hyperparameters
    starfruit_slope = None
    starfruit_slope_threshold = 0.14
    bidask_tolerance = 2
    max_directional_inventory = 5
    num_directional_bet = 0
    strawberry_slope_threshold = 0.1
    strawberry_max_directional_inventory = 200

    POSITION_LIMIT = {'AMETHYSTS' : 20, 
                      'STARFRUIT': 20,
                      'ORCHIDS': 100,
                      'CHOCOLATE':250,
                      'STRAWBERRIES':350,
                      'ROSES':60,
                      'GIFT_BASKET':60,
                      'COCONUT': 300,
                      'COCONUT_COUPON': 600}
    
    PRODUCTS = {'AMETHYSTS' : 0, 
                'STARFRUIT' : 0,
                'ORCHIDS': 0,
                'CHOCOLATE':0,
                'STRAWBERRIES':0,
                'ROSES':0,
                'GIFT_BASKET':0,
                'COCONUT': 0,
                'COCONUT_COUPON': 0}

    trader_data_snapshot = {'AMETHYSTS' : [],
                            
                            'STARFRUIT' : {
                                'STARFRUIT_ts' : [],
                                'STARFRUIT_short_ma': [],
                                'STARFRUIT_long_ma': [],
                                'num_directional_bet': 0,
                                'SHORT_TERM_MOVING_AVERAGE': 5,
                                'LONG_TERM_MOVING_AVERAGE': 40,
                            },

                            'ORCHIDS': {
                                'production_ts': [], # store production time series
                                'price_direction': 0, # -1 decrease, 0 unchanged, 1 increasing
                                'insufficient_sunlight_tracker': 0, # to keep track of 10 minute forecast from insufficient sunlight
                                'SUNLIGHT_THRESHOLD': 2500, # minimum sunlight to consider it as a tick with sunlight
                                'directional_bet_limit': 50, # maximum directional bet position limit
                                'direction_bet_position': 0, # current position from directional bet
                                'CONFIDENCE_THRESHOLD': 2, # currently useless
                                'conversion': 0, # conversion amount - currently useless
                                'starting_position': 0, # starting position of each iteration
                                'cumulative_arb_pos': 0, # cumulative arbitrage trades, just for analysis
                                'export_profit-best_ask': 0,
                                'best_bid-import_cost': 0,
                                'directional_bet_target_inventory': 0, # the target inventory for directional bet
                            },

                            'BASKET': {
                                'bundle_pos': 0,
                                'bundle_limit':58,

                                'basket_std': 76.42310842343223,
                                'signal_std': 0.4,
                                'basket_premium': 380,

                                'rolling_window': 100,
                                'spread_ts': [380]*100,

                                'buy_delay_tracker': 0,
                                'sell_delay_tracker': 0,
                                'buy_delay_signal': 1,
                                'sell_delay_signal': 1,

                                'signal_std_upper': 1.5,
                                'pos_lowersignal_bound': 50,
                                
                                'rose_choco_ratio': 1.8328332608080198,
                                'rose_choco_ratio_std': 0.012955391130802825,
                                'rose_choco_signal_ratio_std': 0.5,
                                'rose_choco_prem': 1323.79745,
                                'rose_choco_std': 111.92672097489925,
                                'rose_choco_signal_std': 1.2, # 0.6: 378k, 0.9: 381k 1.2: 390k

                                # 'STRAWBERRY_short_ma': [],
                                # 'STRAWBERRY_long_ma': [],
                                # 'STRAWBERRY_SHORT_TERM_MOVING_AVERAGE': 5,
                                # 'STRAWBERRY_LONG_TERM_MOVING_AVERAGE': 40,
                            },

                            'COCONUT': {
                                'K': 10000,
                                'T': 245, # MINUS ONE EVERY ROUND
                                'timestamp_counter': 0,
                                'r': 0,
                                'sigma': 0.15996826171874962,

                                'spread_ts': [],
                                'spread_mean': 1.477081480644844,
                                'spread_std': 13.504750883260034,
                                'signal_std': 0.5, # 0.5, 12k; 1, 7k
                                'max_pos_piter': 600,
                                'delta_hedging_freq': 20,
                                'delta_hedging_rate': 0.5,

                                'IV_mean':0.16090022222584424,
                                'IV_std': 0.0034364302155112576,
                                'IV_signal_std': 0.5,
                                'could_not_converge': False,

                                'RSI_period': 14,
                            },

                            'R5': {
                                # cumulative position by bots
                                'person_pos': {
                                    'Rhianna': PRODUCTS,
                                },
                                # actions taken in current round by bots
                                'person_actions': {
                                    'Rhianna': PRODUCTS,
                                },

                                'roses_direction': 0,

                                'strawberry_target': 0,
                            }
                    }

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

    def market_making(self, product, order_depth, acc_bid, acc_ask, position):
        orders: list[Order] = []

        if len(order_depth.buy_orders.keys())==0 or len(order_depth.sell_orders.keys())==0:
            return orders
        
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))



        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = position[product]

        mx_with_buy = -1

        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid+acc_ask)/2

        # i am buying
        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT[product]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
                continue

            # directional bet
            if (product=='STARFRUIT') and self.starfruit_slope>self.starfruit_slope_threshold and (cpos < 0) and ask < acc_bid+self.bidask_tolerance:
                order_for = min(-vol, self.max_directional_inventory - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
                # logger.print(f"Buy Traded at ${ask}: {order_for}, midprice is {mprice_actual} where prediction is {mprice_ours} with cpos {cpos-order_for}")
                
                self.num_directional_bet += 1

        if product == "STRAWBERRIES":
            logger.print(f"Strawberry slope, {self.starfruit_slope}")
        
        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        # directional bets
        if (product=='STARFRUIT') and self.starfruit_slope>self.starfruit_slope_threshold and (cpos < 0):
            
            # mx_with_buy = max(mx_with_buy, ask)
            order_for = min(self.POSITION_LIMIT[product]*2, self.max_directional_inventory - cpos)
            cpos += order_for
            assert(order_for >= 0)
            orders.append(Order(product, acc_ask+self.bidask_tolerance, order_for))
            # logger.print(f"Buy BOT Traded at ${ask}: {order_for}, midprice is {mprice_actual} where prediction is {mprice_ours} with cpos {cpos-order_for}")
            
            self.num_directional_bet += 1
            

        # directional bets
        if (product=='STRAWBERRIES') and self.starfruit_slope>self.strawberry_slope_threshold and (cpos < 0):
            # mx_with_buy = max(mx_with_buy, ask)
            order_for = min(700, self.strawberry_max_directional_inventory - cpos)
            cpos += order_for
            assert(order_for >= 0)
            orders.append(Order(product, acc_ask+self.bidask_tolerance, order_for))
            # logger.print(f"Buy BOT Traded at ${ask}: {order_for}, midprice is {mprice_actual} where prediction is {mprice_ours} with cpos {cpos-order_for}")
            self.num_directional_bet += 1

        if (cpos < self.POSITION_LIMIT[product]) and (cpos > 15):
            num = min(self.POSITION_LIMIT[product]*2, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            cpos += num
            
        if (cpos < self.POSITION_LIMIT[product]) and (position[product] < 0):
            num = min(self.POSITION_LIMIT[product]*2, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            cpos += num
        
        if cpos < self.POSITION_LIMIT[product]:
            num = min(self.POSITION_LIMIT[product]*2, self.POSITION_LIMIT[product] - cpos)
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

            order_for = max(-self.POSITION_LIMIT[product]*2, -(self.max_directional_inventory + cpos))
            cpos += order_for
            assert(order_for <= 0)
            orders.append(Order(product, acc_bid-self.bidask_tolerance, order_for))
            # logger.print(f"Sell BOT Traded at ${bid}: {order_for}, midprice is {mprice_actual} where prediction is {mprice_ours} with cpos {cpos-order_for}")
            
            self.num_directional_bet += 1

        if product=="STRAWBERRIES" and self.starfruit_slope < -self.strawberry_slope_threshold and (cpos > 0):

            order_for = max(-700, -(self.strawberry_max_directional_inventory + cpos))
            cpos += order_for
            assert(order_for <= 0)
            orders.append(Order(product, acc_bid-self.bidask_tolerance, order_for))
            # logger.print(f"Sell BOT Traded at ${bid}: {order_for}, midprice is {mprice_actual} where prediction is {mprice_ours} with cpos {cpos-order_for}")

        if (cpos > -self.POSITION_LIMIT[product]) and (cpos < -15):
            num = max(-self.POSITION_LIMIT[product]*2, -self.POSITION_LIMIT[product]-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num
            
        if (cpos > -self.POSITION_LIMIT[product]) and (position[product] > 0):
            num = max(-self.POSITION_LIMIT[product]*2, -self.POSITION_LIMIT[product]-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT[product]:
            num = max(-self.POSITION_LIMIT[product]*2, -self.POSITION_LIMIT[product]-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
        
    def compute_orders(self, product, order_depth, acc_bid, acc_ask, position):

        if product == "AMETHYSTS":
            return self.market_making(product, order_depth, acc_bid, acc_ask, position)

        elif product == "STARFRUIT":
            return self.market_making(product, order_depth, acc_bid, acc_ask, position)
            # return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product], position)
    

    def compute_orchid_production_ts(self, product, conversionObservations):
        
        # get number of days that sunlight intensity > 2500
        sunlight_ts_bin = bin(int(self.trader_data_snapshot['ORCHIDS']['SUNLIGHT_ts'], 16))[2:].zfill(len(self.trader_data_snapshot['ORCHIDS']['SUNLIGHT_ts'])*4)
        sunlight_ts = list(sunlight_ts_bin)
        sunlight_ts.append(conversionObservations[product].sunlight > self.trader_data_snapshot['ORCHIDS']['SUNLIGHT_THRESHOLD'])
        sunlight_ts.pop(0)
        percent_with_sunlight = 0
        for i in sunlight_ts:
            percent_with_sunlight+=int(i)
        percent_with_sunlight/len(sunlight_ts)

        logger.print(f"Percent with sunlight: {percent_with_sunlight}")
        
        # calculate discrete production percentage
        if percent_with_sunlight > 7/12:
            # sufficient sunlight
            self.trader_data_snapshot['ORCHIDS']['insufficient_sunlight_tracker'] = max(0, self.trader_data_snapshot['ORCHIDS']['insufficient_sunlight_tracker'] - 1)
        else:
            self.trader_data_snapshot['ORCHIDS']['insufficient_sunlight_tracker'] = 140

        discount = 0
        if self.trader_data_snapshot['ORCHIDS']['insufficient_sunlight_tracker'] > 0:
            discount += 4
        
        cur_humitidy = conversionObservations[product].humidity
        if cur_humitidy < 60:
            discount -= (2) * ((60 - cur_humitidy)//5 + 1)

        if cur_humitidy > 80:
            discount -= (2) * ((cur_humitidy - 80)//5 + 1)
        
        # # for testing purposes
        # random_number = random.choice([-1, 1])
        # discount += max(random_number*2,0)

        self.trader_data_snapshot['ORCHIDS']['production_ts'].append(max(100-discount,0))

        logger.print(f'production series is {max(100-discount,0)}')

        if len(self.trader_data_snapshot['ORCHIDS']['production_ts']) == 1:
            self.trader_data_snapshot['ORCHIDS']['production_ts'].append(max(100-discount,0))

        # logger.print(self.trader_data_snapshot['ORCHIDS']['production_ts'][-1])

        # convert sunlight_ts back to hexademical
        updated_bin = ''.join(sunlight_ts)
        hd = (len(updated_bin) + 3)// 4
        hex_num = '%.*x' % (hd, int('0b'+updated_bin, 0))
        self.trader_data_snapshot['ORCHIDS']['SUNLIGHT_ts'] = hex_num
        return 

    def compute_orchid_production_ts_humidity(self,product, conversionObservations):
        
        discount = 0
        
        cur_humitidy = conversionObservations[product].humidity
        if cur_humitidy < 60:
            discount -= (2) * ((60 - cur_humitidy)//5 + 1)

        if cur_humitidy > 80:
            discount -= (2) * ((cur_humitidy - 80)//5 + 1)
        
        # # for testing purposes
        # random_number = random.choice([-1, 1])
        # discount += max(random_number*2,0)
        logger.print(f"Sunlight: {conversionObservations[product].sunlight}")

        self.trader_data_snapshot['ORCHIDS']['production_ts'].append(max(100-discount,0))


        if len(self.trader_data_snapshot['ORCHIDS']['production_ts']) == 1:
            self.trader_data_snapshot['ORCHIDS']['production_ts'].append(max(100-discount,0))

        if len(self.trader_data_snapshot['ORCHIDS']['production_ts']) > 100:
            self.trader_data_snapshot['ORCHIDS']['production_ts'].pop(0)

        # logger.print(self.trader_data_snapshot['ORCHIDS']['production_ts'][-1])

        return

    def prediction_price_orchids(self, product, conversionObservations):
        sunlight_ts = self.trader_data_snapshot['ORCHIDS']['SUNLIGHT_ts']
        # update time series
        if len(sunlight_ts)<10000:
            sunlight_ts.append(conversionObservations[product].sunlight)
            rolling_sunlight_avg = (sunlight_ts[0]*(10000-len(sunlight_ts)+1)+np.sum(sunlight_ts[1:]))/10000
        else:
            sunlight_ts.append(conversionObservations[product].sunlight)
            sunlight_ts.pop()
            rolling_sunlight_avg = np.mean(sunlight_ts)

        # compute production percentage
        production_pct = 100
        discount = 0
        
        if rolling_sunlight_avg < self.trader_data_snapshot['ORCHIDS']['SUNLIGHT_THRESHOLD']:
            self.trader_data_snapshot['ORCHIDS']['insufficient_sunlight_tracker'] += 1
        else:
            self.trader_data_snapshot['ORCHIDS']['insufficient_sunlight_tracker'] = 0
        
        if self.trader_data_snapshot['ORCHIDS']['insufficient_sunlight_tracker'] > 140:
            discount += 4 * (self.trader_data_snapshot['ORCHIDS']['insufficient_sunlight_tracker'] // 140)
        
        humidity = conversionObservations[product].humidity
        transportFees = conversionObservations[product].transportFees
        importTariff = conversionObservations[product].importTariff
        exportTariff = conversionObservations[product].exportTariff

        if humidity < 60:
            discount += (2) * ((60 - humidity)/5)

        if humidity > 80:
             discount += (2) * ((humidity - 80)/5)
            
        if production_pct - discount > 0:
            production_pct = production_pct - discount 
        else:
            production_pct = 0
        
        logger.print(f"Production_pct: {production_pct}")

        prediction = 2588.437196 + -15.328297*production_pct \
                    -72.137611*transportFees \
                    + 3.567693*importTariff \
                    + 8.050881*exportTariff \
        
        return prediction
        
    def compute_OB_arbitrage_orchids(self, product, order_depth, observations, position):
        # ref = self.get_fairprice_orchids()
        # Arbitrage idea
        # 1.  - import price - import fee - transportation fee + best bid price > 0 -> buy from south island
        # 2.  export price - export fee - transportation fee - best ask price > 0 -> sell to south island
        
        orders: list[Order] = []
        if len(order_depth.buy_orders.keys())==0 or len(order_depth.sell_orders.keys())==0:
            return orders
        
        import_price = observations.conversionObservations[product].askPrice
        export_price = observations.conversionObservations[product].bidPrice
        transportFees = observations.conversionObservations[product].transportFees
        exportTariff = observations.conversionObservations[product].exportTariff
        importTariff = observations.conversionObservations[product].importTariff

        import_cost = import_price + importTariff + transportFees 
        export_profit = export_price - exportTariff - transportFees

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        # sell_vol, best_sell_pr = self.values_extract(osell)
        # buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = position[product]
        conversion_num = 0

        # 1. buy in local market, export to foreign market if there are any arbitrage opportunities
        for ask, vol in osell.items():
            # vol is negative here
            
            # we track the possibility of arbitrage trade
            if self.trader_data_snapshot['ORCHIDS']['export_profit-best_ask'] == 0:
                self.trader_data_snapshot['ORCHIDS']['export_profit-best_ask'] = export_profit - ask


            if (export_profit - ask > 0):
                order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
                conversion_num+=order_for
                # logger.print(f"Buy {order_for} units in local market for {ask}, export profit is {export_profit}, conversion will thus be {-order_for}")
                
                # update the orderbook
                if abs(order_for) == abs(vol):
                    order_depth.sell_orders.pop(ask)
                else:
                    order_depth.sell_orders[ask] = -(abs(vol) - abs(order_for))


        # 2. sell in local market, import from foreign market if there are any arbitrage opportunities
        for bid, vol in obuy.items():
            # vol is positive here

            # we track the possibility of arbitrage trade 
            if self.trader_data_snapshot['ORCHIDS']['best_bid-import_cost'] == 0:
                self.trader_data_snapshot['ORCHIDS']['best_bid-import_cost'] = bid - import_cost
            
            if (bid - import_cost) > 0:
                order_for = max(-vol, -self.POSITION_LIMIT[product]-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
                conversion_num-=order_for
                # logger.print(f"Sell {order_for} units in local market for {bid}, import cost is {import_cost}, conversion will thus be {order_for}")

                # update the orderbook
                if abs(order_for) == abs(vol):
                    order_depth.buy_orders.pop(bid)
                else:
                    order_depth.buy_orders[bid] = (abs(vol) - abs(order_for))
                
        position[product] = cpos
            
        return orders, conversion_num

    def compute_hidden_arbitrage_orchids(self, product, order_depth, observations,position):
        orders: list[Order] = []
        try:
            cpos = position[product]
            import_price = observations.conversionObservations[product].askPrice
            export_price = observations.conversionObservations[product].bidPrice
            transportFees = observations.conversionObservations[product].transportFees
            exportTariff = observations.conversionObservations[product].exportTariff
            importTariff = observations.conversionObservations[product].importTariff

            import_cost = import_price + importTariff + transportFees 
            export_profit = export_price - exportTariff - transportFees

            max_buy = min(self.POSITION_LIMIT[product] - cpos, self.POSITION_LIMIT[product] - self.trader_data_snapshot['ORCHIDS']['starting_position'], self.POSITION_LIMIT[product])
            max_sell = max(-(cpos+self.POSITION_LIMIT[product]), -(self.POSITION_LIMIT[product] + self.trader_data_snapshot['ORCHIDS']['starting_position']),-self.POSITION_LIMIT[product])


            # # generate better signals
            # # import we look pay import ask -> if we know price will go up -> we make sure the threshold is large (2)
            # # import we look pay import ask -> if we know price will go down -> we make sure the threshold is lower (1)

            # orders.append(Order(product, int(export_profit)-2, max_buy)) # we will just never do export arbitrage
            # orders.append(Order(product, int(import_cost)+2, max_sell))

            # # if buy price will go up in next tick, the arbitrage will be more costly, we need a larger compensation
            # if self.trader_data_snapshot['ORCHIDS']['price_direction'] == 1:
            #     orders.append(Order(product, int(import_cost)+3, max_sell))
            # # if buy price will go down in next tick, the arbitrage will be more profitbale, 
            # # we can lower the compensation for a higher probability of execution
            # elif self.trader_data_snapshot['ORCHIDS']['price_direction'] == -1:
            #     orders.append(Order(product, int(import_cost)+2, max_sell))
            # # if it will be roughly the same, we just do normally
            # else:
            #     orders.append(Order(product, int(import_cost)+2, max_sell))

            # return orders


            if importTariff + transportFees < -4:
                deltaP_o = 1
                orders.append(Order(product, math.ceil(import_price + importTariff + transportFees)+deltaP_o, max_sell))
                
            else:
                deltaP_o = 0
                orders.append(Order(product, math.ceil(import_price + importTariff + transportFees)+deltaP_o, max_sell))

            orders.append(Order(product, int(export_profit)-2, max_buy)) # we will just never do export arbitrage
        
        except:
            return orders

        return orders


        # # if the margins are small, make the premium lower, vice versa
        # if self.trader_data_snapshot['ORCHIDS']['best_bid-import_cost'] < 1:
        #     orders.append(Order(product, math.ceil(import_price + importTariff + transportFees), max_sell))

    def directional_bet_orchids(self, product, order_depth, pred_price, position):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        logger.print(f"Osell is: {osell}")

        cpos = position[product]
        
        best_sell_pr = 0
        best_buy_pr = 0

        # we go long
        for ask, vol in osell.items():
            if pred_price > ask + self.trader_data_snapshot['ORCHIDS']['CONFIDENCE_THRESHOLD'] and abs(cpos) < self.trader_data_snapshot['ORCHIDS']['directional_bet_limit']:
                order_for = min(-vol, self.trader_data_snapshot['ORCHIDS']['directional_bet_limit'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
                # update directional bet position
                self.trader_data_snapshot['ORCHIDS']['direction_bet_position'] += order_for
                best_buy_pr = ask
                break
        
        # we go short
        for bid, vol in obuy.items():
            if pred_price < bid - self.trader_data_snapshot['ORCHIDS']['CONFIDENCE_THRESHOLD'] and abs(cpos) < self.trader_data_snapshot['ORCHIDS']['directional_bet_limit']:
                order_for = max(-vol, -self.trader_data_snapshot['ORCHIDS']['directional_bet_limit']-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
                # update directional bet position
                self.trader_data_snapshot['ORCHIDS']['direction_bet_position'] += order_for
                best_sell_pr = bid
                break


        logger.print(f"Prediction price is {pred_price}, given current price is {(best_sell_pr + best_buy_pr) / 2}")
        # update position dictionary
        position[product] = cpos
        
        return orders
    
    def trade_production_signal_orchids(self, product, order_depth, position):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        production_series = self.trader_data_snapshot[product]['production_ts']
        cpos = position[product]
        current_directional_pos = self.trader_data_snapshot[product]['direction_bet_position']

        # production today is greater than yesterday, we predict that price will go down
        if production_series[-1] > production_series[-2]:        
            # simply update our target inventory to -100
            self.trader_data_snapshot[product]['directional_bet_target_inventory'] = -self.trader_data_snapshot[product]['directional_bet_limit']

        # production today is less than yesterday, we predict price will go up
        elif production_series[-1] < production_series[-2]:
            # simply update our target inventory to 100
            if production_series[-1] < 98:
                self.trader_data_snapshot[product]['directional_bet_target_inventory'] = self.trader_data_snapshot[product]['directional_bet_limit']
            else:
                self.trader_data_snapshot[product]['directional_bet_target_inventory'] = 0
        
        # # we will just never long at all - maybe reconsider this later
        # self.trader_data_snapshot[product]['directional_bet_target_inventory'] = min(self.trader_data_snapshot[product]['directional_bet_target_inventory'],0)

        diff_to_target_inv = self.trader_data_snapshot[product]['directional_bet_target_inventory'] - current_directional_pos

        logger.print(f"Target is {self.trader_data_snapshot[product]['directional_bet_target_inventory']}, given current inv pos is {current_directional_pos}, the difference to target is {diff_to_target_inv} ")


        # idea is that if there are current orders running (arbitrage), we will just not execute any orders or trades for this round
        # to avoid confusion since we are just too stupid to think of how to balance the inventory
        if self.trader_data_snapshot['ORCHIDS']['export_profit-best_ask'] > -4 or self.trader_data_snapshot['ORCHIDS']['best_bid-import_cost'] > -4:
            # we close out any positions if necessary
            if self.trader_data_snapshot[product]['direction_bet_position'] < 0:
                
                for ask, vol in osell.items():
                    diff_to_target_inv = 0 - self.trader_data_snapshot[product]['direction_bet_position']
                    inv_space = self.POSITION_LIMIT[product] - cpos # calculate inventory space after taking arbitrage
                    order_for = min(-vol, diff_to_target_inv, inv_space)
                    # update directional pos
                    self.trader_data_snapshot[product]['direction_bet_position'] += order_for
                    cpos += order_for
                    assert(order_for >= 0)
                    orders.append(Order(product, ask, order_for))

            elif self.trader_data_snapshot[product]['direction_bet_position'] > 0:
                for bid, vol in obuy.items():
                    diff_to_target_inv = self.trader_data_snapshot[product]['direction_bet_position']
                    inv_space = -self.POSITION_LIMIT[product]-cpos
                    order_for = max(-vol, diff_to_target_inv, inv_space) # negative number
                    # update directional pos
                    self.trader_data_snapshot[product]['direction_bet_position'] += order_for
                    cpos += order_for
                    assert(order_for <= 0)
                    orders.append(Order(product, bid, order_for))

            return orders


        # execute the orders

        # we are buying
        if diff_to_target_inv > 0:
            for ask, vol in osell.items():
                inv_space = self.POSITION_LIMIT[product] - cpos # calculate inventory space after taking arbitrage
                order_for = min(-vol, diff_to_target_inv, inv_space)
                # update directional pos
                self.trader_data_snapshot[product]['direction_bet_position'] += order_for
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
                break
        
        
         # we are selling
        elif diff_to_target_inv < 0:
            for bid, vol in obuy.items():
                inv_space = -self.POSITION_LIMIT[product]-cpos
                order_for = max(-vol, diff_to_target_inv, inv_space) # negative number
                # update directional pos
                self.trader_data_snapshot[product]['direction_bet_position'] += order_for
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
                break # break since we just take the best price, we do this bit by bit     

        # update position
        position[product] = cpos 
        # update target inventory back to directional bet position
        # self.trader_data_snapshot[product]['directional_bet_target_inventory'] = self.trader_data_snapshot[product]['direction_bet_position']

        return orders


    def compute_orders_basket_delay(self, order_depth,position):

        orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'ROSES', 'STRAWBERRIES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0

            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break
                    
        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 - self.trader_data_snapshot['BASKET']['basket_premium']
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 -self.trader_data_snapshot['BASKET']['basket_premium']


        # self.trader_data_snapshot['BASKET']['spread_ts'].append(res_sell)
        # self.trader_data_snapshot['BASKET']['spread_ts'][]

        logger.print(f"basket - items: {res_buy}")

        trade_at = self.trader_data_snapshot['BASKET']['basket_std']*self.trader_data_snapshot['BASKET']['signal_std']

        pb_pos = position['GIFT_BASKET']
        pb_neg = position['GIFT_BASKET']

        
        # control buy sell delay
        if res_sell > trade_at:
            # 1. if there are existing order for buy signal to fill, cut it out
            if self.trader_data_snapshot['BASKET']['buy_delay_tracker']!=0:
                self.trader_data_snapshot['BASKET']['buy_delay_tracker']==0
            # 2. fill in sell order to be executed later
            if self.trader_data_snapshot['BASKET']['sell_delay_tracker']==0:
                self.trader_data_snapshot['BASKET']['sell_delay_tracker']=self.trader_data_snapshot['BASKET']['sell_delay_signal']
                # logger.print(f'sell signal activated, will sell in {self.trader_data_snapshot['BASKET']['sell_delay_tracker']} days')
        
        elif res_buy < -trade_at:
            # 1. if there are existing order for sell signal to fill, cut it out
            if self.trader_data_snapshot['BASKET']['sell_delay_tracker']!=0:
                self.trader_data_snapshot['BASKET']['sell_delay_tracker']==0
            # 2. fill in buy order to be executed later
            if self.trader_data_snapshot['BASKET']['buy_delay_tracker']==0:
                self.trader_data_snapshot['BASKET']['buy_delay_tracker']=self.trader_data_snapshot['BASKET']['buy_delay_signal']
                # logger.print(f'buy signal activated, will buy in {self.trader_data_snapshot['BASKET']['buy_delay_tracker']} days')

        # logger.print(f'buy tracker: {self.trader_data_snapshot['BASKET']['buy_delay_tracker']}')
        # logger.print(f'sell tracker: {self.trader_data_snapshot['BASKET']['sell_delay_tracker']}')
        # we sell if signal tracker is 1
        if self.trader_data_snapshot['BASKET']['sell_delay_tracker']==1:
            vol = position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                pb_neg -= vol

        # we buy if signal tracker is 1
        elif self.trader_data_snapshot['BASKET']['buy_delay_tracker']==1:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                pb_pos += vol

        self.trader_data_snapshot['BASKET']['buy_delay_tracker'] = max(0, self.trader_data_snapshot['BASKET']['buy_delay_tracker']-1)
        self.trader_data_snapshot['BASKET']['sell_delay_tracker'] = max(0, self.trader_data_snapshot['BASKET']['sell_delay_tracker']-1)

        return orders
    
    def buy_bundle(self, bundle_target, obuy, osell, position):
        orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}

        # determine target vol for each product
        strawberry_target = bundle_target*6
        choco_target = bundle_target*4
        roses_target = bundle_target
        basket_target = -bundle_target

        # place orders for strawberries
        for bid, vol in obuy['STRAWBERRIES'].items():
            if strawberry_target >= 0:
                break
            order_for = max(-vol, strawberry_target)
            strawberry_target -= order_for
            position['STRAWBERRIES'] += order_for
            assert(order_for <= 0)
            orders['STRAWBERRIES'].append(Order('STRAWBERRIES', bid, order_for))
            
        # place orders for chocolate
        for bid, vol in obuy['CHOCOLATE'].items():
            if choco_target >= 0:
                break
            order_for = max(-vol, choco_target)
            choco_target -= order_for
            position['CHOCOLATE'] += order_for
            assert(order_for <= 0)
            orders['CHOCOLATE'].append(Order('CHOCOLATE', bid, order_for))
        
        # place orders for roses
        for bid, vol in obuy['ROSES'].items():
            if roses_target >= 0:
                break
            order_for = max(-vol, roses_target)
            roses_target -= order_for
            position['ROSES'] += order_for
            assert(order_for <= 0)
            orders['ROSES'].append(Order('ROSES', bid, order_for))

        # place orders for basket
        for ask, vol in osell['GIFT_BASKET'].items():
            if basket_target <= 0:
                break
            order_for = min(-vol, basket_target)
            basket_target -= order_for
            position['GIFT_BASKET'] += order_for
            assert(order_for >= 0)
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', ask, order_for))

        return orders

    def sell_bundle(self, bundle_target, obuy, osell, position):
        orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}

    # determine target vol for each product
        strawberry_target = bundle_target*6
        choco_target = bundle_target*4
        roses_target = bundle_target
        basket_target = -bundle_target

        # place orders for strawberries
        for ask, vol in osell['STRAWBERRIES'].items():
            if strawberry_target <= 0:
                break
            order_for = min(-vol, strawberry_target)
            strawberry_target -= order_for
            position['STRAWBERRIES'] += order_for
            assert(order_for >= 0)
            orders['STRAWBERRIES'].append(Order('STRAWBERRIES', ask, order_for))
            
        # place orders for chocolate
        for ask, vol in osell['CHOCOLATE'].items():
            if choco_target <= 0:
                break
            order_for = min(-vol, choco_target)
            choco_target -= order_for
            position['CHOCOLATE'] += order_for
            assert(order_for >= 0)
            orders['CHOCOLATE'].append(Order('CHOCOLATE', ask, order_for))
        
        # place orders for roses
        for ask, vol in osell['ROSES'].items():
            if roses_target <= 0:
                break
            order_for = min(-vol, roses_target)
            roses_target -= order_for
            position['ROSES'] += order_for
            assert(order_for >= 0)
            orders['ROSES'].append(Order('ROSES', ask, order_for))

        # place orders for basket
        for bid, vol in obuy['GIFT_BASKET'].items():
            if basket_target >= 0:
                break
            order_for = max(-vol, basket_target)
            basket_target -= order_for
            position['GIFT_BASKET'] += order_for
            assert(order_for <= 0)
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', bid, order_for))
        
        return orders

    def compute_orders_basket_inbundle(self, order_depth, position):
        orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'ROSES', 'STRAWBERRIES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0

            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                

        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 - 380
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 - 380
        res_close = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 - 380


        trade_at = self.trader_data_snapshot['BASKET']['basket_std']*0.5
        close_at = self.trader_data_snapshot['BASKET']['basket_std']*0.1
        
        # sell bundle - sell 1 gift basket, buy 4 choco, buy 1 roses, buy 6 strawberry
        if res_sell > trade_at:

            # determine how many bundle we sell
            max_bundle = min(vol_buy['GIFT_BASKET'], math.floor(vol_sell['STRAWBERRIES']/6), vol_sell['ROSES'], math.floor(vol_sell['CHOCOLATE']/4))
            bundle_target = min(max_bundle, self.trader_data_snapshot['BASKET']['bundle_limit'] - self.trader_data_snapshot['BASKET']['bundle_pos'])
            self.trader_data_snapshot['BASKET']['bundle_pos'] += bundle_target

            orders_bundle = self.sell_bundle(bundle_target, obuy, osell, position)
            orders['GIFT_BASKET'] += orders_bundle['GIFT_BASKET']
            orders['ROSES'] += orders_bundle['ROSES']
            orders['STRAWBERRIES'] += orders_bundle['STRAWBERRIES']
            orders['CHOCOLATE'] += orders_bundle['CHOCOLATE']

        # buy bundle - buy 1 gift basket, sell 4 choco, sell 1 roses, sell 6 strawberry
        elif res_buy < -trade_at:

            # determine how many bundle we buy
            max_bundle = min(vol_sell['GIFT_BASKET'], math.floor(vol_buy['STRAWBERRIES']/6), vol_buy['ROSES'], math.floor(vol_buy['CHOCOLATE']/4))
            bundle_target = max(-max_bundle, -self.trader_data_snapshot['BASKET']['bundle_limit'] - self.trader_data_snapshot['BASKET']['bundle_pos'])
            self.trader_data_snapshot['BASKET']['bundle_pos'] += bundle_target
            
            orders_bundle = self.buy_bundle(bundle_target, obuy, osell, position)
            orders['GIFT_BASKET'] += orders_bundle['GIFT_BASKET']
            orders['ROSES'] += orders_bundle['ROSES']
            orders['STRAWBERRIES'] += orders_bundle['STRAWBERRIES']
            orders['CHOCOLATE'] += orders_bundle['CHOCOLATE']
       
        
        # close existing position
        elif -close_at < res_close < close_at:
            max_bundle = min(vol_buy['GIFT_BASKET'], math.floor(vol_sell['STRAWBERRIES']/6), vol_sell['ROSES'], math.floor(vol_sell['CHOCOLATE']/4))
            target_bundle = -self.trader_data_snapshot['BASKET']['bundle_pos']

            if target_bundle > 0:
                bundle_target = min(target_bundle, max_bundle)
                self.trader_data_snapshot['BASKET']['bundle_pos'] += bundle_target
                orders_bundle = self.buy_bundle(bundle_target, obuy, osell, position)
                orders['GIFT_BASKET'] += orders_bundle['GIFT_BASKET']
                orders['ROSES'] += orders_bundle['ROSES']
                orders['STRAWBERRIES'] += orders_bundle['STRAWBERRIES']
                orders['CHOCOLATE'] += orders_bundle['CHOCOLATE']

            elif target_bundle < 0:
                bundle_target = max(target_bundle, -max_bundle)
                self.trader_data_snapshot['BASKET']['bundle_pos'] += bundle_target
                orders_bundle = self.sell_bundle(bundle_target, obuy, osell, position)
                orders['GIFT_BASKET'] += orders_bundle['GIFT_BASKET']
                orders['ROSES'] += orders_bundle['ROSES']
                orders['STRAWBERRIES'] += orders_bundle['STRAWBERRIES']
                orders['CHOCOLATE'] += orders_bundle['CHOCOLATE']


        return orders

    def compute_orders_basket_multilayer(self, order_depth,position):
        orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'ROSES', 'STRAWBERRIES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0

            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break
                    
        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 - self.trader_data_snapshot['BASKET']['basket_premium']
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 -self.trader_data_snapshot['BASKET']['basket_premium']

        # self.trader_data_snapshot['BASKET']['spread_ts'].append(res_sell)
        # self.trader_data_snapshot['BASKET']['spread_ts'][]

        logger.print(f"basket - items: {res_buy}")

        trade_at_lower = self.trader_data_snapshot['BASKET']['basket_std']*self.trader_data_snapshot['BASKET']['signal_std']
        trade_at_upper = self.trader_data_snapshot['BASKET']['basket_std']*self.trader_data_snapshot['BASKET']['signal_std_upper']
        
        pb_pos = position['GIFT_BASKET']
        pb_neg = position['GIFT_BASKET']

        # logger.print(f'buy tracker: {self.trader_data_snapshot['BASKET']['buy_delay_tracker']}')
        # logger.print(f'sell tracker: {self.trader_data_snapshot['BASKET']['sell_delay_tracker']}')

        # sell signal
        if res_sell > trade_at_upper:
            vol = position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                pb_neg -= vol

        elif res_sell > trade_at_lower:
            vol = max(position['GIFT_BASKET'] + self.trader_data_snapshot['BASKET']['pos_lowersignal_bound'],0)
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                pb_neg -= vol

        # buy signal
        if res_buy < -trade_at_upper:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                pb_pos += vol

        elif res_buy < -trade_at_lower:
            vol = max(self.POSITION_LIMIT['GIFT_BASKET'] - self.trader_data_snapshot['BASKET']['pos_lowersignal_bound'],0,self.POSITION_LIMIT['GIFT_BASKET']-position['GIFT_BASKET'])
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                pb_pos += vol

        self.trader_data_snapshot['BASKET']['buy_delay_tracker'] = max(0, self.trader_data_snapshot['BASKET']['buy_delay_tracker']-1)
        self.trader_data_snapshot['BASKET']['sell_delay_tracker'] = max(0, self.trader_data_snapshot['BASKET']['sell_delay_tracker']-1)

        return orders

    def compute_orders_basket_rolling(self, order_depth,position):

        orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'ROSES', 'STRAWBERRIES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0

            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break
            
            logger.print(f'Worst buy {p}: {worst_buy[p]}; Worst sell {p}: {worst_sell[p]}')
        
        spread = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6

        self.trader_data_snapshot['BASKET']['spread_ts'].pop(0)
        self.trader_data_snapshot['BASKET']['spread_ts'].append(spread)
        logger.print(f'spread ts: {self.trader_data_snapshot['BASKET']['spread_ts']}')
        rolling_std = np.std(self.trader_data_snapshot['BASKET']['spread_ts'])
        rolling_mean = np.mean(self.trader_data_snapshot['BASKET']['spread_ts'])


        res_buy = spread - rolling_mean
        res_sell = spread - rolling_mean
        
        logger.print(f"basket - items: {res_buy}")

        trade_at = rolling_std*self.trader_data_snapshot['BASKET']['signal_std']

        pb_pos = position['GIFT_BASKET']
        pb_neg = position['GIFT_BASKET']

        
        # control buy sell delay
        if res_sell > trade_at:
            # 1. if there are existing order for buy signal to fill, cut it out
            if self.trader_data_snapshot['BASKET']['buy_delay_tracker']!=0:
                self.trader_data_snapshot['BASKET']['buy_delay_tracker']==0
            # 2. fill in sell order to be executed later
            if self.trader_data_snapshot['BASKET']['sell_delay_tracker']==0:
                self.trader_data_snapshot['BASKET']['sell_delay_tracker']=self.trader_data_snapshot['BASKET']['sell_delay_signal']
                # logger.print(f'sell signal activated, will sell in {self.trader_data_snapshot['BASKET']['sell_delay_tracker']} days')
        
        elif res_buy < -trade_at:
            # 1. if there are existing order for sell signal to fill, cut it out
            if self.trader_data_snapshot['BASKET']['sell_delay_tracker']!=0:
                self.trader_data_snapshot['BASKET']['sell_delay_tracker']==0
            # 2. fill in buy order to be executed later
            if self.trader_data_snapshot['BASKET']['buy_delay_tracker']==0:
                self.trader_data_snapshot['BASKET']['buy_delay_tracker']=self.trader_data_snapshot['BASKET']['buy_delay_signal']
                # logger.print(f'buy signal activated, will buy in {self.trader_data_snapshot['BASKET']['buy_delay_tracker']} days')

        # logger.print(f'buy tracker: {self.trader_data_snapshot['BASKET']['buy_delay_tracker']}')
        # logger.print(f'sell tracker: {self.trader_data_snapshot['BASKET']['sell_delay_tracker']}')
        # we sell if signal tracker is 1
        if self.trader_data_snapshot['BASKET']['sell_delay_tracker']==1:
            vol = position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                pb_neg -= vol

        # we buy if signal tracker is 1
        elif self.trader_data_snapshot['BASKET']['buy_delay_tracker']==1:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                pb_pos += vol

        self.trader_data_snapshot['BASKET']['buy_delay_tracker'] = max(0, self.trader_data_snapshot['BASKET']['buy_delay_tracker']-1)
        self.trader_data_snapshot['BASKET']['sell_delay_tracker'] = max(0, self.trader_data_snapshot['BASKET']['sell_delay_tracker']-1)

        return orders

    def compute_orders_basket(self, order_depth, position):
        orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'ROSES', 'STRAWBERRIES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0

            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break
                    
        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 - self.trader_data_snapshot['BASKET']['basket_premium']
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 -self.trader_data_snapshot['BASKET']['basket_premium']

        # self.trader_data_snapshot['BASKET']['spread_ts'].append(res_sell)
        # self.trader_data_snapshot['BASKET']['spread_ts'][]

        logger.print(f"basket - items: {res_buy}")

        trade_at = self.trader_data_snapshot['BASKET']['basket_std']*self.trader_data_snapshot['BASKET']['signal_std']

        pb_pos = position['GIFT_BASKET']
        pb_neg = position['GIFT_BASKET']

        # control buy sell delay
        if res_sell > trade_at:
            vol = position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                pb_neg -= vol
        
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                pb_pos += vol

        return orders

    def compute_orders_basket_wchoco(self, order_depth, position):
        orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'ROSES', 'STRAWBERRIES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0

            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break
                    
        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 - self.trader_data_snapshot['BASKET']['basket_premium']
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES']*1 - mid_price['STRAWBERRIES']*6 -self.trader_data_snapshot['BASKET']['basket_premium']

        # self.trader_data_snapshot['BASKET']['spread_ts'].append(res_sell)
        # self.trader_data_snapshot['BASKET']['spread_ts'][]

        logger.print(f"basket - items: {res_buy}")

        trade_at = self.trader_data_snapshot['BASKET']['basket_std']*self.trader_data_snapshot['BASKET']['signal_std']

        pb_pos = position['GIFT_BASKET']
        pb_neg = position['GIFT_BASKET']

        # control buy sell delay
        if res_sell > trade_at:
            # sell gift basket
            vol = position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                pb_neg -= vol
            
            # buy choco
            vol = self.POSITION_LIMIT['CHOCOLATE'] - position['CHOCOLATE']
            assert(vol >= 0)
            if vol > 0:
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], vol))

        
        elif res_buy < -trade_at:
            # buy gift basket
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                pb_pos += vol

            # sell choco
            vol = position['CHOCOLATE'] + self.POSITION_LIMIT['CHOCOLATE']
            assert(vol >= 0)
            if vol > 0:
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -vol)) 

        return orders

    def choco_rose_pt(self, order_depth, position):
        orders = {'CHOCOLATE' : [], 'ROSES': []}
        prods = ['CHOCOLATE', 'ROSES']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))
            
            try:
                best_sell[p] = next(iter(osell[p]))
                worst_sell[p] = next(reversed(osell[p]))
            except:
                best_sell[p] = None
                worst_sell[p] = None

            try:
                best_buy[p] = next(iter(obuy[p]))
                worst_buy[p] = next(reversed(obuy[p]))
            except:
                best_buy[p] = None
                worst_sell[p] = None

            if best_buy[p] == None:
                mid_price[p] = best_sell[p]
            elif best_sell[p] == None: 
                mid_price[p] = best_buy[p]
            else:
                mid_price[p] = (best_sell[p] + best_buy[p])/2
        
            
        # ratio = mid_price['ROSES']/mid_price['CHOCOLATE']
        # ratio_prem = ratio - self.trader_data_snapshot['rose_choco_ratio']
        # signal = self.trader_data_snapshot['BASKET']['rose_choco_ratio_std']*self.trader_data_snapshot['BASKET']['rose_choco_ratio_signal_std']

        choco_rose_product = mid_price['CHOCOLATE'] * 2 - mid_price['ROSES']
        premium = choco_rose_product - self.trader_data_snapshot['BASKET']['rose_choco_prem']
        product_signal = self.trader_data_snapshot['BASKET']['rose_choco_std']*self.trader_data_snapshot['BASKET']['rose_choco_signal_std']

        # short product - short 2 choco, long 1 rose (product limit is limit of rose)
        if premium > product_signal:
            roses_target = self.POSITION_LIMIT['ROSES'] - position['ROSES']
            choco_target = -(roses_target * 2)

            if worst_sell['ROSES']!=None and worst_buy['CHOCOLATE']!=None:
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], roses_target))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], choco_target)) 

        # long product - long 2 choco, short 1 rose (product limit is limit of rose)
        elif premium < -product_signal:
            roses_target = - self.POSITION_LIMIT['ROSES'] - position['ROSES']
            choco_target = -(roses_target * 2)
            if worst_buy['ROSES']!=None and worst_sell['CHOCOLATE']!=None:
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], roses_target))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], choco_target)) 

        return orders
 
    def issue_buy(self, orders, product, price, quantity, max_quantity):
        MM_buy_quantity = max_quantity
        if MM_buy_quantity != 0: orders.append(Order(product, price, min(MM_buy_quantity, quantity)))
        MM_buy_quantity -= min(MM_buy_quantity, quantity)
        return MM_buy_quantity
    
    def issue_sell(self, orders, product, price, quantity, max_quantity):
        MM_sell_quantity = max_quantity
        if MM_sell_quantity != 0: orders.append(Order(product, price, -min(MM_sell_quantity, quantity)))
        MM_sell_quantity -= min(MM_sell_quantity, quantity)
        return MM_sell_quantity
    
    def append_cache(self, li, item, maxlen):
        li.append(item)
        if len(li) > maxlen: li.pop(0)
        
    # returns target buy/sell prices by calculating from base price deviations
    def calculate_target_buysell(self, pos, base_price = 10_000, product = 'AMETHYSTS'):
        if product == 'GIFT_BASKET':
            buy_adjust, sell_adjust = -0.4,0.4 
            sgn_pos = 2*(pos>0) - 1
            adjust = sgn_pos * abs(pos)**0.2/0.45
            buy_adjust -= adjust
            sell_adjust -= adjust
            return buy_adjust+base_price, sell_adjust+base_price
    
    # get order book prices at each depth
    def get_depths(self, order_depth, depthskip, depth_len):
        depth_bids = []
        cumulative_quantity = 0
        current_depth = 1
        for buy_order in list(order_depth.buy_orders.items()): # buy orders in order of decreasing price
            price, quantity = buy_order
            cumulative_quantity += abs(quantity)
            while (current_depth <= cumulative_quantity):
                if len(depth_bids) == depth_len: break
                depth_bids.append(price)
                current_depth += depthskip
        depth_asks = []
        cumulative_quantity = 0
        current_depth = 1
        for sell_order in list(order_depth.sell_orders.items()): # buy orders in order of decreasing price
            price, quantity = sell_order
            cumulative_quantity += abs(quantity)
            while (current_depth <= cumulative_quantity):
                if len(depth_asks) == 4: break
                depth_asks.append(price)
                current_depth += depthskip
        return depth_bids, depth_asks
    
    def take_markets_iterative(self, orders, pos, depth_bids, depth_asks, product, fair_price, MM_buy_quantity, MM_sell_quantity):
        ind_buy, ind_sell, cpos = 0, 0, pos
        buys, sells = {}, {}
        while (True):
            target_buy, target_sell = self.calculate_target_buysell(cpos, fair_price, product)
            if (depth_asks[ind_buy] <= target_buy):
                if depth_asks[ind_buy] not in buys.keys(): buys[depth_asks[ind_buy]] = 1
                else: buys[depth_asks[ind_buy]] += 1
                #MM_buy_quantity = self.issue_buy(orders, product, depth_asks[ind_buy], 1, MM_buy_quantity)
                if ind_buy + 1 >= len(depth_asks): break
                ind_buy += 1
                cpos += 1
                if ind_buy >= MM_buy_quantity: break
            elif (depth_bids[ind_sell] >= target_sell):
                if depth_bids[ind_sell] not in sells.keys(): sells[depth_bids[ind_sell]] = 1
                else: sells[depth_bids[ind_sell]] += 1
                #MM_sell_quantity = self.issue_sell(orders, product, depth_bids[ind_sell], 1, MM_sell_quantity)
                if ind_sell + 1 >= len(depth_bids): break
                ind_sell += 1
                cpos -= 1
                if ind_sell >= MM_sell_quantity: break
            else:
                break
        return buys, sells, ind_buy, ind_sell

    def compute_orders_basket_regression(self, state,position):
        components = ['CHOCOLATE','STRAWBERRIES','ROSES','GIFT_BASKET']
        mid_price = {}
        for comp in components:
            if len(state.order_depths[comp].buy_orders.keys()) == 0 or len(state.order_depths[comp].sell_orders.keys()) == 0:
                return []
            
            mid_price[comp] = (list(state.order_depths[comp].buy_orders.items())[0][0] + list(state.order_depths[comp].sell_orders.items())[0][0])/2

        # Iterate over all the keys (the available products) contained in the order depths to find our position
        for key, val in state.position.items():
            position[key] = val
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # How much size we will do when market making
            MM_buy_quantity = self.POSITION_LIMIT[product] - position[product]
            MM_sell_quantity = self.POSITION_LIMIT[product] + position[product]
            if product == 'GIFT_BASKET':
                bid_qty, ask_qty = list(order_depth.buy_orders.items())[0][1], abs(list(order_depth.sell_orders.items())[0][1])
                depth_bids, depth_asks = (self.get_depths(order_depth, depthskip = 1, depth_len = 120))
                gift_basket_movement_price_200 = 194.60574189550456 - 0.54343901*(mid_price['GIFT_BASKET']-4*mid_price['CHOCOLATE'] - 6*mid_price['STRAWBERRIES'] - mid_price['ROSES'])
                gift_basket_movement_price = 1.2807754491903711
                gift_basket_movement_price += 1.42537697 * (bid_qty - ask_qty)
                gift_basket_movement_price += -6.47967002 * (np.sqrt(bid_qty) - np.sqrt(ask_qty))                
                gift_basket_fair_price = gift_basket_movement_price + 0.24*gift_basket_movement_price_200 + mid_price['GIFT_BASKET']
                target_buy, target_sell = self.calculate_target_buysell(position[product], gift_basket_fair_price, product=product)
                buys, sells, ind_buy, ind_sell = self.take_markets_iterative(orders, position[product], depth_bids, depth_asks, product, gift_basket_fair_price, MM_buy_quantity, MM_sell_quantity)
                for key in sorted((sells.keys()),reverse=True): MM_sell_quantity = self.issue_sell(orders, product, key, sells[key], MM_sell_quantity)
                for key in sorted((buys.keys())): MM_buy_quantity = self.issue_buy(orders, product, key, buys[key], MM_buy_quantity)               
        
                return orders



    def binary_search_implied_vol(self, target, S, K, r, T, left=0.0, right=1.0, epsilon=1e-6, max_iterations=1000):
        iteration = 0
        while iteration < max_iterations:
            mid = (left + right) / 2
            value = self.bs_optionpricing(S, K, mid, r, T)[0]

            if abs(value - target) < epsilon:
                return mid

            if value < target:
                left = mid
            else:
                right = mid

            iteration += 1
       
        self.trader_data_snapshot['COCONUT']['could_not_converge']=True
        return self.trader_data_snapshot['COCONUT']['IV_mean']
    
    def bs_optionpricing(self, S, K, sigma, r, T, option_type="call"):
    
        assert option_type in ["call", "put"], "option_type only accepts 'call' or 'put' as argument"
        
        d1 = (np.log(S/K) + (r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == "call":
            price = S*NormalDist(mu=0.0, sigma=1.0).cdf(d1) - K*np.exp(-r*T)*NormalDist(mu=0.0, sigma=1.0).cdf(d2)
            delta = NormalDist(mu=0.0, sigma=1.0).cdf(d1)
            # print("To dynamically replicate payoff of the call at expiration,")
            # print("Delta: ", np.exp(-delta*T)*NormalDist(mu=0.0, sigma=1.0).cdf(d1))
            # print("B:", -K*np.exp(r*T)*NormalDist(mu=0.0, sigma=1.0).cdf(d2))
            
        # elif option_type == "put":
        #     price = K*np.exp(-r*T)*NormalDist.cdf(-d2,0,1) - S*np.exp(-delta*T)*NormalDist.cdf(-d1)
            
            # print("To dynamically replicate payoff of the put at expiration,")
            # print("Delta: ", -np.exp(-delta*T)*NormalDist.cdf(d1,0,1))
            # print("B:", K*np.exp(r*T)*NormalDist.cdf(d2,0,1))

        return price, delta
    
    def trade_coco_bs2(self, order_depth, position):
        orders = {'COCONUT' : [], 'COCONUT_COUPON': []}
        prods = ['COCONUT', 'COCONUT_COUPON']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}
        try:
            for p in prods:
                osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
                obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))
                
                try:
                    best_sell[p] = next(iter(osell[p]))
                    worst_sell[p] = next(reversed(osell[p]))
                except:
                    return orders # just end it
                    

                try:
                    best_buy[p] = next(iter(obuy[p]))
                    worst_buy[p] = next(reversed(obuy[p]))
                except:
                    return orders
                    

                if best_buy[p] == None:
                    mid_price[p] = best_sell[p]
                elif best_sell[p] == None: 
                    mid_price[p] = best_buy[p]
                else:
                    mid_price[p] = (best_sell[p] + best_buy[p])/2
        except:
            return orders
        
        stock_price = mid_price['COCONUT']
        strike_price = self.trader_data_snapshot['COCONUT']['K']
        expiration_date = (10000-self.trader_data_snapshot['COCONUT']['timestamp_counter']+(self.trader_data_snapshot['COCONUT']['T']-1)*10000)/(250*10000)
        self.trader_data_snapshot['COCONUT']['timestamp_counter'] += 1
        sigma = self.trader_data_snapshot['COCONUT']['sigma']
        r = self.trader_data_snapshot['COCONUT']['r']
        
        fair_price = self.bs_optionpricing(stock_price, 
                                           strike_price,
                                           sigma,
                                           r,
                                           expiration_date,
                                           )[0]
        
        option_price = mid_price['COCONUT_COUPON']
        valuation_error = fair_price - option_price
        
        trade_at = self.trader_data_snapshot['COCONUT']['spread_std']*self.trader_data_snapshot['COCONUT']['signal_std']

        inst_spread = valuation_error - self.trader_data_snapshot['COCONUT']['spread_mean']

        # control buy sell delay
        if inst_spread < -trade_at:
            vol = position['COCONUT_COUPON'] + self.POSITION_LIMIT['COCONUT_COUPON']
            vol = min(vol,self.trader_data_snapshot['COCONUT']['max_pos_piter'])
            assert(vol >= 0)
            if vol > 0:
                if worst_buy['COCONUT_COUPON'] == None:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', math.ceil(mid_price['COCONUT_COUPON']), -vol)) 
                else:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_buy['COCONUT_COUPON'], -vol)) 

        elif inst_spread > trade_at:
            vol = self.POSITION_LIMIT['COCONUT_COUPON'] - position['COCONUT_COUPON']
            vol = min(vol,self.trader_data_snapshot['COCONUT']['max_pos_piter'])
            assert(vol >= 0)
            if vol > 0:
                if worst_sell['COCONUT_COUPON'] == None:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', math.floor(mid_price['COCONUT_COUPON']), vol))
                else:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_sell['COCONUT_COUPON'], vol))

        return orders

    def delta_hedging(self, order_depth, position):
        orders = {'COCONUT' : [], 'COCONUT_COUPON': []}
        prods = ['COCONUT', 'COCONUT_COUPON']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))
            
            try:
                best_sell[p] = next(iter(osell[p]))
                worst_sell[p] = next(reversed(osell[p]))
            except:
                return orders
                best_sell[p] = None
                worst_sell[p] = None

            try:
                best_buy[p] = next(iter(obuy[p]))
                worst_buy[p] = next(reversed(obuy[p]))
            except:
                return orders
                best_buy[p] = None
                worst_buy[p] = None

            if best_buy[p] == None:
                mid_price[p] = best_sell[p]
            elif best_sell[p] == None: 
                mid_price[p] = best_buy[p]
            else:
                mid_price[p] = (best_sell[p] + best_buy[p])/2
        
        option_pos = position['COCONUT_COUPON']

        stock_price = mid_price['COCONUT']
        strike_price = self.trader_data_snapshot['COCONUT']['K']
        expiration_date = ((self.trader_data_snapshot['COCONUT']['T'])*10000-self.trader_data_snapshot['COCONUT']['timestamp_counter'])/(250*10000)
        sigma = self.trader_data_snapshot['COCONUT']['sigma']
        r = self.trader_data_snapshot['COCONUT']['r']

        delta = self.bs_optionpricing(stock_price, 
                                           strike_price,
                                           sigma,
                                           r,
                                           expiration_date,
                                           )[1]

        target_pos = ((-delta*self.trader_data_snapshot['COCONUT']['delta_hedging_rate']*option_pos) - position['COCONUT'])

        # control buy sell delay
        if target_pos < 0:
            vol = position['COCONUT'] + self.POSITION_LIMIT['COCONUT']
            assert(vol >= 0)
            vol = max(round(target_pos),-vol)
            if worst_buy['COCONUT']!=None:
                orders['COCONUT'].append(Order('COCONUT', best_buy['COCONUT'], vol)) 

        elif target_pos > 0:
            vol = self.POSITION_LIMIT['COCONUT'] - position['COCONUT']
            vol = min(round(target_pos),vol)
            assert(vol >= 0)
            if worst_sell['COCONUT']!=None:
                orders['COCONUT'].append(Order('COCONUT', best_sell['COCONUT'], vol))

        return orders
        
    def trade_coco_IV(self, order_depth, position):
        orders = {'COCONUT' : [], 'COCONUT_COUPON': []}
        prods = ['COCONUT', 'COCONUT_COUPON']
        
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}
        try:
            for p in prods:
                osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
                obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))
                
                try:
                    best_sell[p] = next(iter(osell[p]))
                    worst_sell[p] = next(reversed(osell[p]))
                except:
                    return orders # just end it
                    best_sell[p] = None
                    worst_sell[p] = None

                try:
                    best_buy[p] = next(iter(obuy[p]))
                    worst_buy[p] = next(reversed(obuy[p]))
                except:
                    return orders
                    best_buy[p] = None
                    worst_buy[p] = None

                if best_buy[p] == None:
                    mid_price[p] = best_sell[p]
                elif best_sell[p] == None: 
                    mid_price[p] = best_buy[p]
                else:
                    mid_price[p] = (best_sell[p] + best_buy[p])/2
        except:
            return orders
        
        stock_price = mid_price['COCONUT']
        strike_price = self.trader_data_snapshot['COCONUT']['K']
        expiration_date = (10000-self.trader_data_snapshot['COCONUT']['timestamp_counter']+(self.trader_data_snapshot['COCONUT']['T']-1)*10000)/(250*10000)
        self.trader_data_snapshot['COCONUT']['timestamp_counter'] += 1
        r = self.trader_data_snapshot['COCONUT']['r']
        option_price = mid_price['COCONUT_COUPON']

        implied_volatility = self.binary_search_implied_vol(option_price, stock_price, strike_price, r, expiration_date, left=0.0, right=1.0, epsilon=1e-6, max_iterations=1000)

        # just end the iteration if can't converge
        if self.trader_data_snapshot['COCONUT']['could_not_converge']:
            self.trader_data_snapshot['COCONUT']['could_not_converge'] = False
            return orders

        valuation_error = implied_volatility - self.trader_data_snapshot['COCONUT']['IV_mean']
        
        logger.print(f"Option_spread: {valuation_error}")

        trade_at = self.trader_data_snapshot['COCONUT']['IV_std']*self.trader_data_snapshot['COCONUT']['IV_signal_std']

        # if IV higher than realised vol, we buy
        if valuation_error > trade_at:
            vol = position['COCONUT_COUPON'] + self.POSITION_LIMIT['COCONUT_COUPON']
            vol = min(vol,self.trader_data_snapshot['COCONUT']['max_pos_piter'])
            assert(vol >= 0)
            if vol > 0:
                if worst_buy['COCONUT_COUPON'] == None:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', math.ceil(mid_price['COCONUT_COUPON']), -vol)) 
                else:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_buy['COCONUT_COUPON'], -vol)) 

        elif valuation_error < -trade_at:
            vol = self.POSITION_LIMIT['COCONUT_COUPON'] - position['COCONUT_COUPON']
            vol = min(vol,self.trader_data_snapshot['COCONUT']['max_pos_piter'])
            assert(vol >= 0)
            if vol > 0:
                if worst_sell['COCONUT_COUPON'] == None:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', math.floor(mid_price['COCONUT_COUPON']), vol))
                else:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_sell['COCONUT_COUPON'], vol))

        return orders
        
    def calculate_rsi(self, prices, n=14):
        """
        Calculate the Relative Strength Index (RSI) of a price series.
        
        Parameters:
            - prices (list or np.ndarray): List or array of price values.
            - n (int): Number of periods to use for RSI calculation (default: 14).
        
        Returns:
            - rsi (np.ndarray): Array of RSI values for each corresponding price.
        """
        deltas = np.diff(prices)
        seed = deltas[:n + 1]
        up_periods = seed[seed >= 0].sum() / n
        down_periods = -seed[seed < 0].sum() / n
        rs = up_periods / down_periods
        rsi = np.zeros_like(prices)
        rsi[:n] = 100.0 - 100.0 / (1.0 + rs)
        
        for i in range(n, len(prices)):
            delta = deltas[i - 1]

            if delta > 0:
                up_periods = delta
                down_periods = 0
            else:
                up_periods = 0
                down_periods = -delta

            rs = (up_periods + (n - 1) * rs) / n
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        return rsi[-1]
    
    def trade_coco_bs_RSI(self, order_depth, position):
        orders = {'COCONUT' : [], 'COCONUT_COUPON': []}
        prods = ['COCONUT', 'COCONUT_COUPON']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}
        try:
            for p in prods:
                osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
                obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))
                
                try:
                    best_sell[p] = next(iter(osell[p]))
                    worst_sell[p] = next(reversed(osell[p]))
                except:
                    return orders # just end it
                    

                try:
                    best_buy[p] = next(iter(obuy[p]))
                    worst_buy[p] = next(reversed(obuy[p]))
                except:
                    return orders
                    

                if best_buy[p] == None:
                    mid_price[p] = best_sell[p]
                elif best_sell[p] == None: 
                    mid_price[p] = best_buy[p]
                else:
                    mid_price[p] = (best_sell[p] + best_buy[p])/2
        except:
            return orders
        

        stock_price = mid_price['COCONUT']
        strike_price = self.trader_data_snapshot['COCONUT']['K']
        expiration_date = (10000-self.trader_data_snapshot['COCONUT']['timestamp_counter']+(self.trader_data_snapshot['COCONUT']['T']-1)*10000)/(self.trader_data_snapshot['COCONUT']['T']*10000)
        self.trader_data_snapshot['COCONUT']['timestamp_counter'] += 1
        sigma = self.trader_data_snapshot['COCONUT']['sigma']
        r = self.trader_data_snapshot['COCONUT']['r']

        fair_price = self.bs_optionpricing(stock_price, 
                                           strike_price,
                                           sigma,
                                           r,
                                           expiration_date,
                                           )[0]
        
        option_price = mid_price['COCONUT_COUPON']
        valuation_error = fair_price - option_price

        self.trader_data_snapshot['COCONUT']['spread_ts'].append(valuation_error)

        if len(self.trader_data_snapshot['COCONUT']['spread_ts']) > self.trader_data_snapshot['COCONUT']['RSI_period']:
            self.trader_data_snapshot['COCONUT']['spread_ts'].pop(0)
            spread_ts = self.trader_data_snapshot['COCONUT']['spread_ts'][-self.trader_data_snapshot['COCONUT']['RSI_period']:]
        else:
            spread_ts = self.trader_data_snapshot['COCONUT']['spread_ts']

        rsi = self.calculate_rsi(spread_ts)

        if rsi<30:
        # we sell
            vol = position['COCONUT_COUPON'] + self.POSITION_LIMIT['COCONUT_COUPON']
            vol = min(vol,self.trader_data_snapshot['COCONUT']['max_pos_piter'])
            assert(vol >= 0)
            if vol > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_buy['COCONUT_COUPON'], -vol)) 
            
        elif rsi>70:
        # we buy
            vol = self.POSITION_LIMIT['COCONUT_COUPON'] - position['COCONUT_COUPON']
            vol = min(vol,self.trader_data_snapshot['COCONUT']['max_pos_piter'])
            assert(vol >= 0)
            if vol > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_sell['COCONUT_COUPON'], vol))

        return orders

    def trade_coco_bs(self, order_depth, position):
        orders = {'COCONUT' : [], 'COCONUT_COUPON': []}
        prods = ['COCONUT', 'COCONUT_COUPON']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}
        try:
            for p in prods:
                osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
                obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))
                
                try:
                    best_sell[p] = next(iter(osell[p]))
                    worst_sell[p] = next(reversed(osell[p]))
                except:
                    return orders # just end it
                    

                try:
                    best_buy[p] = next(iter(obuy[p]))
                    worst_buy[p] = next(reversed(obuy[p]))
                except:
                    return orders
                    

                if best_buy[p] == None:
                    mid_price[p] = best_sell[p]
                elif best_sell[p] == None: 
                    mid_price[p] = best_buy[p]
                else:
                    mid_price[p] = (best_sell[p] + best_buy[p])/2
        except:
            return orders
        
         ## We can maybe improve this with a model. Currenlty use the mid price. Handle edge cases with no buy/sell orders?
        coconut_buys = list(order_depth['COCONUT'].buy_orders.items())
        coconut_sells = list(order_depth['COCONUT'].sell_orders.items())
        underlying_price = ((coconut_buys)[0][0] + (coconut_sells)[0][0])/2
        
        if len(list(order_depth['COCONUT_COUPON'].buy_orders.items())) == 0: curr_midprice = list(order_depth['COCONUT_COUPON'].sell_orders.items())[0][0]
        elif len(list(order_depth['COCONUT_COUPON'].sell_orders.items())) == 0: curr_midprice = list(order_depth['COCONUT_COUPON'].buy_orders.items())[0][0]
        else: curr_midprice = (list(order_depth['COCONUT_COUPON'].buy_orders.items())[0][0] + list(order_depth['COCONUT_COUPON'].sell_orders.items())[0][0])/2
        
        bid_volume_1, bid_volume_2, bid_volume_3, ask_volume_1, ask_volume_2, ask_volume_3 = 0,0,0,0,0,0
        if len(coconut_buys) >= 1: bid_volume_1 = coconut_buys[0][1]
        if len(coconut_buys) >= 2: bid_volume_2 = coconut_buys[1][1]
        if len(coconut_buys) >= 3: bid_volume_3 = coconut_buys[2][1]
        if len(coconut_sells) >= 1: ask_volume_1 = -coconut_sells[0][1]
        if len(coconut_sells) >= 2: ask_volume_2 = -coconut_sells[1][1]
        if len(coconut_sells) >= 3: ask_volume_3 = -coconut_sells[2][1]
        
        feature = [bid_volume_1 - ask_volume_1,
                    np.sqrt(bid_volume_1) - np.sqrt(ask_volume_1),
                    bid_volume_1 + bid_volume_2 - ask_volume_1 - ask_volume_2,
                    np.sqrt(bid_volume_1 + bid_volume_2) - np.sqrt(ask_volume_1 + ask_volume_2),
                    bid_volume_1 + bid_volume_2 + bid_volume_3 - ask_volume_1 - ask_volume_2 - ask_volume_3
                    ]
        #logger.print(feature)
        inter = -0.0021415023892787616
        underlying_adjust = inter + np.array([ 0.00014343, 0.04449674,  0.00040202,  0.0412647,  -0.00141033]).T @ np.array(feature)
        #logger.print(underlying_adjust)
        stock_price = mid_price['COCONUT']
        strike_price = self.trader_data_snapshot['COCONUT']['K']
        expiration_date = (10000-self.trader_data_snapshot['COCONUT']['timestamp_counter']+(self.trader_data_snapshot['COCONUT']['T']-1)*10000)/(250*10000)
        self.trader_data_snapshot['COCONUT']['timestamp_counter'] += 1
        sigma = self.trader_data_snapshot['COCONUT']['sigma']
        r = self.trader_data_snapshot['COCONUT']['r']
        
        fair_price = self.bs_optionpricing(stock_price + underlying_adjust, 
                                           strike_price,
                                           sigma,
                                           r,
                                           expiration_date,
                                           )[0]
        
        option_price = mid_price['COCONUT_COUPON']
        valuation_error = fair_price - option_price
        
        trade_at = self.trader_data_snapshot['COCONUT']['spread_std']*self.trader_data_snapshot['COCONUT']['signal_std']

        inst_spread = valuation_error - self.trader_data_snapshot['COCONUT']['spread_mean']

        # control buy sell delay
        if inst_spread < -trade_at:
            vol = position['COCONUT_COUPON'] + self.POSITION_LIMIT['COCONUT_COUPON']
            vol = min(vol,self.trader_data_snapshot['COCONUT']['max_pos_piter'])
            assert(vol >= 0)
            if vol > 0:
                if worst_buy['COCONUT_COUPON'] == None:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', math.ceil(mid_price['COCONUT_COUPON']), -vol)) 
                else:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_buy['COCONUT_COUPON'], -vol)) 

        elif inst_spread > trade_at:
            vol = self.POSITION_LIMIT['COCONUT_COUPON'] - position['COCONUT_COUPON']
            vol = min(vol,self.trader_data_snapshot['COCONUT']['max_pos_piter'])
            assert(vol >= 0)
            if vol > 0:
                if worst_sell['COCONUT_COUPON'] == None:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', math.floor(mid_price['COCONUT_COUPON']), vol))
                else:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_sell['COCONUT_COUPON'], vol))

        return orders

    def roses_signal(self,order_depth, position):
        orders: list[Order] = []

        trade_signals = self.trader_data_snapshot['R5']['person_actions']
        # parse orderbook
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        try:
            best_sell = next(iter(osell))
            worst_sell = next(reversed(osell))
        except:
            return orders # just end it
        try:
            best_buy = next(iter(obuy))
            worst_buy = next(reversed(obuy))
        except:
            return orders
    
        # buy
        if self.trader_data_snapshot['R5']['roses_direction']==1:
            vol = self.POSITION_LIMIT['ROSES'] - position['ROSES']
            orders.append(Order('ROSES', best_sell, vol)) 

        # sell
        elif self.trader_data_snapshot['R5']['roses_direction']==-1:
            vol = self.POSITION_LIMIT['ROSES'] + position['ROSES']
            orders.append(Order('ROSES', best_buy, -vol)) 
        
        return orders

    def strawberry_signal(self, order_depth,position):
        orders: list[Order] = []
        
        trade_signal = self.trader_data_snapshot['R5']['strawberry_target']
        logger.print(f'trader_signal: {trade_signal}')
        # parse orderbook
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        try:
            best_sell = next(iter(osell))
            worst_sell = next(reversed(osell))
        except:
            return orders # just end it
        try:
            best_buy = next(iter(obuy))
            worst_buy = next(reversed(obuy))
        except:
            return orders
    
        # buy
        if trade_signal>0:
            vol = trade_signal - position['STRAWBERRIES']
            orders.append(Order('STRAWBERRIES', best_sell, vol)) 

        # sell
        elif trade_signal<0:
            vol = -trade_signal + position['STRAWBERRIES']
            orders.append(Order('STRAWBERRIES', best_buy, -vol)) 
        
        return orders


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # initialize tracking variables
        start_time = time.time()
        if state.traderData == "":
            trader_data = self.trader_data_snapshot
            state.traderData = trader_data
        else:
            state.traderData = json.loads(state.traderData)
        
        self.trader_data_snapshot = state.traderData

        position = {'AMETHYSTS' : 0, 
                    'STARFRUIT' : 0,
                    'ORCHIDS': 0,
                    'CHOCOLATE':0,
                    'STRAWBERRIES':0,
                    'ROSES':0,
                    'GIFT_BASKET':0,
                    'COCONUT': 0,
                    'COCONUT_COUPON': 0}
        
        result = {'AMETHYSTS' : [],
                  'STARFRUIT' : [],
                  'ORCHIDS': [],
                  'CHOCOLATE': [],
                  'STRAWBERRIES': [],
                  'ROSES': [],
                  'GIFT_BASKET': [],
                  'COCONUT': [],
                  'COCONUT_COUPON': []}

        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            position[key] = val

        # for key, val in position.items():
        #     logger.print(f'{key} position: {val}')
    
        # handle conversion for ORCHIDS first and rebalance current position on ORCHIDS for previous round conversion
        if self.trader_data_snapshot['ORCHIDS']['direction_bet_position']-position['ORCHIDS'] >= 0:
            conversions = min(self.trader_data_snapshot['ORCHIDS']['direction_bet_position']-position['ORCHIDS'], -position['ORCHIDS'])
        else:
            conversions = max(self.trader_data_snapshot['ORCHIDS']['direction_bet_position']-position['ORCHIDS'], position['ORCHIDS'])

        self.trader_data_snapshot['ORCHIDS']['export_profit-best_ask'] = 0
        self.trader_data_snapshot['ORCHIDS']['best_bid-import_cost'] = 0

        self.trader_data_snapshot['ORCHIDS']['cumulative_arb_pos'] += conversions # useless, just for tracking purposes
        position['ORCHIDS'] += conversions
        self.trader_data_snapshot["ORCHIDS"]["starting_position"] = position['ORCHIDS']

        # compute regression statistics and slope for starfruit directional bets
        starfruit_fp, starfruit_ts_sma = self.get_regression(self.trader_data_snapshot['STARFRUIT']['STARFRUIT_short_ma'], self.get_midprice(state.order_depths['STARFRUIT']), self.trader_data_snapshot['STARFRUIT']['SHORT_TERM_MOVING_AVERAGE'])
        self.trader_data_snapshot['STARFRUIT']['STARFRUIT_short_ma'] = starfruit_ts_sma
        starfruit_fp_lma, starfruit_ts_lma = self.get_regression(self.trader_data_snapshot['STARFRUIT']['STARFRUIT_long_ma'], self.get_midprice(state.order_depths['STARFRUIT']), self.trader_data_snapshot['STARFRUIT']['LONG_TERM_MOVING_AVERAGE'])
        self.trader_data_snapshot['STARFRUIT']['STARFRUIT_long_ma'] = starfruit_ts_lma

        acc_bid = {'AMETHYSTS' : 10000,
                   'STARFRUIT' : round(starfruit_fp)}
        acc_ask = {'AMETHYSTS' : 10000,
                   'STARFRUIT' : round(starfruit_fp)}

        for product in ['AMETHYSTS','STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product], position)
            result[product] += orders

        state.traderData['STARFRUIT']['num_directional_bet'] = self.num_directional_bet

        # Trade Orchid

        # 1. Trade on any foreign exchange arbitrage opportunities matching orderbook
        orders, conversion_num = self.compute_OB_arbitrage_orchids("ORCHIDS", state.order_depths["ORCHIDS"], state.observations, position)
        result["ORCHIDS"] += orders
        # # 2. Directional Bet
        # self.compute_orchid_production_ts_humidity('ORCHIDS', state.observations.conversionObservations) # right now we just use humidity as signal
        # orders = self.trade_production_signal_orchids("ORCHIDS", state.order_depths["ORCHIDS"], position)
        # result["ORCHIDS"] += orders
        # 3. With existing position, offer any bid to hidden bot signals for arbitrage
        orders = self.compute_hidden_arbitrage_orchids("ORCHIDS", state.order_depths["ORCHIDS"], state.observations, position)
        result["ORCHIDS"]+=orders
 
        # # trade basket
        # orders = self.compute_orders_basket(state.order_depths,position)
        # result['GIFT_BASKET'] += orders['GIFT_BASKET']
        # result['ROSES'] += orders['ROSES']
        # result['STRAWBERRIES'] += orders['STRAWBERRIES']
        # result['CHOCOLATE'] += orders['CHOCOLATE']

        # # pair trade choco and rose
        # orders = self.choco_rose_pt(state.order_depths,position)
        # result['ROSES'] += orders['ROSES']
        # result['CHOCOLATE'] += orders['CHOCOLATE']

        #testing
        orders = self.compute_orders_basket_regression(state,position)
        result['GIFT_BASKET'] += orders
        
        # result['ROSES'] += orders['ROSES']
        # result['STRAWBERRIES'] += orders['STRAWBERRIES']
        # result['CHOCOLATE'] += orders['CHOCOLATE']
        # # market make in strawberries
        # # compute regression statistics and slope for starfruit directional bets
        # strawberry_fp, strawberry_ts_sma = self.get_regression(self.trader_data_snapshot['BASKET']['STRAWBERRY_short_ma'], self.get_midprice(state.order_depths['STRAWBERRIES']), self.trader_data_snapshot['BASKET']['STRAWBERRY_SHORT_TERM_MOVING_AVERAGE'])
        # self.trader_data_snapshot['BASKET']['STRAWBERRY_short_ma'] = strawberry_ts_sma
        # strawberry_fp_lma, strawberry_ts_lma = self.get_regression(self.trader_data_snapshot['BASKET']['STRAWBERRY_long_ma'], self.get_midprice(state.order_depths['STRAWBERRIES']), self.trader_data_snapshot['BASKET']['STRAWBERRY_LONG_TERM_MOVING_AVERAGE'])
        # self.trader_data_snapshot['BASKET']['STRAWBERRY_short_ma'] = strawberry_ts_lma

        # acc_bid = {'STRAWBERRIES' : round(strawberry_fp)}
        # acc_ask = {'STRAWBERRIES' : round(strawberry_fp)}

        # for product in ['STRAWBERRIES']:
        #     order_depth: OrderDepth = state.order_depths[product]
        #     orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product], position)
        #     result[product] += orders


        # Coconut
        orders = self.trade_coco_bs(state.order_depths, position)
        result['COCONUT'] += orders['COCONUT']
        result['COCONUT_COUPON'] += orders['COCONUT_COUPON']
        if self.trader_data_snapshot['COCONUT']['timestamp_counter']%self.trader_data_snapshot['COCONUT']['delta_hedging_freq']==0:
            orders = self.delta_hedging(state.order_depths,position)
            result['COCONUT'] += orders['COCONUT']
            result['COCONUT_COUPON'] += orders['COCONUT_COUPON']
            
        # # use implied vol as trading signal
        # orders = self.trade_coco_IV(state.order_depths, position)
        # result['COCONUT'] += orders['COCONUT']
        # result['COCONUT_COUPON'] += orders['COCONUT_COUPON']

        # ROUND 5 

        # parse market trades
        for product in state.market_trades.keys():
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue

                # logger.print(f"{product}: Buyer:{trade.buyer}, Seller: {trade.seller}")

                # update cumulative position
                if self.trader_data_snapshot['R5']['person_pos'].get(trade.buyer):
                    self.trader_data_snapshot['R5']['person_pos'][trade.buyer][product]+= trade.quantity
                else:
                    self.trader_data_snapshot['R5']['person_pos'][trade.buyer] = copy.deepcopy(self.PRODUCTS)
                    self.trader_data_snapshot['R5']['person_pos'][trade.buyer][product] = trade.quantity
                
                if self.trader_data_snapshot['R5']['person_pos'].get(trade.seller):
                    self.trader_data_snapshot['R5']['person_pos'][trade.seller][product]+= -trade.quantity
                else:
                    self.trader_data_snapshot['R5']['person_pos'][trade.buyer] = copy.deepcopy(self.PRODUCTS)
                    self.trader_data_snapshot['R5']['person_pos'][trade.buyer][product] = trade.quantity
                
                # update current action
                if self.trader_data_snapshot['R5']['person_actions'].get(trade.buyer):
                    self.trader_data_snapshot['R5']['person_actions'][trade.buyer][product] = 1
                else:
                    self.trader_data_snapshot['R5']['person_actions'][trade.buyer] = copy.deepcopy(self.PRODUCTS)
                    self.trader_data_snapshot['R5']['person_actions'][trade.buyer][product] = 1
                
                if self.trader_data_snapshot['R5']['person_actions'].get(trade.seller):
                    self.trader_data_snapshot['R5']['person_actions'][trade.seller][product] = -1
                else:
                    self.trader_data_snapshot['R5']['person_actions'][trade.seller] = copy.deepcopy(self.PRODUCTS)
                    self.trader_data_snapshot['R5']['person_actions'][trade.seller][product] = -1
                

                if trade.buyer=='Rhianna' and trade.seller=='Vinnie' and product=='ROSES':
                    self.trader_data_snapshot['R5']['roses_direction'] = 1
                elif trade.buyer=='Vinnie' and trade.seller=='Rhianna' and product=='ROSES':
                    self.trader_data_snapshot['R5']['roses_direction'] = -1

                # if trade.buyer=='Vinnie' and trade.seller=='Remy' and product=='STRAWBERRIES':
                #     self.trader_data_snapshot['R5']['strawberry_target'] = self.POSITION_LIMIT['STRAWBERRIES']
                #     logger.print("We should BUY")
                # elif trade.buyer=='Remy' and trade.seller=='Vinnie' and product=='STRAWBERRIES':
                #     self.trader_data_snapshot['R5']['strawberry_target'] = -self.POSITION_LIMIT['STRAWBERRIES']
                #     logger.print("We should SELL")


        # orders = self.strawberry_signal(state.order_depths['STRAWBERRIES'],position)
        # result['STRAWBERRIES'] += orders


        orders = self.roses_signal(state.order_depths['ROSES'], position)
        result['ROSES'] += orders

        # conversions=0 # DELETE THIS LATER

        # reset actions
        self.trader_data_snapshot['R5']['person_actions'] = {}

        false_state = copy.deepcopy(state)
        false_state.traderData = ""
        logger.flush(false_state, result, conversions, "")

        # update trader data at the end
        state.traderData = self.trader_data_snapshot

        end_time = time.time()
        # logger.print(f"Run time for this iteration is {end_time - start_time} seconds")
        return result, conversions, json.dumps(state.traderData)

