import jsonpickle
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import math

def quantify(price, quantity, mode = 'log'):

    if mode == 'log':
        SELL_CONSTANTS = [1.0, .71, .59, .51, .44, .39, .34, .29, .26, .22, .19, .16, .13, .11, .09, .07, .05, .05, .02, .01, 0.0]
    elif mode == 'linear':
        SELL_CONSTANTS = [1.0, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1, .05, 0.0]

    BUY_CONSTANTS = SELL_CONSTANTS[-1::-1]

    lower_price = math.floor(price)
    higher_price = math.ceil(price)
    fraction = price - price // 1
    c_index = round(fraction*20)

    if quantity < 0:
        c_ratio = SELL_CONSTANTS[c_index]
        return [ [lower_price, round(quantity * c_ratio)], [higher_price, quantity - round(quantity * c_ratio)]]
    elif quantity > 0:
        c_ratio = BUY_CONSTANTS[c_index]
        return [ [higher_price, round(quantity * c_ratio)], [lower_price, quantity - round(quantity * c_ratio)]]
    else:
        return []

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
inf = int(1e9)

class Trader:

    

    def __init__(self) -> None:

        # self.historic_ask_prices = dict[Symbol, List[int]] = []
        # self.historic_bid_prices : dict[Symbol, List[int]] = []
        self.historic_mid_prices : dict[Symbol, List[int]] = {}
        self.selling_margin = 1.8
        self.buying_margin = 1.8
        self.moving_average_period = 10
        self.min_pos, self.max_pos = -20, 20
        
    
    def run(self, state: TradingState):
        
        traderData = "simple_mm"
        conversions = 0
        logger.print("start")

        result = {}

        for product in state.listings:
            
            logger.print(f"For {product}")

            if product in state.market_trades:
                current_market_prices = [ [trade.price, trade.quantity] for trade in state.market_trades[product]
                                        if trade.timestamp == state.timestamp ]
            else:
                current_market_prices = []
            
            # If there is a trade happened, take the one with dominant quantity as mid price, else infer it from the average
            # of bid and ask prices
            if len(current_market_prices) > 0:

                current_market_prices.sort(key = lambda x : x[1], reverse=True)
                current_mid_price = current_market_prices[0][0]

            else :

                current_mid_price = (max(state.order_depths[product].buy_orders) + min(state.order_depths[product].sell_orders))/2

            if not product in self.historic_mid_prices:
                self.historic_mid_prices[product] = []
            
            self.historic_mid_prices[product].append(current_mid_price)
            logger.print(f"current_mid_price {current_mid_price}")

            moving_average_mid_price = sum(self.historic_mid_prices[product][self.moving_average_period * -1:]) / len(self.historic_mid_prices[product][self.moving_average_period * -1:])
            logger.print(f"moving_average_mid_price {moving_average_mid_price}")

            orders: List[Order] = []

            match(product):

                case "AMETHYSTS":

                    selling_price = moving_average_mid_price + self.selling_margin
                    buying_price = moving_average_mid_price - self.buying_margin
                    
                    if product in state.position:
                        selling_quantity = -(20 + state.position[product])
                        buying_quantity = 20 - state.position[product]
                    else:
                        selling_quantity = -20
                        buying_quantity = 20

                    for price, quantity in quantify(selling_price, selling_quantity, 'linear'):
                        orders.append(Order(product, price, quantity))

                    for price, quantity in quantify(buying_price, buying_quantity, 'linear'):
                        orders.append(Order(product, price, quantity))

                case "STARFRUIT":

                    selling_price = moving_average_mid_price + self.selling_margin
                    buying_price = moving_average_mid_price - self.buying_margin
                    
                    if product in state.position:
                        selling_quantity = -(20 + state.position[product])
                        buying_quantity = 20 - state.position[product]
                    else:
                        selling_quantity = -20
                        buying_quantity = 20

                    for price, quantity in quantify(selling_price, selling_quantity, 'linear'):
                        orders.append(Order(product, price, quantity))

                    for price, quantity in quantify(buying_price, buying_quantity, 'linear'):
                        orders.append(Order(product, price, quantity))

                case _:
                    
                    continue

            result[product] = orders

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData