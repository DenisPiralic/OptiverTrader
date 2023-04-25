# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
import pandas as pd
import numpy as np

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0

        # Pandas series to hold midpoint prices for future and etf
        self.future_price = pd.Series(dtype='float64')
        self.etf_price = pd.Series(dtype='float64')

        # Pandas series to hold ratios
        self.ratios = pd.Series(dtype='float64')

        # Two variables to hold previous and current sell signal
        self.previous_signal = None
        self.current_signal = None

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                        sequence_number)
        # Try... Except used to report errors in the console since the autotrader class does not
        # flag errors by default
        try:
            # Generate midpoint price
            # Dividing by 200 gets expected prices
            self.midpoint_price = pd.Series((bid_prices[0] + ask_prices[0]) / 200.0)

            # Add midpoint to the instrument price Series
            # Instrument 0 means future price
            if instrument == 0:
                # Using concat to create a new series that contains the new price value
                self.future_price = pd.concat([self.future_price, self.midpoint_price], ignore_index=True)

            # Instrument 1 means ETF price
            else:
                # Using concat to create a new series that contains the new price value
                self.etf_price = pd.concat([self.etf_price, self.midpoint_price], ignore_index=True)

            # Find the price ratio of the Future and ETF price
            # Future / ETF
            # Removing anomalous first and last value
            self.new_ratio = pd.Series(self.future_price.iloc[-1] / self.etf_price.iloc[-1])
            self.ratios = pd.concat([self.ratios, self.new_ratio], ignore_index=True)


            # Calculate Z-score of the ratio
            # Removing final anomalous result
            self.zscore = ((self.ratio - self.ratio.mean()) / self.ratio.std())[:-1]

            # Buy and Sell signals
            # Whenever the z score is less than -1.25 we buy and whenever the z score is greater than
            # 1.25 we sell
            if self.zscore.size > 0:
                # Get the last ZScore
                self.last_zscore = self.zscore.iloc[-1]

                # Buy signal
                if self.last_zscore < -1.25:
                    # Record the current signal
                    self.current_signal = "Buy"
                # Sell signal
                elif self.last_zscore > 1.25:
                    # Record the current signal
                    self.current_signal = "Sell"

            # Boilerplate code that sets the bid and ask price
            # There is the potential to optimise here if a better price can be calculated and here is also
            # where we can look into volume
            price_adjustment = - (self.position // LOT_SIZE) * TICK_SIZE_IN_CENTS
            new_ask_price = ask_prices[0] + price_adjustment if ask_prices[0] != 0 else 0
            new_bid_price = bid_prices[0] + price_adjustment if bid_prices[0] != 0 else 0

            # Only produce a signal if there is a change in the signal
            # This will result in alternating buy and sell signals
            # Signal has changed to buy
            if self.current_signal == "Buy" and self.previous_signal != self.current_signal:
                print("Buy signal")
                # Buy Future and Sell ETF

                # Boilerplate code to create a fill and kill order
                self.ask_id = next(self.order_ids)
                self.ask_price = new_ask_price
                # Changing the order type is a possible area of optimisation - currently a FILL_AND_KIll order
                # is being used
                self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, LOT_SIZE, Lifespan.FILL_AND_KILL)
                self.asks.add(self.ask_id)

                print("Order sent to order book")
                # Set previous signal for later use
                self.previous_signal = "Buy"

            # Signal has changed to sell
            elif self.current_signal == "Sell" and self.previous_signal != self.current_signal:
                print("Sell signal")
                # Sell Future and Buy ETF

                # Boilerplate code to create a fill and kill order
                self.bid_id = next(self.order_ids)
                self.bid_price = new_bid_price
                # Changing the order type is a possible area of optimisation - currently a FILL_AND_KILL order
                # is being used
                self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, LOT_SIZE, Lifespan.FILL_AND_KILL)
                self.bids.add(self.bid_id)

                print("Order sent to order book")
                # Set previous signal for later use
                self.previous_signal = "Sell"

        except Exception as e:
            print(e)

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)
