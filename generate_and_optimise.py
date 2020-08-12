### in this code i want to try a renko implementation that uses a fixed (not volatility based) proportion
### of price as the brick size, and the mean (o+h+l+c/4) price from each candle as the input series

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.lines import Line2D
import math
import statistics
import time

### define functions

def create_pairs_list(quote, source='ohlc'):
    if source == 'ohlc':
        stored_files_path = Path(f'V:/ohlc_data/')
        files_list = list(stored_files_path.glob(f'*{quote}-1m-data.csv'))
        pairs = [str(pair)[13:-12] for pair in files_list]
    else:
        folders_list = list(source.iterdir())
        pairs = [str(pair.stem) for pair in folders_list]


    return pairs

### get_dates calculates the index numbers for slicing the ohlc data to feed into the walk-forward testing function
def get_dates(counter, long, short, set):
    if set == 'train':
        from_date = counter * short
        to_date = from_date + long + 1
    elif set == 'test':
        from_date = counter * short + long
        to_date = from_date + short + 1
    return from_date, to_date

def load_data(pair):
    filepath = Path(f'V:/ohlc_data/{pair}-1m-data.csv')
    data = pd.read_csv(filepath, index_col=0)

    data['avg_price'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    price = list(data['avg_price'])  # In live trading, this data can only be known at the close of each period. Potential source of look-ahead bias
    # print(f'ohlc periods: {len(price)}')

    vol = list(data['volume'])

    return price, vol

def create_bricks(size, price, vol):
    # print(f'Creating bricks, size: {size}')
    size = size / 10000  # range objects work in integers, values need to be basis points
    ind = 0
    colour = []
    price_index = []
    open_p = [] # would have called this open but that's a keyword
    close = []
    vol_list = []
    vol_count = 0

    for i in range(len(price)):
        vol_count += vol[i]
        if not colour: # if colour is empty, this is the first brick, so thresholds are simply one brick above and below price[0]
            up_thresh = price[0] + (price[0]*size)
            down_thresh = price[0] - (price[0]*size)
            if price[i] >= up_thresh:
                colour.append(1)
                price_index.append(ind)
                ind = i+1
                open_p.append(price[0])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
            if price[i] <= down_thresh:
                colour.append(0)
                price_index.append(ind)
                ind = i+1
                open_p.append(price[0])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
        if colour and colour[-1] == 0: # if previous brick was down, another down brick would open at the previous close, an up brick would open at the previous open
            up_thresh = open_p[-1] + (open_p[-1]*size)
            down_thresh = close[-1] - (close[-1]*size)
            if price[i] >= up_thresh:
                colour.append(1)
                price_index.append(ind)
                ind = i+1
                open_p.append(open_p[-1])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
            if price[i] <= down_thresh:
                colour.append(0)
                price_index.append(ind)
                ind = i+1
                open_p.append(close[-1])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
        if colour and colour[-1] == 1: # if previous brick was up, another up brick would open at the previous close, a down brick would open at the previous open
            up_thresh = close[-1] + (close[-1]*size)
            down_thresh = open_p[-1] - (open_p[-1]*size)
            if price[i] >= up_thresh:
                colour.append(1)
                price_index.append(ind)
                ind = i+1
                open_p.append(close[-1])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
            if price[i] <= down_thresh:
                colour.append(0)
                price_index.append(ind)
                ind = i+1
                open_p.append(open_p[-1])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0

        #TODO add some code here to close the last brick wherever price ends up so it appears on the plot as a half-formed brick

    # print(f'{len(price_index)} bricks')
    bricks = {'index': price_index, 'colour': colour, 'open': open_p, 'close': close, 'volume': vol_list}
    return pd.DataFrame(bricks)

def create_bricks_forward(size_dict, price, vol, test_length):
    # size_dict has integer keys for the training period, and integer values for the size (in bps) in that period
    # print(f'Creating bricks, size: {size}')
    ind = 0
    colour = []
    price_index = []
    open_p = [] # would have called this open but that's a keyword
    close = []
    vol_list = []
    vol_count = 0

    size = 100 # just for the try/except stuff
    for i in range(len(price)):
        period = int(i / test_length)
        # print(f'i: {i}')
        try:
            size = size_dict.get(period) / 10000  # range objects work in integers, values need to be basis points
            # print(size_dict.get(period))
        except:
            x = 0
            # print(size_dict.get(period))
        vol_count += vol[i]
        if not colour: # if colour is empty, this is the first brick, so thresholds are simply one brick above and below price[0]
            up_thresh = price[0] + (price[0]*size)
            down_thresh = price[0] - (price[0]*size)
            if price[i] >= up_thresh:
                colour.append(1)
                price_index.append(ind)
                ind = i+1
                open_p.append(price[0])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
            if price[i] <= down_thresh:
                colour.append(0)
                price_index.append(ind)
                ind = i+1
                open_p.append(price[0])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
        if colour and colour[-1] == 0: # if previous brick was down, another down brick would open at the previous close, an up brick would open at the previous open
            up_thresh = open_p[-1] + (open_p[-1]*size)
            down_thresh = close[-1] - (close[-1]*size)
            if price[i] >= up_thresh:
                colour.append(1)
                price_index.append(ind)
                ind = i+1
                open_p.append(open_p[-1])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
            if price[i] <= down_thresh:
                colour.append(0)
                price_index.append(ind)
                ind = i+1
                open_p.append(close[-1])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
        if colour and colour[-1] == 1: # if previous brick was up, another up brick would open at the previous close, a down brick would open at the previous open
            up_thresh = close[-1] + (close[-1]*size)
            down_thresh = open_p[-1] - (open_p[-1]*size)
            if price[i] >= up_thresh:
                colour.append(1)
                price_index.append(ind)
                ind = i+1
                open_p.append(close[-1])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0
            if price[i] <= down_thresh:
                colour.append(0)
                price_index.append(ind)
                ind = i+1
                open_p.append(open_p[-1])
                close.append(price[i])
                vol_list.append(vol_count)
                vol_count = 0

        #TODO add some code here to close the last brick wherever price ends up so it appears on the plot as a half-formed brick

    # print(f'{len(price_index)} bricks')
    bricks = {'index': price_index, 'colour': colour, 'open': open_p, 'close': close, 'volume': vol_list}
    return pd.DataFrame(bricks)

### backtest_one takes the output of create_bricks and makes an equity curve and a list of 'signals' (prices at which trades would have triggered)
def backtest_one(confs, bricks, price, tot_vol, printout=False):

    index_list = list(bricks['index'])
    colour_list = list(bricks['colour'])
    close_list = list(bricks['close'])
    vol_list = tot_vol
    trade_list = []
    prev = None
    position = None

    startcash = 1000
    cash = startcash
    asset = 0
    fees = 0.00075
    comm = 1 - fees
    equity_curve = []

    if printout:
        print(f'colour_list: {colour_list}')
    for i in range(confs, len(colour_list)):  # iterate through list of bricks, starting at first potential trend confirmation
        ohlc_limit = index_list[i+1] if i < (len(index_list)-1) else index_list[-1] # no slippage allowed past the signal brick
        sell_condition = sum(colour_list[i - (confs-1):i + 1]) == 0 and prev == 1 and position == 'long'
        buy_condition = sum(colour_list[i - (confs-1):i + 1]) == confs and prev == 0 and position == 'short'
        initial_sell_cond = sum(colour_list[i - (confs-1):i + 1]) == 0
        initial_buy_cond = sum(colour_list[i - (confs-1):i + 1]) == confs
        if printout:
            print('-' * 80)
            print(f'i: {i}')
            print(f'price index: {index_list[i]}')
            print(f'sum_bricks: {sum(colour_list[i-confs:i+1])}, prev: {prev}')
        if prev == None:
            # initial sell condition won't be useful until ive implemented shorting logic
            # if initial_sell_cond: # if the last 'num' bricks were red and preceded by none
            #     ohlc_index = index_list[i] + 1
            #     print(f'ohlc_index before: {ohlc_index}') ####
            #     trade_vol = 0
            #     cash = comm * asset * close_list[i]
            #     while trade_vol < cash and ohlc_index < (len(vol_list)-1 and ohlc_limit):
            #         trade_vol += vol_list[ohlc_index]
            #         trade_vol /= 2 # volume figures are for buys and sells combined, i can only draw on half the liquidity
            #         ohlc_index += 1
            #     print(f'ohlc_index after: {ohlc_index}, trade_vol: {trade_vol}, cash: {cash}')
            #     cash = comm * asset * price[ohlc_index]
            #     equity_curve.append(cash)
            #     if printout:
            #         print(f'sold {asset:.2f} units at {price[ohlc_index]}, commision: {(fees * cash):.3f}')
            #     trade_list.append((i, 's', price[ohlc_index]))  # record a sell signal
            #     position = 'short'
            #     prev = 0  # update prev
            if initial_buy_cond: # if the last 'num' bricks were green and preceded by none
                ohlc_index = index_list[i] + 1
                how_many = 0 # for recording how many ohlc periods it takes to fill the order
                # print(f'ohlc_index before: {ohlc_index}') ####
                trade_vol = 0
                asset = cash * comm / close_list[i]
                cash_value = comm * asset * close_list[i] # position is in base currency but volume is given in quote
                while trade_vol < cash_value and ohlc_index < len(vol_list)-1 and  ohlc_index < ohlc_limit:
                    trade_vol += vol_list[ohlc_index]
                    trade_vol /= 2  # volume figures are for buys and sells combined, i can only draw on half the liquidity
                    ohlc_index += 1
                    how_many += 1
                # print(f'ohlc_index after: {ohlc_index}, trade_vol: {trade_vol}, cash: {cash}')
                asset = cash * comm / price[ohlc_index]
                if printout:
                    print(f'bought {asset:.2f} units at {price[ohlc_index]}, commision: {(fees * cash):.3f}')
                trade_list.append((i, 'b', price[ohlc_index], ohlc_index, how_many))  # record a buy signal
                position = 'long'
                prev = 1  # update prev
        if sell_condition:  # if the last 'num' bricks were red and preceded by a green
            if index_list[i] + 1 < len(price):
                ohlc_index = index_list[i] + 1 # this line causes out of index error on its own
                how_many = 0
            else:
                break
            # print(f'ohlc_index before: {ohlc_index}') ####
            trade_vol = 0
            cash = comm * asset * close_list[i]
            mins = 1
            while trade_vol < cash and ohlc_index < len(vol_list)-1 and  ohlc_index < ohlc_limit:
                mins += 1
                trade_vol += vol_list[ohlc_index] / 2  # volume figures are for buys and sells combined, i can only draw on half the liquidity
                ohlc_index += 1
                how_many += 1
            # print(f'ohlc_index after: {ohlc_index}, trade_vol: {trade_vol}, cash: {cash}')
            cash = comm * asset * price[ohlc_index]
            equity_curve.append(cash)
            if printout:
                print(f'sold {asset:.2f} units at {price[ohlc_index]}, commision: {(fees * cash):.3f}')
            trade_list.append((i, 's', price[ohlc_index], ohlc_index, how_many))  # record a sell signal
            position = 'short'
            prev = 0  # update prev
        if buy_condition:  # if the last 'num' bricks were green and preceded by a red
            if index_list[i] + 1 < len(price):
                ohlc_index = index_list[i] + 1 # this line causes out of index error on its own
                how_many = 0
            else:
                break
            # print(f'ohlc_index before: {ohlc_index}') ####
            trade_vol = 0
            asset = cash * comm / close_list[i]
            cash_value = comm * asset * close_list[i]  # position is in base currency but volume is given in quote
            mins = 1
            while trade_vol < cash_value and ohlc_index < len(vol_list)-1 and  ohlc_index < ohlc_limit:
                mins += 1
                trade_vol += vol_list[ohlc_index] / 2  # volume figures are for buys and sells combined, i can only draw on half the liquidity
                ohlc_index += 1
                how_many += 1
            # print(f'ohlc_index after: {ohlc_index}, trade_vol: {trade_vol}, cash: {cash}')
            asset = cash * comm / price[ohlc_index]
            if printout:
                print(f'bought {asset:.2f} units at {price[ohlc_index]}, commision: {(fees * cash):.3f}')
            trade_list.append((i, 'b', price[ohlc_index], ohlc_index, how_many))  # record a buy signal
            position = 'long'
            prev = 1  # update prev
        if printout:
            if equity_curve:
                print(equity_curve[-1])
    if printout:
        print(f'Number of trades: {len(trade_list)}')

    run = 1
    run_list = []
    for i in range(1, len(colour_list)):
        if colour_list[i] == colour_list[i - 1]:
            run += 1
        else:
            run_list.append(run)
            run = 1
    if len(run_list) >= 2:
        avg_run = statistics.mean(run_list)
        std_run = statistics.stdev(run_list)
    else:
        avg_run = 0
        std_run = 0
    vol_per_trade = sum(tot_vol) / len(tot_vol)
    score = avg_run/ (len(run_list)+1) * vol_per_trade
    # print(f'backtest eq curve: {equity_curve}')
    return {'trades': trade_list, 'equity curve': equity_curve, 'avg run': avg_run, 'std run': std_run, 'brick score': score}

### backtest_range calls create_bricks using a range of brick sizes and then creates a list of 'signals' for each setting
def backtest_range(sizes, confs, price, tot_vol):

    sizes_list = []
    confs_list = []
    trades_array = []
    eq_curves = []
    avg_run_list = []
    std_run_list = []
    score_list = []
    for size in sizes: # optimising for brick size
        # print(f'size: {size}')
        bricks = create_bricks(size, price, tot_vol)
        for num in confs: # optimising for number of bricks to confirm trend change
            # print(f'num: {num}')
            backtest = backtest_one(num, bricks, price, tot_vol)
            sizes_list.append(size)
            confs_list.append(num)
            trades_array.append(backtest['trades'])
            eq_curves.append(backtest['equity curve'])
            avg_run_list.append(backtest['avg run'])
            std_run_list.append(backtest['std run'])
            score_list.append(backtest['brick score'])

    return {'sizes': sizes_list, 'confs': confs_list, 'trades': trades_array, 'eq curves': eq_curves,
            'avg run': avg_run_list, 'std run': std_run_list, 'score': score_list}

### optimise takes signals from backtest_range and returns a dataframe of results and statistics
def optimise(signals, days, pair, train_str, set=None, set_num=None):
    size_list = []
    conf_list = []
    trad_list = []
    prof_list = []
    sqn_list = []
    winrate_list = []
    avg_win_list = []
    avg_loss_list = []
    tpd_list = []
    ppd_list = []
    avg_run_list = []
    std_run_list = []
    score_list = []
    for x in range(len(signals.get('sizes'))):
        startcash = 1000
        cash = startcash
        # asset = 0
        # fees = 0.00075
        # comm = 1 - fees
        # equity_curve = []
        # if signals.get('trades')[0][0][1] == 's':
        #     r = range(1, len(signals.get('trades')[x]))
        # else:
        #     r = range(len(signals.get('trades')[x]))
        #
        # for i in r:
        #     sig = signals.get('trades')[x][i][1]
        #     price = signals.get('trades')[x][i][2]
        #     if sig == 'b':
        #         asset = cash * comm / price
        #         # print(f'bought {asset:.2f} units at {price}, commision: {(fees * cash):.3f}')
        #     else:
        #         cash = comm * asset * price
        #         equity_curve.append(cash)
        #         # print(f'sold {asset:.2f} units at {price}, commision: {(fees * cash):.3f}')

        equity_curve = signals.get('eq curves')[x]
        if len(equity_curve) > 5 and statistics.stdev(equity_curve) > 0 and days > 0:
            profit = (100 * (cash - startcash) / startcash)

            #TODO this pnl_series calc is probably going to be a problem, work out what to do about divide by 0 errors
            pnl_series = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1] for i in range(1, len(equity_curve))]
            if len(pnl_series) > 1:  # to avoid StatisticsError: variance requires at least two data points
                sqn = math.sqrt(len(equity_curve)) * statistics.mean(pnl_series) / statistics.stdev(pnl_series)
            else:
                sqn = -1

            wins = 0
            losses = 0
            win_list = []
            loss_list = []
            for i in range(1, len(pnl_series)):
                if pnl_series[i] > 0:
                    wins += 1
                    win_list.append(pnl_series[i])
                else:
                    losses += 1
                    loss_list.append(pnl_series[i])
            winrate = round(100 * wins / (wins + losses))
            if len(win_list) > 0:
                avg_win = statistics.mean(win_list)
            else:
                avg_win = 0
            if len(loss_list) > 0:
                avg_loss = statistics.mean(loss_list)
            else:
                avg_loss = 0

            trades_per_day = len(equity_curve) / days
            prof_per_day = profit / days #TODO this should use a logarithm

            sx = signals.get('sizes')[x]
            cx = signals.get('confs')[x]
            # print(f'Settings: Size {sx} Confirmations {cx}, {len(equity_curve)} round-trip trades, Profit: {profit:.6}%')
            size_list.append(sx)
            conf_list.append(cx)
            trad_list.append(len(equity_curve))
            prof_list.append(profit)
            sqn_list.append(sqn)
            winrate_list.append(winrate)
            avg_win_list.append(avg_win)
            avg_loss_list.append(avg_loss)
            tpd_list.append(trades_per_day)
            ppd_list.append(prof_per_day)
            avg_run_list.append(signals.get('avg run')[x])
            std_run_list.append(signals.get('std run')[x])
            score_list.append(signals.get('score')[x])

    results = {'size': size_list, 'confs': conf_list, 'num trades': trad_list, 'profit': prof_list, 'sqn': sqn_list,
               'win rate': winrate_list, 'avg wins': avg_win_list, 'avg losses': avg_loss_list,
               'trades per day': tpd_list, 'pnl per day': ppd_list, 'avg run': avg_run_list, 'std run': std_run_list, 'score': score_list}
    results_df = pd.DataFrame(results)

    if set:
        res_path = Path(f'V:/results/renko_static_ohlc/walk-forward/{train_str}/{params}/{pair}')

        res_name = Path(f'{set}_{set_num}.csv')
    else:
        res_path = Path(f'V:/results/renko_static_ohlc/backtest/{params}')
        res_name = Path(f'{pair}.csv')

    res_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(res_path / res_name)

    return results_df

### calculate takes signals from backtest and prints out results and statistics
def calculate(signals, days):
    equity_curve = signals.get('equity curve')
    startcash = 1000
    cash = equity_curve[-1]
    asset = 0
    fees = 0.00075
    comm = 1 - fees

    # this block creates the eq curve list, not needed now
    # equity_curve = []
    # if signals.get('trades')[0][1] == 's':
    #     r = range(1, len(signals.get('trades')))
    # else:
    #     r = range(len(signals.get('trades')))
    # for i in r:
    #     sig = signals.get('trades')[i][1]
    #     price = signals.get('trades')[i][2]
    #     if sig == 'b':
    #         asset = cash * comm / price
    #         # print(f'bought {asset:.2f} units at {price}, commision: {(fees * cash):.3f}')
    #     else:
    #         cash = comm * asset * price
    #         equity_curve.append(cash)
    #         # print(f'sold {asset:.2f} units at {price}, commision: {(fees * cash):.3f}')

    if len(equity_curve) > 5 and statistics.stdev(equity_curve) > 0 and days > 0:
        profit = (100 * (cash - startcash) / startcash)

        # TODO this pnl_series calc is probably going to be a problem, work out what to do about divide by 0 errors
        pnl_series = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1] for i in
                      range(1, len(equity_curve))]
        if len(pnl_series) > 1:  # to avoid StatisticsError: variance requires at least two data points
            sqn = math.sqrt(len(equity_curve)) * statistics.mean(pnl_series) / statistics.stdev(pnl_series)
        else:
            sqn = -1

        wins = 0
        losses = 0
        for i in range(1, len(pnl_series)):
            if pnl_series[i] > 0:
                wins += 1
            else:
                losses += 1
        winrate = round(100 * wins / (wins + losses))

        trades_per_day = len(equity_curve) / days
        prof_per_day = profit / days #TODO this really should be using some kind of logarithm or something

        print(f'{len(equity_curve)} round-trip trades, Profit: {profit:.6}%')
        print(f'SQN: {sqn:.3}, win rate: {winrate}%, avg trades/day: {trades_per_day:.3}, avg profit/day: {prof_per_day:.3}%')
        return {'sqn': sqn, 'win rate': winrate, 'avg trades/day': trades_per_day, 'avg profit/day': prof_per_day}

### draw_bars plots the renko bricks on a chart
def draw_bars(data, num_bars=0):

    # get the last num_bars
    if num_bars == 0:
        df = data
        num_bars = len(df.index)
    else:
        df = data.tail(num_bars)
    renkos = zip(df['open'], df['close'])

    # create the figure
    fig = plt.figure(1, figsize=(20, 10))
    fig.clf()
    axes = fig.gca()

    # plot the bars, green for 'up', red for 'down'
    index = 1
    for open_price, close_price in renkos:
        if (open_price < close_price):
            renko = patch.Rectangle((index, open_price), 1, close_price - open_price, edgecolor='darkgreen',
                                    facecolor='green', alpha=0.5)
            axes.add_patch(renko)
        else:
            renko = patch.Rectangle((index, open_price), 1, close_price - open_price, edgecolor='darkred', facecolor='red',
                                    alpha=0.5)
            axes.add_patch(renko)
        index = index + 1

    # adjust the axes
    plt.xlim([0, num_bars])
    plt.ylim([min(min(df['open']), min(df['close'])), max(max(df['open']), max(df['close']))])
    # fig.suptitle('Bars from ' + min(df['date_time']).strftime("%d-%b-%Y %H:%M") + " to " + max(df['date_time']).strftime("%d-%b-%Y %H:%M") \
    #     + '\nPrice movement = ' + str(price_move), fontsize=14)
    plt.xlabel('Bar Number')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

### draw_ohlc plots a price chart with trades marked to visualise what the strategy actually does
def draw_ohlc(data, price, pair):
    trades = data.get('trades')

    plt.plot(price)

    plt.ylabel('Price')
    plt.title(f'{pair} trades')
    plt.show()

### single_test uses the above functions to test one set of params
def single_test(pair, size, confs, num_bars=0):
    print(f'Testing {pair}, size: {size}, confirmations: {confs}')
    price, vol = load_data(pair)
    bricks = create_bricks(size, price, vol) # size is in basis points
    results = backtest_one(confs, bricks, price, vol)
    print(results)
    days = len(price) / 1440
    calculate(results, days)
    draw_bars(bricks, num_bars)
    plot_eq(results.get('equity curve'), pair, 'sqn')

### test_all uses the above functions to test a range of params
def test_all(s, c, printout=False):
    print(f'Starting tests at {time.ctime()[11:-8]}')
    start = time.perf_counter()

    pairs_list = create_pairs_list('USDT')
    # pairs_list = ['RLCUSDT']
    for pair in pairs_list:
        if printout:
            print(f'Testing {pair}')
        price, vol = load_data(pair)
        res_dict = backtest_range(s, c, price, vol)
        days = len(price) / 1440
        results = optimise(res_dict, days, pair)
        if printout:
            print(f'Tests recorded: {len(results.index)}')
        if len(results.index) > 0:
            if printout:
                print(f'Best SQN: {results["sqn"].max()}')
            best = results['sqn'].argmax()
            if printout:
                print(f'Best settings: {results.iloc[best]}')
        if printout:
            print('-' * 40)

    end = time.perf_counter()
    seconds = round(end - start)
    print(f'Time taken: {seconds // 60} minutes, {seconds % 60} seconds')

### walk_forward uses the above functions to test a range of params in a series of date ranges
def walk_forward(s, c, train_length, test_length, printout=False):
    print(f'Starting tests at {time.ctime()[11:-8]}')
    start = time.perf_counter()

    pairs_list = create_pairs_list('USDT')
    # pairs_list = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    for pair in pairs_list:
        if printout:
            print(f'Testing {pair}')
        training = True
        i = 0
        while training:
            from_index, to_index = get_dates(i, train_length, test_length, 'train')
            price, vol = load_data(pair)
            if (train_length + test_length) > len(price):
                print(f'Not enough data for {pair} test')
                training = False
            elif (to_index + test_length) > len(price):
                print(f'Not enough data for another training period, {pair} finished')
                training = False
            else:
                print(f'training {i} from: {from_index}, to: {to_index}')
                price = price[from_index:to_index]
                vol = vol[from_index:to_index]
                days = (len(price) / 1440)
                res_dict = backtest_range(s, c, price, vol)
                train_string = f'{train_length//1000}k-{test_length//1000}k'
                results = optimise(res_dict, days, pair, train_string, 'train', i)
                if printout:
                    print(f'Tests recorded: {len(results.index)}')
                if len(results.index) > 0:
                    if printout:
                        print(f'Best SQN: {results["sqn"].max()}')
                    best = results['sqn'].argmax()
                    if printout:
                        print(f'Best settings: {results.iloc[best]}')
                if printout:
                    print('-' * 40)
                i += 1

    end = time.perf_counter()
    seconds = round(end - start)
    print(f'Time taken: {seconds // 60} minutes, {seconds % 60} seconds')

def load_results(pair, train_str):
    folder = Path(f'V:/results/renko_static_ohlc/walk-forward/{train_str}/{params}/{pair}')
    files_list = list(folder.glob('*.csv'))
    set_num_list = [int(file.stem[6:]) for file in files_list]
    names_list = [file.name for file in files_list]
    df_list = [pd.read_csv(folder / name, index_col=0) for name in names_list]
    df_dict = dict(zip(set_num_list, df_list))

    # print(df_dict.get(1).columns)
    return df_dict

def get_best(metric, df_dict):
    results = {}
    for i in range(len(df_dict.keys())):
        df = df_dict.get(i)
        df = df.loc[df['num trades'] > 30]
        best = df.sort_values(metric, ascending=False, ignore_index=True).head(1)
        # TODO if this method doesnt produce good results, it may be because i am just choosing the highest score from
        #  each metric in each period, when i could instead choose the middle of the widest and highest local maximum
        if len(best.index) > 0:
            # print(f'\n\n{i} best {metric}:\n{best.iloc[0, 1]}')
            results[i] = best.iloc[0, 0] # (best.iloc[0, 0], best.iloc[0, 1]) # tuple for size and confs
        else:
            # print(f'\n\n{i} empty df')
            results[i] = None # (None, None) # tuple for size and confs
    return results

def plot_eq(eq_curve, pair, metric):
    plt.plot(eq_curve)

    plt.xlabel('Trades')
    plt.ylabel('Equity')
    plt.yscale('log')
    plt.title(f'{pair} optimised by {metric}')
    plt.show()

def forward_run(pair, train_length, test_length, metric, single_run=True, printout=False):
    # this function needs to create bricks according to the settings for each training period, then run a single backtest of those bricks

    # call load_data to get price and vol data
    price, vol = load_data(pair)
    price = price[train_length:]  # forward test starts from the beginning of the first test period
    vol = vol[train_length:]
    days = len(price) / 1440
    train_string = f'{train_length // 1000}k-{test_length // 1000}k'
    if printout:
        print(train_string)

    # call load_results to get walk-forward test results
    df_dict = load_results(pair, train_string)
    if printout:
        print(f'df_dict: {df_dict}')

    # call get_best to get settings for each period for a particular metric
    best = get_best(metric, df_dict)
    # print(best.values())

    # call create create_bricks_forward to generate the renko chart
    bricks = create_bricks_forward(best, price, vol, test_length)
    if printout:
        print(f'bricks: {bricks}')

    # call backtest_one to generate the signals
    backtest = backtest_one(1, bricks, price, vol)
    if printout:
        print(f'backtest: {backtest}')

    # call calculate to generate final statistics
    fwd_results = calculate(backtest, days)

    # call draw_ohlc to plot trades on ohlc chart
    # call draw_bricks to draw renko chart
    if single_run:
        draw_ohlc(backtest, price, pair)
        # draw_bars(bricks, 500)
        # chart the equity curves of the different optimisation metrics
        plot_eq(backtest.get('equity curve'), pair, metric)
        #TODO get draw_ohlc and plot_eq as subplots of the same chart

    return fwd_results

def forward_run_all(train_length, test_length):
    print(f'Starting tests at {time.ctime()[11:-8]}')
    start = time.perf_counter()

    train_string = f'{train_length//1000}k-{test_length//1000}k'
    source = Path(f'V:/results/renko_static_ohlc/walk-forward/{train_string}/{params}')
    pairs_list = create_pairs_list('USDT', source)
    metrics = ['sqn', 'win rate', 'pnl per day', 'avg run', 'score']
    results = {}
    for metric in metrics:
        print(f'running {metric} tests')
        results[metric] = {}
        for pair in pairs_list:
            # print(f'running {pair} tests')
            final_results = forward_run(pair, train_length, test_length, metric, single_run=False)
            results[metric][pair] = final_results
            # print(f'results dictionary: {results}')

    sqn_df = pd.DataFrame(results['sqn'])
    winrate_df = pd.DataFrame(results['win rate'])
    pnl_df = pd.DataFrame(results['pnl per day'])
    avg_run_df = pd.DataFrame(results['avg run'])
    score_df = pd.DataFrame(results['score'])

    res_path = Path(f'V:/results/renko_static_ohlc/forward-run/{train_string}/{params}')
    res_path.mkdir(parents=True, exist_ok=True)

    sqn_df.to_csv(res_path / 'sqn.csv')
    winrate_df.to_csv(res_path / 'winrate.csv')
    pnl_df.to_csv(res_path / 'pnl_per_day.csv')
    avg_run_df.to_csv(res_path / 'avg_run.csv')
    score_df.to_csv(res_path / 'score.csv')

    end = time.perf_counter()
    seconds = round(end - start)
    print(f'Time taken: {seconds // 60} minutes, {seconds % 60} seconds')


### run tests

timescale = '1m'

s_low = 10
s_hi = 1000
s_step = 5
s = range(s_low, s_hi, s_step) # the range of brick sizes to be tested, in basis points
c_low = 1
c_hi = 2
c = range(c_low, c_hi) # the range of confirmations to be tested, ie how many bricks to confirm a trend change
params = f'sizes{s_low}-{s_hi}-{s_step}_confs{c_low}-{c_hi}'

# s, c = [100], [0]
# test_all(s, c, True)

# single_test('BTCUSDT', 190, 1, 500)

# walk_forward(s, c, 80000, 2000)

# forward_run('BTCUSDT', 80000, 2000, 'score')

forward_run_all(80000, 2000)

#TODO it seems as though 1 brick confirmation is best in almost all situations, so at some point i will have to rewrite
# everything to remove all references to the optimisation of confs