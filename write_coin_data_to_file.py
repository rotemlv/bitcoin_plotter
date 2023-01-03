import requests  # for making http requests to binance
import json  # for parsing what binance sends back to us
import pandas as pd  # for storing and manipulating the data we get back
import matplotlib.pyplot as plt  # for charts and such
import datetime as dt  # for dealing with times

# example of converting time (as in from a txt file) to a datetime object (for graphing etc)
_a = dt.datetime.strptime('2021-09-13 05:31:59.999000', "%Y-%m-%d %H:%M:%S.%f")

root_url = 'https://api.binance.com/api/v3/klines'  # updated to v3 (for all the nothing I know it to do)
DEFAULT = 'USDT'  # exchange to this - I can't figuratively find a USD proper exchange option here
INTERVAL = '3m'  # default time interval between prices
LIMIT = 480  # set how far back should we sample prices within given interval - 480 with 3 minute interval gives ~24hrs
PERIOD = None  # convert interval to period by multiplying by LIMIT

MANUAL_FILENAME = False  # change to declare a different filename for the data file


def get_all_coin_symbols():
    """Get symbols for all exchungus"""
    return (x for x in json.loads(requests.get('https://api.binance.com/api/v3/ticker/price').content))


def create_coin_designates(exchange_to=DEFAULT, trade_stables=False, ignore_up_downs=True):
    """Return tuples for each coin that supports exchange with exchange_to
    * messy function"""
    coins = []
    for symbol in get_all_coin_symbols():
        if exchange_to in symbol['symbol'] and exchange_to not in symbol['symbol'][:len(exchange_to)] \
                and exchange_to in symbol['symbol']:
            if not trade_stables and 'USD' not in symbol['symbol'][:len('USD') + 1]:
                # print(f"symbol: {symbol['symbol']}")
                coins.append((f"{symbol['symbol'][:symbol['symbol'].find(exchange_to)]}",
                              symbol['symbol']))
            if trade_stables:
                # print(f"symbol: {symbol['symbol']}")
                coins.append((f"{symbol['symbol'][:symbol['symbol'].find(exchange_to)]}",
                              symbol['symbol']))

    # takes care of ignoring trades like BTCUPUSDT etc if needed
    return [coin for coin in coins if 'UP' not in coin[0][-2:] and 'DOWN' not in coin[0][-4:]] if \
        ignore_up_downs else coins


def generate_coins_selection(exchange_to=DEFAULT, trade_stables=False, ignore_up_downs=True):
    """Return list of coins that support exchange to a certain currency"""
    return [coin[0] for coin in create_coin_designates(exchange_to, trade_stables, ignore_up_downs)]


def create_coins_dict(lst):
    """receive a list of tuples for coin and exchanges, create a dict with indices
    such that accessing dct['BTC'] will give the appropriate symbol from the tuple"""
    return {value[0]: value[1] for value in lst}


def get_symbols(find_symbol='BTC'):
    """Use uppercase designation for the desired \'coin\'"""
    api_request = requests.get('https://api.binance.com/api/v3/ticker/price')
    # print(api_request.status_code)
    api = json.loads(api_request.content)
    # returns flags for all exchange options for find_symbol
    symbol_length = len(find_symbol)
    return [x['symbol'] for x in api if find_symbol in x['symbol'][:symbol_length]]


def get_bars(symbol, interval=INTERVAL, limit=LIMIT):
    url = root_url + '?symbol=' + symbol + '&interval=' + interval + '&limit=' + str(limit)
    data = json.loads(requests.get(url).text)
    # print(data)
    df = pd.DataFrame(data)
    df.columns = ['open_time',
                  'o', 'h', 'l', 'c', 'v',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.close_time]
    return df


def plot_coin_data(symbol, interval=INTERVAL):
    """Calls function to get api data, and plots it (only coin data, no closest lines). Calls plt.show()"""
    my_data = get_bars(symbol, interval)
    plt.xlabel(f"Price history for {symbol}")
    # print(my_data) # not needed
    plt.plot(my_data['close_time'].index, my_data['c'].astype('float'))
    plt.show()


def get_data(coin, coins_dict, interval=INTERVAL, plot=False):
    """The main tool of this part - returns the api request in a pandas series ( ? )"""
    if plot:
        plot_coin_data(coins_dict[coin])
    return get_bars(coins_dict[coin], interval)


def choose_a_coin():
    """Asks for user input - returns the string in uppercase (string must fit a coin available in the API)"""
    coin_opts = sorted(generate_coins_selection())
    print('Options:')
    for i, coin in enumerate(coin_opts):
        # pretty print (only newline every 10th coin)
        print(f'{coin:8}', end=' ' if (i + 1) % 10 else '\n')
    choice = input('\nEnter a coin designate')
    while choice.upper() not in coin_opts:
        print(f'Invalid input ({choice.upper()} does not exist in database)!')
        choice = input('Enter a coin designate')
    return choice.upper()


def request_filename_from_user():
    from glob import glob  # verify compatibility, also possible to import os and use os.listdir()
    folder_contents = glob('*')
    folder_contents = [e.lower() for e in folder_contents]
    filename = input('insert file name:')
    while filename.lower() in folder_contents:
        flag = input(f"File {filename} already exists in folder! Replace y/N?").lower() != 'y'
        if flag:
            filename = input('insert file name:')
        else:
            break
    return filename


def write_data_to_file(plot=False):
    """Creates a file with the relevant data (close prices) for a coin chosen by the user.
    File is written as a txt file, contains rows of the format: FormattedDateTime, UnicodeIntTime, ClosePrice"""
    # used to check if file exists. verify compatibility, also possible to import os and use os.listdir()
    from glob import glob

    # get a dict of coins and symbols
    coins_dict = create_coins_dict(create_coin_designates())  # insert arguments for inner call if needed
    all_coins_data = []  # create data for all coins available to trade via USDT
    print_times = False  # not in current use
    grab_all_coins = False
    if grab_all_coins:
        coins_select = generate_coins_selection()
        for coin in coins_select:
            # add option to choose plotting t/f
            all_coins_data.append(get_data(coin, coins_dict))
            print(coin)  # check BTCDOWN etc what is that

    coin_choice = choose_a_coin()
    # add option to choose plotting t/f
    data = get_data(coin_choice, coins_dict, plot=plot)  # dont plot here

    # see times associated with prices
    skuff_const = 16
    filename = request_filename_from_user() if MANUAL_FILENAME else f'{coin_choice}_DATA.txt'
    folder_contents = glob('*')
    folder_contents = [e.lower() for e in folder_contents]
    if filename.lower() in folder_contents:
        flag = input(f"File {filename} already exists in folder! Overwrite? (y/N)").lower()
        while flag.lower() not in {'y', 'n', ''}:
            flag = input(f"File {filename} already exists in folder! Overwrite? (y/N)").lower()
        if not flag == 'y':
            return

    with open(filename, 'w') as f:
        #  open_time o h l c v close_time qav num_trades taker_base_vol taker_quote_vol ignore
        counts = data.count()[0]
        for i in range(counts):
            try:
                # a more anorexic write
                f.write(f"{data['close_time'].index[i]},{data['close_time'][i]},{data['c'][i]}\n")
            except IndexError:
                print("I had an accident")
                raise Exception("Index error - count does not match the number of \"rows\" in the data dict.")
                # break

        print(f'successfully wrote {coin_choice} data to file {filename}')
    return filename

