from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_results(pair):
    folder = Path(f'V:/results/renko_static_ohlc/walk-forward/sizes10-1000-5_confs1-2/{pair}')
    files_list = list(folder.glob('*.csv'))
    set_num_list = [int(file.stem[6:]) for file in files_list]
    names_list = [file.name for file in files_list]
    df_list = [pd.read_csv(folder / name, index_col=0) for name in names_list]
    df_dict = dict(zip(set_num_list, df_list))

    # print(df_dict.get(1).columns)
    return df_dict

def get_best(metric):
    results = {}
    for i in range(len(df_dict.keys())):
        df = df_dict.get(i)
        df = df.loc[df['num trades'] > 30]
        best = df.sort_values(metric, ascending=False, ignore_index=True).head(1)
        # TODO if this method doesnt produce good results, it may be because i am just choosing the highest score from
        #  each metric in each period, when i could instead choose the middle of the widest and highest local maximum
        prev = 0
        if len(best.index) > 0:
            # print(f'\n\n{i} best {metric}:\n{best.iloc[0, 1]}')
            results[i] = best.iloc[0, 0] # (best.iloc[0, 0], best.iloc[0, 1]) # tuple for size and confs
            prev = best.iloc[0, 0]
        else:
            # print(f'\n\n{i} empty df')
            results[i] = prev # (None, None)
    return results

df_dict = load_results('BNBUSDT')
# print(df_dict.get(1).columns)
sqn_results = get_best('sqn')
winrate_results = get_best('win rate')
pnl_results = get_best('pnl per day')

t = []
s = []
repeat = 0
for keys,values in sqn_results.items():
    t.append(keys)
    # print(type(values))
    s.append(values)

print(s)

plt.plot(s)

plt.xlabel('Period')
plt.ylabel('Brick Size')
plt.title('BTCUSDT')
plt.show()