from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import statistics

def load_results(pair):
    folder = Path(f'V:/results/renko_static_ohlc/walk-forward/80k-2k/sizes10-600-5_confs1-2/{pair}')
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

pair = 'BNBUSDT'

df_dict = load_results(pair)
# print(df_dict.get(1).columns)
sqn_results = get_best('sqn')
winrate_results = get_best('win rate')

s = []
repeat = 0
count = 0
for keys,values in sqn_results.items():
    if values:
        s.append(values)
        repeat = values
    else:
        s.append(repeat)
        count += 1

t = []
med = 5
for i in range(len(s)):
    if i == 0:
        t.append(s[i])
    elif i <= med:
        x = round(statistics.mean(s[:i]))
        t.append(x)
    else:
        x = round(statistics.mean(s[i-med:i]))
        t.append(x)



print(f'Total values: {len(s)} Nones fixed: {count}')
print(s[:10])
print(t[:10])

plt.plot(s, label='size')
plt.plot(t, label='med')

plt.xlabel('Period')
plt.ylabel('Brick Size')
plt.title(f'{pair}')
plt.legend()
plt.show()