import pandas as pd
import numpy as np
import csv

from collections import Counter

def comp(x, y):
    return 'above/eq' if x >= y else 'below'

# show all df
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# matches df
df_matches = pd.read_csv("2020_LoL_esports_match_data_from_OraclesElixir_20201207.csv")
df_matches = df_matches[df_matches['datacompleteness'] == 'complete']

# game statistics df
df_game = pd.read_csv('champ_game_stats.csv')
df_game = df_game.replace(np.nan, 0)

# avgs of df_game
AVG_WIN = np.mean(df_game['winPercent'])
AVG_PICK = np.mean(df_game['playPercent'])
AVG_BAN = np.mean(df_game['banRate'])
AVG_KILL = np.mean(df_game['kills']) 
AVG_ASSIST = np.mean(df_game['assists']) 
AVG_DEATH = np.mean(df_game['deaths'])
AVG_GOLD = np.mean(df_game['goldEarned'])
AVG_XP = np.mean(df_game['experience'])

# base stats df
df_base = pd.read_csv('champ_base_stats.csv')
df_base = df_base.replace(np.nan, 0)

# champs base stats avgs
# put here

games = set(df_matches['gameid'])
games = [g for g in games if g == g]


# all teams names 
teams = set(df_matches['team'])
teams = [t for t in teams if t == t]

# victory defeat per team
victs = {}
defeats = {}
for t in teams:
    victs[t], defeats[t] = (
        [v for v in set(df_matches[ (df_matches['team'] == t) & 
                                         (df_matches['result'] == 1) ]['gameid']) if v == v],
        [d for d in set(df_matches[ (df_matches['team'] == t) & 
                                         (df_matches['result'] == 0) ]['gameid']) if d == d])

# defining most effective team
# factor = pVictory * (1 + numGames/50)
teams_f = []
for t in teams:
    v, d = len(victs[t]), len(defeats[t])
    numGames = v+d
    pVictory = v / numGames * 100.0
    teams_f.append((t, pVictory * (1+numGames/50) )) 
teams_f = list(sorted(teams_f, key=lambda x: x[1], reverse=True))

d_teams_f = {t: v for t, v in teams_f}

'''
# most win rate may not be most effective
# for t, _ in teams_f:
#     v, d = len(victs[t]), len(defeats[t])
#     print('{} ganhou {:2%}.'.format(t, v/(v+d)))

# most picked per team
team_champs = {}
for name, _ in teams_f:
    champs = [c for c in list(df_matches[df_matches['team'] == name]['champion']) if c == c]
    team_champs[name] = Counter(champs)

# most picked of the 50 best teams
all_champs = Counter()
for t, v in list(team_champs.items())[:50]:
    all_champs += v

# comparing champs from pro to casual
champ_pick_comp = []
for c in all_champs.keys():
    mean_p = np.mean(df_game[df_game['champion'] == c]['playPercent'])
    mean_w = np.mean(df_game[df_game['champion'] == c]['winPercent'])
    mean_kda = (np.mean(df_game[df_game['champion'] == c]['kills']) + 
                np.mean(df_game[df_game['champion'] == c]['assists'])) / (
                np.mean(df_game[df_game['champion'] == c]['deaths']))
    mean_ban = np.mean(df_game[df_game['champion'] == c]['banRate'])
    if mean_p:
        champ_pick_comp.append((all_champs[c],
                                c, 
                                comp(mean_p, AVG_PICK), 
                                comp(mean_w, AVG_WIN),
                                comp(mean_kda, AVG_KDA),
                                comp(mean_ban, AVG_BAN)))

champ_pick_comp = list(sorted(champ_pick_comp, key=lambda x: x[0], reverse=True))
for c in champ_pick_comp:
    print(f'{c[1]} is {c[2]} avg pick and {c[3]} avg win rate and {c[4]} kda and {c[5]} ban rate')


################ making dataset for model #############

columns = list(df_base.columns)[1:]
cols = []
for i in range(2, 11):
    cols += [x+str(i) for x in columns]

columns += cols

with open('model.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    spamwriter.writerow(columns + ['target'])

    for g in games[:6000]:
        won = df_matches[ (df_matches['gameid'] == g) & (df_matches['result'] == 1) ][['team','champion']]
        lost = df_matches[ (df_matches['gameid'] == g) & (df_matches['result'] == 0) ][['team','champion']]

        won_l, lost_l = [], []
        for champ_w, champ_l in zip(list(won['champion'])[:-1], list(lost['champion'])[:-1]):
            won_l += df_base[df_base['Champions'] == champ_w].values.tolist()[0][1:]
            lost_l += df_base[df_base['Champions'] == champ_l].values.tolist()[0][1:]

        spamwriter.writerow(won_l + lost_l + [1])
        spamwriter.writerow(lost_l + won_l + [0])
'''

columns = ['winPercent', 'experience', 'kills', 'deaths', 'assists', 'goldEarned', 'HP', 'MP', 'AD', 'AS', 'AR', 'MR', 'MS', 'Range']
cols = []
for i in range(2, 11):
    cols += [x+str(i) for x in columns]

columns = ['team', 'team2'] + columns +  cols + ['target']

with open('model2.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    spamwriter.writerow(columns)

    for i, g in enumerate(games[:5000]):
        won = df_matches[ (df_matches['gameid'] == g) & (df_matches['result'] == 1) ][['team','champion']]
        lost = df_matches[ (df_matches['gameid'] == g) & (df_matches['result'] == 0) ][['team','champion']]
        
        won_t, lost_t = list(won['team'])[0], list(lost['team'])[0]
        
        if won_t in d_teams_f:
            won_t = d_teams_f[won_t]
        else:
            continue

        if lost_t in d_teams_f:
            lost_t = d_teams_f[lost_t]
        else:
            continue
            
        won_l, lost_l = [], []
        for champ_w, champ_l in zip(list(won['champion'])[:-1], list(lost['champion'])[:-1]):
            l = df_game[ df_game['champion'] == champ_w][['winPercent', 
                                                               'experience', 
                                                               'kills', 
                                                               'deaths', 
                                                               'assists', 
                                                               'goldEarned']].values.tolist()
            if l:
                won_l += l[0]
            else:
                won_l += [AVG_WIN, AVG_XP, AVG_KILL, AVG_DEATH, AVG_ASSIST, AVG_GOLD]

            won_l += df_base[df_base['Champions'] == champ_w][['HP', 
                                                               'MP', 
                                                               'AD', 
                                                               'AS', 
                                                               'AR', 
                                                               'MR', 
                                                               'MS', 
                                                               'Range']].values.tolist()[0]

            l = df_game[ df_game['champion'] == champ_l][['winPercent', 
                                                                'experience', 
                                                                'kills', 
                                                                'deaths', 
                                                                'assists', 
                                                                'goldEarned']].values.tolist()
            if l:
                lost_l += l[0]
            else:
                lost_l += [AVG_WIN, AVG_XP, AVG_KILL, AVG_DEATH, AVG_ASSIST, AVG_GOLD]

            lost_l += df_base[df_base['Champions'] == champ_l][['HP', 
                                                                'MP', 
                                                                'AD', 
                                                                'AS', 
                                                                'AR', 
                                                                'MR', 
                                                                'MS', 
                                                                'Range']].values.tolist()[0]

        line = []
        if i % 2 == 0:
            line += [won_t, lost_t] + won_l + lost_l + [1]
        else:
            line += [lost_t, won_t] + lost_l + won_l + [0]

        spamwriter.writerow(line)
