import requests
import csv

from bs4 import BeautifulSoup

############## Game stats #################

page = requests.get('https://champion.gg/statistics/')

soup = BeautifulSoup(page.text, 'html.parser')

scripts = soup.find_all('script')

data = None
for s in scripts:
    if s.contents:
        if 'matchupData.stats' in s.contents[0]:
            data = str(s)

initial_pos = data.index('[')
end_pos = data.index(';')
data = data.replace(':null', ':None')

champions = eval(data[initial_pos: end_pos])

header = ['champion', 'role']
header += list(champions[0]['general'].keys())

# write csv
with open('champ_game_stats.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(header)
    for champ in champions: 
        data = [champ['title'], champ['role']] + [elem if elem else 0 for elem in champ['general'].values()]
        spamwriter.writerow(data)

#################### Base stats ###################

page = requests.get('https://leagueoflegends.fandom.com/wiki/List_of_champions/Base_statistics')

soup = BeautifulSoup(page.text, 'html.parser')

data = []
table = soup.find('table')
table_body = table.find('tbody')

rows = table_body.find_all('th')
t_header = [row.text.strip() for row in rows]

rows = table_body.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    cols = [c.replace('%', '') for c in cols]
    if cols:  # Get rid of empty values
        data.append(cols)

with open('champ_base_stats.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(t_header)
    for champ in data: 
        spamwriter.writerow(champ)
