import pandas as pd

sheet_id = "1qgYEIdqjF1ey0_BwU8rj9XHo3XpFLWbt5UtiTHR9mqs"
sheet_name = "data"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

df = pd.read_csv(url)

WEBSITE_LIST = df[['Nom', 'Adresse', 'Cookie selector', 'Link selector', 'Content selector', 'Title selector', 'Date selector', 'Author selector', 'Page url complement', 'Number of pages', 'paginator formula']].values.tolist()
INITIAL_WEBSITE_LIST = df.loc[df['Initial crawling']==  'x'][['Nom', 'Adresse', 'Cookie selector', 'Link selector', 'Content selector', 'Title selector', 'Date selector', 'Author selector', 'Page url complement', 'Number of pages', 'paginator formula']].values.tolist()