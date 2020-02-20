import pandas as pd
google = pd.read_csv("GOOG-year.csv")
close = google.Predicted.values.tolist()
print(len(close))
close = [x for x in close if str(x) != 'nan']
print(len(close))