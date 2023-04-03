'''
Take the exported misinformation tweets and their assigned topic clusters, output a new file in the same format
but with tweets re-sorted and those not meeting requirements deleted
'''
import csv

import pandas as pd
from datetime import datetime

def filter_and_sort(infile, out_file, col_timestamp, col_prob, col_topic_label, minprob=0.5):
    df = pd.read_csv(infile, header=0, delimiter=',', quoting=0, encoding="utf-8")
    header=list(df.columns)

    current_month=None
    current_day_data=[]
    add_header=True
    for index, row in df.iterrows():
        prob = row[col_prob]
        if prob<minprob:
            continue
        datestr = row[col_timestamp]
        date = datetime.strptime(datestr, '%d/%m/%Y %H:%M')
        row[col_timestamp]=date
        datestr = datestr.split(" ")[0].strip()
        datestr = datestr.split("/")
        datestr = datestr[1]+"/"+datestr[2]
        if current_month is None:
            current_month=datestr
            current_day_data.append(list(row))
        elif current_month!=datestr:
            #sort and output
            new_df=pd.DataFrame(columns=header)
            new_df = new_df.append(pd.DataFrame(current_day_data,
                                        columns=header),
                           ignore_index=True)
            new_df = new_df.sort_values([col_topic_label, col_timestamp],
                                 ascending=[True, True])
            new_df.to_csv(out_file, mode='a', sep=',', encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL, index=False, header=add_header)
            add_header=False
            current_day_data.clear()
            current_month=datestr
        else:
            current_day_data.append(list(row))

    if len(current_day_data)!=0:
        new_df = pd.DataFrame(columns=header)
        new_df = new_df.append(pd.DataFrame(current_day_data,
                                            columns=header),
                               ignore_index=True)
        new_df = new_df.sort_values([col_topic_label, col_timestamp],
                                    ascending=[True, True])
        new_df.to_csv(out_file, mode='a', sep=',', encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL, index=False,
                      header=False)

if __name__ == "__main__":
    filter_and_sort("/home/zz/Data/tbcov/topics/fake_non_retweets_top10_230301.csv",
                    "/home/zz/Data/tbcov/topics/fake_non_retweets_top10_230301_filtered+sorted.csv",
                    "created_at",
                    "prob",
                    "Name")

