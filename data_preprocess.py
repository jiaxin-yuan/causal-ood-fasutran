"""
This module takes care of constructing prefix and suffix dataframes,
given a log (as a pandas dataframe). It first encodes categoricals to integers, takes care of missing
values, constructs the prefix dataframe, suffix dataframe, and
suffix dataframes pertaining to the labels. Inspired by paper""
"""

from pm4py.objects.conversion.log import converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py
import argparse
import pandas as pd

case_id = "case:concept:name"
timestamp = "time:timestamp"
act_label = "concept:name"

class DataPreprocessor:
    def __init__(self, name, output_dir):
        self.name = name
        self.output_dir = output_dir



    def preprocess(self):
        if self.name == "bpi15":
            path = f"ori_data/{self.name}"
            
            cat_casefts = ['case:Responsible_actor']
            cat_eventfts = ["monitoringResource", "org:resource"]
            num_casefts = ['case:SUMleges', "domain"]
            num_eventfts = []

            for munic_nr in range(1, 6):
                df = pm4py.read_xes(path + "/" + self.name + "_" + str(munic_nr) + ".xes")
                df["domain"] = munic_nr
                if munic_nr == 1:
                    dataset = df
                else:
                    dataset = pd.concat([dataset, df])

        if self.name == "bpi19":
            path = f"ori_data/{self.name}"
            cat_casefts = [ 'case:Spend area text', 'case:Company', 'case:Document Type', 
                            'case:Sub spend area text', 'case:Item',
                            'case:Vendor', 'case:Item Type', 
                            'case:Item Category', 'case:Spend classification text', 
                            'case:GR-Based Inv. Verif.', 'case:Goods Receipt']
            cat_eventfts = ['org:resource']
            num_casefts = []
            num_eventfts = ['Cumulative net worth (EUR)']
            df = pm4py.read_xes(path + "/" + self.name)
            print()

        # dataset = converter.apply(dataset, variant=converter.Variants.TO_EVENT_LOG)
        # time-related attributes generation
        dataset = self.create_numeric_timeCols(dataset)
        num_eventfts.extend(['tt_next', 'ts_prev', 'ts_start', 'rtime'])
        cat_cols = cat_casefts + cat_eventfts
        num_cols = num_casefts + num_eventfts
        needed_cols = [case_id, act_label, timestamp] + cat_cols + num_cols
        dataset = dataset[needed_cols]
        print("dataset converted to csv")
        dataset.to_csv(self.output_dir + "/" + self.name + ".csv", index=False)



    def sort_log(self, df, case_id = 'case:concept:name', timestamp = 'time:timestamp', act_label = 'concept:name'):
        """Sort events in event log such that cases that occur first are stored
        first, and such that events within the same case are stored based on timestamp.

        Parameters
        ----------
        df: pd.DataFrame
            Event log to be preprocessed.
        case_id : str, optional
            Column name of column containing case IDs. By default
            'case:concept:name'.
        timestamp : str, optional
            Column name of column containing timestamps. Column Should be of
            the datetime64 dtype. By default 'time:timestamp'.
        act_label : str, optional
            Column name of column containing activity labels. By default
            'concept:name'.
        """
        df_help = df.sort_values([case_id, timestamp], ascending = [True, True], kind='mergesort').copy()
        # Now take first row of every case_id: this contains first stamp
        df_first = df_help.drop_duplicates(subset = case_id)[[case_id, timestamp]].copy()
        df_first = df_first.sort_values(timestamp, ascending = True, kind='mergesort')
        # Include integer index to sort on.
        df_first['case_id_int'] = [i for i in range(len(df_first))]
        df_first = df_first.drop(timestamp, axis = 1)
        df = df.merge(df_first, on = case_id, how = 'left')
        df = df.sort_values(['case_id_int', timestamp], ascending = [True, True], kind='mergesort')
        df = df.drop('case_id_int', axis = 1)
        return df.reset_index(drop=True)

    def create_numeric_timeCols(self, df,
                                case_id='case:concept:name',
                                timestamp='time:timestamp',
                                act_label='concept:name'):
        """Adds the following columns to df:

        - 'case_length' : the number of events for each case. Constant for
          all events of the same case.
        - 'tt_next' : for each event, the time (in seconds) until the next
          event occurs. Zero for the last event of each case.
        - 'ts_prev' : for each event, time (in sec) from previous event.
          Zero for the first event of each case.
        - 'ts_start' : for each event, time (in sec) from the first event
          of that case. Zero for first event of each case.
        - 'rtime' : for each event, time (in sec) until last event of its
          case. Zero for last event of each case.

        Parameters
        ----------
        df: pd.DataFrame
            Event log to be preprocessed.
        case_id : str, optional
            Column name of column containing case IDs. By default
            'case:concept:name'.
        timestamp : str, optional
            Column name of column containing timestamps. Column Should be of
            the datetime64 dtype. By default 'time:timestamp'.
        act_label : str, optional
            Column name of column containing activity labels. By default
            'concept:name'.
        """

        # Compute for each case the case length ('num_events'). Needed for later steps.
        df['case_length'] = df.groupby(case_id, sort=False)[act_label].transform(len)

        # Sorting the cases and events (in case this did not happen yet)
        df = self.sort_log(df, case_id=case_id, timestamp=timestamp, act_label=act_label)

        # Create a df with only one event per case, with that single event solely containing the
        # case id and timestamp of the FIRST EVNET (so start time) of every case.
        case_df = df.drop_duplicates(subset=case_id).copy()
        case_df = case_df[[case_id, timestamp]]
        case_df.columns = [case_id, 'first_stamp']

        # Create a df with only one event per case, with that single event solely containing the
        # case id and timestamp of the LAST EVENT (so end time) of every case.
        last_stamp_df = df[[case_id, timestamp]].groupby(case_id, sort=False).last().reset_index()
        last_stamp_df.columns = [case_id, 'last_stamp']

        # Adding the case-constant 'last_stamp' and 'first_stamp' column to the train or test df.
        df = df.merge(last_stamp_df, on=case_id, how='left')  # adding 'last_stamp' "case feature"
        df = df.merge(case_df, on=case_id, how='left')  # adding 'first_stamp' "case feature"

        # Creating the 'next_stamp' column, which contains, for each event of the train or test df,
        # the timestamp of the subsequent event. Needed for computing the ttne target column for each
        # event. (For each last event, a NaN value is provided and that will later be filled with O.)
        df['next_stamp'] = df.groupby([case_id], sort=False)[timestamp].shift(-1)

        # Time till next event (in seconds)
        df['tt_next'] = (df['next_stamp'] - df[timestamp]) / pd.Timedelta(seconds=1)

        # Exactly same thing as with 'next_stamp', but then 'previous_stamp'. Hence every first event
        # of every case wil also first get a NaN value, and will later be assigned a 0. In contrast to
        # the 'next_stamp' column, this 'previous_stamp' column is not needed for computing a prediction
        # (time label) target, but for the suffixes and prefixes I believe.
        df['previous_stamp'] = df.groupby([case_id])[timestamp].shift(1)

        # Time since previous event (in seconds)
        df['ts_prev'] = (df[timestamp] - df['previous_stamp']) / pd.Timedelta(seconds=1)

        # Time since start case (in seconds)
        df['ts_start'] = (df[timestamp] - df['first_stamp']) / pd.Timedelta(seconds=1)

        # Remaining runtime (in seconds)
        # df['rtime'] = (df['last_stamp'] - df[timestamp]) / pd.Timedelta(seconds=1)

        df.drop(['next_stamp', 'previous_stamp', 'first_stamp', 'last_stamp'], axis=1, inplace=True)
        # Filling the NaN's of 'ts_prev' (first event of each case) and 'tt_next' (last event of each case)
        # correctly with 0.
        values = {'ts_prev': 0, 'tt_next': 0}
        df = df.fillna(value=values)

        return df

    def cat_mapping(self):
        # Dictionary for retrieving the final cardinalities for each categorical,
        # including potential missing values and OOV tokens.
        cardinality_dict = {}
        categorical_mapping_dict = {}
        for cat_col in tqdm(cat_cols_ext):
            cat_to_int = {}
            uni_level_train = list(train_df[cat_col].unique())
            uni_level_test = list(test_df[cat_col].unique())
            if cat_col in missing_value_catcols:
                if 'MISSINGVL' in uni_level_train:
                    uni_level_train.remove('MISSINGVL')
                if 'MISSINGVL' in uni_level_test:
                    uni_level_test.remove('MISSINGVL')
                int_mapping = [i for i in range(1, len(uni_level_train) + 1)]
                cat_to_int = dict(zip(uni_level_train, int_mapping))
                # Zero for missing values (if any)
                cat_to_int['MISSINGVL'] = 0
                # Every level occurring in test but not train should be
                # mapped to the same out of value token. (Last level)
                unseen_index = len(int_mapping) + 1
            else:
                # If no missing values for that categorical, no MV level of 0 created.
                int_mapping = [i for i in range(len(uni_level_train))]
                cat_to_int = dict(zip(uni_level_train, int_mapping))
                # Every level occurring in test but not train should be
                # mapped to the same out of value token. (Last level)
                unseen_index = len(int_mapping)

            for test_level in uni_level_test:
                if test_level not in uni_level_train:
                    cat_to_int[test_level] = unseen_index
            train_df[cat_col] = train_df[cat_col].map(cat_to_int)
            test_df[cat_col] = test_df[cat_col].map(cat_to_int)

            train_df[cat_col] = train_df[cat_col].astype('int')
            test_df[cat_col] = test_df[cat_col].astype('int')

            # Storing the final cardinalities for each categorical.
            cardinality_dict[cat_col] = len(set(cat_to_int.values()))
            categorical_mapping_dict[cat_col] = cat_to_int

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="输入log名称", choices=["bpi15"], default="bpi15")
    parser.add_argument("--output", type=str, help="输出log路径", default="processed_data")
    args = parser.parse_args()
    preprocessor = DataPreprocessor(args.name, args.output)
    preprocessor.preprocess()

if __name__ == "__main__":
    main()