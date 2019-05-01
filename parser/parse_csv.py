import pandas as pd
import numpy as np

def preprocess_summary_file(dt, labels, dist_threshold, get_last=True):
    def process_solve_status(s):
        if s == "Infeasible" or s == "Unbounded" or s == "InfeasibleOrUnbounded":
            return "ProvablyRobustByClass"
        elif s == "InfeasibleDistance" or (dist_threshold == None and
                (s.startswith("InfeasibleDistance") or
                s.startswith("InfeasibleUndecidedDistance"))):
            return "ProvablyRobustByDistanceUnknownThreshold"
        elif s.startswith("InfeasibleDistance") and dist_threshold != None:
            min_threshold = float(s[18:])
            if min_threshold > dist_threshold:
                return "ProvablyRobustByDistance > " + str(dist_threshold)
            else:
                return "ProvablyRobustByDistance <= " + str(dist_threshold)
        elif s.startswith("InfeasibleUndecidedDistance") and dist_threshold != None:
            min_threshold = float(s[27:])
            if min_threshold > dist_threshold:
                return "ProvablyRobustByDistance > " + str(dist_threshold)
            else:
                return "ProvablyRobustByDistance <= " + str(dist_threshold)
        elif s == "UserLimit":
            return "StatusUnknown"
        else:
            return "Vulnerable"

    dt.SampleNumber -=1
    dt.PredictedIndex -=1
    dt = dt.drop_duplicates(
        subset = "SampleNumber", keep= "last" if get_last else "first"
    ).set_index("SampleNumber").sort_index().join(labels)
    dt["IsCorrectlyPredicted"] = dt.PredictedIndex == dt.TrueIndex
    dt["ProcessedSolveStatus"] = dt["SolveStatus"].apply(process_solve_status)
    dt["BuildTime"] = dt["TotalTime"] - dt["SolveTime"]
    return dt

def get_dt(filename, labels, dist_threshold=None):
    dt = pd.read_csv(filename)
    return preprocess_summary_file(dt, labels, dist_threshold)

def summarize_processed_solve_status(filename, labels, dist_threshold=None):
    dt = get_dt(filename, labels, dist_threshold)
    return dt.groupby("ProcessedSolveStatus").TotalTime.count().rename(filename).transpose()

def summarize_time(filename, labels, agg_by="mean", correct_only=True, exclude_timeouts=False, exclude_skipped_natural_incorrect=True):
    dt = get_dt(filename, labels)
    if correct_only:
        dt = dt[dt.IsCorrectlyPredicted]
    if exclude_timeouts:
        dt = dt[dt["ProcessedSolveStatus"]!="StatusUnknown"]
    if exclude_skipped_natural_incorrect:
        dt = dt[dt["SolveTime"]!=0]
    return dt[["BuildTime", "SolveTime", "TotalTime"]].agg(agg_by).rename(filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pass in csv to parse")
    parser.add_argument("csv_name", type=str)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--valid", dest="is_valid", action="store_true")
    group.add_argument("--test", dest="is_valid", action="store_false")
    parser.add_argument("--dist_threshold", dest="dist_threshold", type=float, default=None)
    config = parser.parse_args()

    dist_threshold = config.dist_threshold

    if config.is_valid:
        labels = pd.read_csv("parser/mnist_valid_labels.csv").set_index("SampleNumber")
    else:
        labels = pd.read_csv("parser/mnist_test_labels.csv").set_index("SampleNumber")

    # Main code
    print(summarize_processed_solve_status(config.csv_name, labels, config.dist_threshold))
    print(summarize_time(config.csv_name, labels))
