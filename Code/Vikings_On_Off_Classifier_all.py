#!/usr/bin/env python
# coding: utf-8
import os, sys
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

random_seed = int(sys.argv[1])
np.random.seed(random_seed)
verbose_lvoo_sp = False
train_type = sys.argv[2]  # One of "problem", "instance", "leave_one_out", "single_problem"
type_scorer = sys.argv[3]  # One of "f1", "custom"
scorer_sub = sys.argv[4]  # One of "True", "False"
split_t_t = float(sys.argv[5])  # Train-test split, test size
solver = sys.argv[6]  # "minion", "kissat", "kissat_mdd", "chuffed"

sarofe_dir = "../Dataset/on-off-features-instances"
files = os.listdir(sarofe_dir)

# Loading results
result_df = pd.read_csv("../Dataset/results_on_off_"+solver+".csv")
result_df.drop_duplicates(["Problem", "Policy"], keep="first", inplace=True)
max_time = result_df['SolverTotalTime'].max() + result_df['SavileRowTotalTime'].max()

targets = dict()
tot_time_without_tab = 0.0
tot_time_with_tab = 0.0
for prob in result_df["Problem"].unique():
    baseline = result_df.query("Problem=='"+prob+"' and Policy=='baseline'")
    if len(baseline) == 0:
        continue
    baseline_time = baseline['SolverTotalTime'].values[0] + baseline['SavileRowTotalTime'].values[0]
    tab2 = result_df.query("Problem=='"+prob+"' and Policy=='2'")
    if len(tab2) == 0:
        continue
    tab2_time = tab2['SolverTotalTime'].values[0] + tab2['SavileRowTotalTime'].values[0]
    baselineTimeOut = baseline["SolverTimeOut"].fillna(1).values+baseline["SavileRowTimeOut"].fillna(1).values
    tab2TimeOut = tab2["SolverTimeOut"].fillna(1).values+tab2["SavileRowTimeOut"].fillna(1).values
    if baselineTimeOut>0 and tab2TimeOut>0:
        continue  # drop row
    if baselineTimeOut>0 and tab2TimeOut==0:
        clipped = np.clip(tab2_time, 0, 3600)
        targets[prob] = {'Target': 1, 'Score': 3600-clipped, 'Tab_t': clipped, 'Base_t': 3600}
    elif baselineTimeOut==0 and tab2TimeOut>0:
        clipped = np.clip(baseline_time, 0, 3600)
        targets[prob] = {'Target': 0, 'Score': clipped-3600, 'Tab_t': 3600, 'Base_t': clipped}
    elif baselineTimeOut==0 and tab2TimeOut==0 and np.abs(tab2_time-baseline_time)<1:
        continue # Skip small differences
    elif baselineTimeOut==0 and tab2TimeOut==0:
        clipped_1 = np.clip(tab2_time, 0, 3600)
        clipped_2 = np.clip(baseline_time, 0, 3600)
        label = 1 if tab2_time < baseline_time else 0
        targets[prob] = {'Target': label, 'Score': clipped_2-clipped_1, 'Tab_t': clipped_1, 'Base_t': clipped_2}
target_df = pd.DataFrame.from_dict(targets, orient ='index')
target_df['Problem'] = target_df.index

files_regex = re.compile(".*.sarofe.csv$")
obj_regex = re.compile("[s|o]_*")
files_to_use = [x for x in files if files_regex.match(x)]
cont = 0
while os.path.getsize(os.path.join(sarofe_dir, files_to_use[cont])) < 66:
    cont += 1
df = pd.read_csv(os.path.join(sarofe_dir, files_to_use[cont]))
dict_float32 = dict()
for feat in [x for x in df.columns if "SavileRowTotalTime:" not in x and not obj_regex.match(x)]:
    dict_float32[feat] = "float32"
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Necessary for older versions of pandas
df = df.astype(dict_float32)
df["Problem"] = files_to_use[cont].split(".")[0]
for i in range(cont+1, len(files_to_use)):
    dir_file = os.path.join(sarofe_dir, files_to_use[i])
    if os.path.getsize(dir_file) >= 66:
        tmp_df = pd.read_csv(dir_file)
        tmp_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Necessary for older versions of pandas
        tmp_df = tmp_df.astype(dict_float32)
        name_prob = files_to_use[i].split(".")[0]
        tmp_df["Problem"] = name_prob
        df = df.append(tmp_df)
if 'SavileRowClauseOut:0' in df.columns:
    df = df.drop(['SavileRowClauseOut:0'], axis=1)
df.drop_duplicates(["Problem"], keep="first", inplace=True)
df['c_std_tightness'].fillna(0, inplace=True)
df['ratio_75_overlap'].fillna(0, inplace=True)
df['c_std_dup_var'].fillna(0, inplace=True)
df['c_std_size_tree'].fillna(0, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(axis=1)
df = df.drop(df.columns[df.nunique() <= 1], axis=1)
df = df.merge(target_df[['Problem', 'Target', 'Score', 'Tab_t', 'Base_t']], on = 'Problem')
time_features = [x for x in df.columns if "time" in x]
df["Time"] = df[time_features].sum(axis=1) / 1000
df = df.drop(time_features, axis=1)
df = df.reset_index()

only_test_probs = ["sendMoreMoney", "tickTackToe", "tomsProblem", "n-queens", "n-queens2", "diet", "farm-puzzle1", "farm-puzzle2", "grocery", "magicSquare"]

class generator_leave_one_out:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.problems = list(self.X['Problem'].apply(lambda x: x.split("_")[0]).unique())
        self.train_scores = 0.0
        self.train_tot_time_base = 0.0
        self.train_tot_time_tab = 0.0
        self.train_virtual_best = 0.0
        self.test_scores = 0.0
        self.test_tot_time_base = 0.0
        self.test_tot_time_tab = 0.0
        self.test_virtual_best = 0.0
    
    def sample(self, problem_id):
        x_train = pd.DataFrame()
        y_train = pd.Series()
        prob = self.problems[problem_id]
        labels = X.Problem.str.match(prob+"_[a-zA-Z_0-9]")
        x_test = X[labels]
        y_test = y[labels]
        problems = [x for x in self.problems if x != self.problems[problem_id]]
        for prob in problems:
            labels = X.Problem.str.match(prob+"_[a-zA-Z_0-9]")
            x_train = pd.concat([x_train, X[labels]])
            y_train = pd.concat([y_train, y[labels]])
        
        self.train_scores = x_train.Score
        self.train_times = x_train.Time
        self.train_tot_time_base = np.sum(x_train.Base_t)
        self.train_tot_time_tab = np.sum(x_train.Tab_t)
        self.train_virtual_best = sum(np.array([1 if x=='Tab' else 0 for x in y_train])*np.array(self.train_scores))
        instances_train = x_train["Problem"]
        x_train = x_train.drop(["Score", "Base_t", "Tab_t", 'Problem', "Time"], axis=1)

        self.test_scores = x_test.Score
        self.test_times = x_test.Time
        self.test_tot_time_base = np.sum(x_test.Base_t)
        self.test_tot_time_tab = np.sum(x_test.Tab_t)
        self.test_virtual_best = sum(np.array([1 if x=='Tab' else 0 for x in y_test])*np.array(self.test_scores))
        instances_test = x_test["Problem"]
        x_test = x_test.drop(["Score", "Base_t", "Tab_t", 'Problem', "Time"], axis=1)
        return x_train, y_train, x_test, y_test, instances_train, instances_test
    

class generator_single_problem:
    def __init__(self, X, y, test_size):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.problems = list(self.X['Problem'].apply(lambda x: x.split("_")[0]).unique())
        self.problems = [x for x in self.problems if X.Problem.str.match(x+"_[a-zA-Z_0-9]").sum()>=10 and x not in only_test_probs]
        self.train_scores = 0.0
        self.train_tot_time_base = 0.0
        self.train_tot_time_tab = 0.0
        self.train_virtual_best = 0.0
        self.test_scores = 0.0
        self.test_tot_time_base = 0.0
        self.test_tot_time_tab = 0.0
        self.test_virtual_best = 0.0
    
    def sample(self, problem_id):
        prob = self.problems[problem_id]
        labels = X.Problem.str.match(prob+"_[a-zA-Z_0-9]")
        x_prob = X[labels]
        y_prob = y[labels]
        x_train, x_test, y_train, y_test = train_test_split(x_prob, y_prob, 
                                                            test_size=self.test_size, 
                                                            random_state=random_seed)
        
        self.train_scores = x_train.Score
        self.train_times = x_train.Time
        self.train_tot_time_base = np.sum(x_train.Base_t)
        self.train_tot_time_tab = np.sum(x_train.Tab_t)
        self.train_virtual_best = sum(np.array([1 if x=='Tab' else 0 for x in y_train])*np.array(self.train_scores))
        instances_train = x_train["Problem"]
        x_train = x_train.drop(["Score", "Base_t", "Tab_t", 'Problem', "Time"], axis=1)

        self.test_scores = x_test.Score
        self.test_times = x_test.Time
        self.test_tot_time_base = np.sum(x_test.Base_t)
        self.test_tot_time_tab = np.sum(x_test.Tab_t)
        self.test_virtual_best = sum(np.array([1 if x=='Tab' else 0 for x in y_test])*np.array(self.test_scores))
        instances_test = x_test["Problem"]
        x_test = x_test.drop(["Score", "Base_t", "Tab_t", 'Problem', "Time"], axis=1)
        return x_train, y_train, x_test, y_test, instances_train, instances_test

X = df.drop(["Target", "index"], axis=1)
y = df.Target.apply(lambda x: "Tab" if x==1 else "No Tab")
feature_names = [x for x in df.drop(["Target", "Problem", "index", "Score", 'Tab_t', 'Base_t', "Time"], axis=1).columns]

if train_type == "problem":
    problem_list = [x for x in df['Problem'].apply(lambda x: x.split("_")[0]).unique() if x not in only_test_probs]
    np.random.shuffle(problem_list)
    test_problems = copy.deepcopy(only_test_probs)
    test_problems = test_problems + problem_list[int(-split_t_t*len(problem_list)):]
    train_problems = problem_list[:int(-split_t_t*len(problem_list))]
    X_train = pd.DataFrame()
    y_train = pd.Series()
    for prob in train_problems:
        labels = X.Problem.str.match(prob+"_[a-zA-Z_0-9]")
        X_train = pd.concat([X_train, X[labels]])
        y_train = pd.concat([y_train, y[labels]])
    X_test = pd.DataFrame()
    y_test = pd.Series()
    for prob in test_problems:
        labels = X.Problem.str.match(prob+"_[a-zA-Z_0-9]")
        X_test = pd.concat([X_test, X[labels]])
        y_test = pd.concat([y_test, y[labels]])
elif train_type == "instance":
    problem_list = [x for x in df['Problem'].apply(lambda x: x.split("_")[0]).unique() if x not in only_test_probs]
    X_both = pd.DataFrame()
    y_both = pd.Series()
    for prob in problem_list:
        labels = X.Problem.str.match(prob+"_[a-zA-Z_0-9]")
        X_both = pd.concat([X_both, X[labels]])
        y_both = pd.concat([y_both, y[labels]])
    X_test_only = pd.DataFrame()
    y_test_only = pd.Series()
    for prob in only_test_probs:
        labels = X.Problem.str.match(prob+"_[a-zA-Z_0-9]")
        X_test_only = pd.concat([X_test_only, X[labels]])
        y_test_only = pd.concat([y_test_only, y[labels]])
    X_train, X_test, y_train, y_test = train_test_split(
        X_both, y_both, test_size=split_t_t, random_state=random_seed)
    X_test = pd.concat([X_test, X_test_only])
    y_test = pd.concat([y_test, y_test_only])
elif train_type == "leave_one_out":
    gen_leave_oo = generator_leave_one_out(X, y)
elif train_type == "single_problem":
    gen_leave_oo = generator_single_problem(X, y, test_size=split_t_t)

if train_type != "leave_one_out" and train_type != "single_problem":
    train_problems = X_train.Problem.unique()
    test_problems = X_test.Problem.unique()
    X = X.drop(['Score', 'Problem'], axis=1)

    train_scores = X_train.Score
    train_times = X_train.Time
    train_tab_t = X_train.Tab_t
    train_tot_time_base = np.sum(X_train.Base_t)
    train_tot_time_tab = np.sum(X_train.Tab_t)
    train_virtual_best = sum(np.array([1 if x=='Tab' else 0 for x in y_train])*np.array(train_scores))

    test_scores = X_test.Score
    test_times = X_test.Time
    test_tab_t = X_test.Tab_t
    test_tot_time_base = np.sum(X_test.Base_t)
    test_tot_time_tab = np.sum(X_test.Tab_t)
    test_virtual_best = sum(np.array([1 if x=='Tab' else 0 for x in y_test])*np.array(test_scores))

def custom_scorer(y_true, y_pred, scores):
    curr_scores = scores[y_true.index]
    
    if scorer_sub == "True":
        result = np.clip((np.sum(y_pred*np.array(curr_scores))-np.sum(curr_scores)) / (np.sum(y_true*np.array(curr_scores))-np.sum(curr_scores)), -1, 1)
    else:
        result = np.clip(np.sum(y_pred*np.array(curr_scores)) / np.sum(y_true*np.array(curr_scores)), -1, 1)
    return result

# Leave-one-out or single_problem training
if train_type == "leave_one_out" or train_type == "single_problem":
    rfc_params = [{'max_depth': list(range(1, 6)), 'n_estimators': [10, 100, 200, 300, 400]}]
    mean_times = {"saved_v_best": [], "saved_s_best": [], "saved_class": [], "saved_heur": [], "tot_best": [], "tot_class": [], 
                  "tot_heur": [], "tot_no_tab": [], "Problem": [], "train_test": [], "prediction": []}
    mean_times_train = {"saved_v_best": [], "saved_s_best": [], "saved_class": [], "saved_heur": [], "tot_best": [], "tot_class": [], 
                  "tot_heur": [], "tot_no_tab": [], "Problem": [], "train_test": [], "prediction": []}
    for i in range(len(gen_leave_oo.problems)):
        X_train, y_train, X_test, y_test, instances_train, instances_test = gen_leave_oo.sample(i)
        train_scores = gen_leave_oo.train_scores
        train_times = gen_leave_oo.train_times
        train_tot_time_base = gen_leave_oo.train_tot_time_base
        train_tot_time_tab = gen_leave_oo.train_tot_time_tab
        train_virtual_best = gen_leave_oo.train_virtual_best
        test_scores = gen_leave_oo.test_scores
        test_times = gen_leave_oo.test_times
        test_tot_time_base = gen_leave_oo.test_tot_time_base
        test_tot_time_tab = gen_leave_oo.test_tot_time_tab
        test_virtual_best = gen_leave_oo.test_virtual_best
        if type_scorer=='custom':
            mk_scorer = make_scorer(custom_scorer, scores=train_scores)
        else:
            mk_scorer = make_scorer(f1_score, pos_label="Tab")
        clf = GridSearchCV(RandomForestClassifier(random_state=random_seed), 
                       param_grid=rfc_params, 
                       scoring=mk_scorer, cv=3,
                       return_train_score = False,
                       n_jobs = -1, refit=True)
        clf.fit(X_train, y_train.apply(lambda x: 1 if x=="Tab" else 0), sample_weight=[np.abs(x) for x in train_scores])

        y_true, y_pred = y_train.apply(lambda x: 1 if x=="Tab" else 0), clf.best_estimator_.predict(X_train)
        time_saved_cl = sum(y_pred*np.array(train_scores)) - sum(train_times)
        mean_times_train["saved_v_best"].append(train_virtual_best)
        mean_times_train["saved_class"].append(time_saved_cl)
        mean_times_train["saved_heur"].append(np.sum(train_scores))
        mean_times_train["saved_s_best"].append(mean_times_train["saved_heur"][-1] if mean_times_train["saved_heur"][-1]>0 else 0)
        mean_times_train["tot_best"].append(train_tot_time_base-train_virtual_best)
        mean_times_train["tot_class"].append(train_tot_time_base-time_saved_cl)
        mean_times_train["tot_heur"].append(train_tot_time_tab)
        mean_times_train["tot_no_tab"].append(train_tot_time_base)
        mean_times_train["prediction"].append(["pos_"+x if key==1 else "neg_"+x for (x, key) in zip(instances_train, y_pred)])
        mean_times_train["Problem"].append(gen_leave_oo.problems[i])
        mean_times_train["train_test"].append("train")

        y_true, y_pred = y_test.apply(lambda x: 1 if x=="Tab" else 0), clf.best_estimator_.predict(X_test)
        time_saved_cl = sum(y_pred*np.array(test_scores)) - sum(test_times)
        mean_times["saved_v_best"].append(test_virtual_best)
        mean_times["saved_class"].append(time_saved_cl)
        mean_times["saved_heur"].append(np.sum(test_scores))
        mean_times["saved_s_best"].append(mean_times["saved_heur"][-1] if mean_times_train["saved_heur"][-1]>0 else 0)
        mean_times["tot_best"].append(test_tot_time_base-test_virtual_best)
        mean_times["tot_class"].append(test_tot_time_base-time_saved_cl)
        mean_times["tot_heur"].append(test_tot_time_tab)
        mean_times["tot_no_tab"].append(test_tot_time_base)
        mean_times["prediction"].append(["pos_"+x if key==1 else "neg_"+x for (x, key) in zip(instances_test, y_pred)])
        mean_times["Problem"].append(gen_leave_oo.problems[i])
        mean_times["train_test"].append("test")
        
    print("Test")
    print("Time saved with virtual best classifier: %0.3f (+/- %0.03f) s" % (np.mean(mean_times["saved_v_best"]), np.std(mean_times["saved_v_best"])))
    print("Time saved with random forest: %0.3f (+/- %0.03f) s" % (np.mean(mean_times["saved_class"]), np.std(mean_times["saved_class"])))
    print("Time saved by always using the heuristics: %0.3f (+/- %0.03f) s" % (np.mean(mean_times["saved_heur"]), np.std(mean_times["saved_heur"])))
    print("\nTotal time with virtual best classifier: %0.3f (+/- %0.03f) s" % (np.mean(mean_times["tot_best"]), np.std(mean_times["tot_best"])))
    print("Total time with random forest: %0.3f (+/- %0.03f) s" % (np.mean(mean_times["tot_class"]), np.std(mean_times["tot_class"])))
    print("Total time by always using the heuristics: %0.3f (+/- %0.03f) s" % (np.mean(mean_times["tot_heur"]), np.std(mean_times["tot_heur"])))
    print("Total time without tabulation: %0.3f (+/- %0.03f) s" % (np.mean(mean_times["tot_no_tab"]), np.std(mean_times["tot_no_tab"])))
    print()

    print("Train")
    print("Time saved with virtual best classifier: %0.3f (+/- %0.03f) s" % (np.mean(mean_times_train["saved_v_best"]), np.std(mean_times_train["saved_v_best"])))
    print("Time saved with random forest: %0.3f (+/- %0.03f) s" % (np.mean(mean_times_train["saved_class"]), np.std(mean_times_train["saved_class"])))
    print("Time saved by always using the heuristics: %0.3f (+/- %0.03f) s" % (np.mean(mean_times_train["saved_heur"]), np.std(mean_times_train["saved_heur"])))
    print("\nTotal time with virtual best classifier: %0.3f (+/- %0.03f) s" % (np.mean(mean_times_train["tot_best"]), np.std(mean_times_train["tot_best"])))
    print("Total time with random forest: %0.3f (+/- %0.03f) s" % (np.mean(mean_times_train["tot_class"]), np.std(mean_times_train["tot_class"])))
    print("Total time by always using the heuristics: %0.3f (+/- %0.03f) s" % (np.mean(mean_times_train["tot_heur"]), np.std(mean_times_train["tot_heur"])))
    print("Total time without tabulation: %0.3f (+/- %0.03f) s" % (np.mean(mean_times_train["tot_no_tab"]), np.std(mean_times_train["tot_no_tab"])))
    print()
    
    # Save in pickle
    if False:
        filename = '/log_' + train_type + "_" + str(random_seed) + "_" + str(split_t_t) + ".pkl"
        with open("folder/"+solver+"/"+train_type+"_"+type_scorer+filename, "wb") as pkl_f:
            pickle.dump([mean_times, mean_times_train], pkl_f)
    
    # Exit
    sys.exit(0)

if type_scorer=='custom':
    mk_scorer = make_scorer(custom_scorer, scores=train_scores)
else:
    mk_scorer = make_scorer(f1_score, pos_label="Tab")

rfc_params = [{'max_depth': list(range(1, 6)), 'n_estimators': [10, 100, 200, 300, 400]}]
clf = GridSearchCV(RandomForestClassifier(random_state=random_seed), 
                   param_grid=rfc_params, 
                   scoring=mk_scorer, cv=3,
                   return_train_score = True,
                   n_jobs = -1)
clf.fit(X_train.drop(['Problem', 'Score', "Base_t", "Tab_t", "Time"], axis=1), y_train.apply(lambda x: 1 if x=="Tab" else 0), sample_weight=[np.abs(x) for x in train_scores])

max_depth_max = clf.best_params_['max_depth']
n_estimators = clf.best_params_['n_estimators']

rf = RandomForestClassifier(max_depth=max_depth_max, n_estimators=n_estimators, random_state=random_seed)
rf.fit(X_train.drop(['Problem', 'Score', "Base_t", "Tab_t", "Time"], axis=1), y_train.apply(lambda x: 1 if x=="Tab" else 0), sample_weight=[np.abs(x) for x in train_scores])
y_true, y_pred = y_test.apply(lambda x: 1 if x=="Tab" else 0), rf.predict(X_test.drop(['Problem', 'Score', "Base_t", "Tab_t", "Time"], axis=1))

all_test = {"saved_v_best": [], "saved_s_best": [], "saved_class": [], "saved_heur": [], "tot_best": [], "tot_class": [], 
            "tot_heur": [], "tot_no_tab": [], "train_test": [], "Problem": [], "prediction": []}
id_prob = 0
for prob in X_test['Problem']:
    tmp_prob = X_test.query("Problem=='"+prob+"'")
    tot_base_t = tmp_prob.Base_t.values[0]
    tot_tab_t = tmp_prob.Tab_t.values[0]
    score = tmp_prob.Score.values[0]
    time = tmp_prob.Time.values[0]
    v_best = 0 if tot_base_t<=tot_tab_t else score
    class_saved = score - time if y_pred[id_prob] == 1 else -time
    all_test["saved_v_best"].append(v_best)
    all_test["saved_class"].append(class_saved)
    all_test["saved_heur"].append(score)
    all_test["saved_s_best"].append(all_test["saved_heur"][-1] if X_train.Score.sum()>0 else 0)
    all_test["tot_best"].append(tot_base_t-v_best)
    all_test["tot_class"].append(tot_base_t-class_saved)
    all_test["tot_heur"].append(tot_tab_t)
    all_test["tot_no_tab"].append(tot_base_t)
    all_test["train_test"].append("test")
    all_test["Problem"].append(prob)
    all_test["prediction"].append("pos_"+prob if y_pred[id_prob] == 1 else "neg_"+prob)
    id_prob += 1

time_saved_cl = sum(y_pred*np.array(test_scores)) - sum(test_times)
print("Test")
print("Time saved with virtual best classifier: ", test_virtual_best, "s")
print("Time saved with random forest: ", time_saved_cl, "s")
print("Time saved by always using the heuristics: ", np.sum(test_scores), "s")
print("\nTotal time with virtual best classifier: ", test_tot_time_base-test_virtual_best, "s")
print("Total time with random forest: ", test_tot_time_base-time_saved_cl, "s")
print("Total time by always using the heuristics: ", test_tot_time_tab, "s")
print("Total time without tabulation: ", test_tot_time_base, "s")

y_true, y_pred = y_train.apply(lambda x: 1 if x=="Tab" else 0), rf.predict(X_train.drop(['Problem', 'Score', "Base_t", "Tab_t", "Time"], axis=1))
all_train = {"saved_v_best": [], "saved_s_best": [], "saved_class": [], "saved_heur": [], "tot_best": [], "tot_class": [], 
            "tot_heur": [], "tot_no_tab": [], "train_test": [], "Problem": [], "prediction": []}
id_prob = 0
for prob in X_train['Problem']:
    tmp_prob = X_train.query("Problem=='"+prob+"'")
    tot_base_t = tmp_prob.Base_t.values[0]
    tot_tab_t = tmp_prob.Tab_t.values[0]
    score = tmp_prob.Score.values[0]
    time = tmp_prob.Time.values[0]
    v_best = 0 if tot_base_t<=tot_tab_t else score
    class_saved = score - time if y_pred[id_prob] == 1 else -time
    all_train["saved_v_best"].append(v_best)
    all_train["saved_class"].append(class_saved)
    all_train["saved_heur"].append(score)
    all_train["saved_s_best"].append(all_train["saved_heur"][-1] if X_train.Score.sum()>0 else 0)
    all_train["tot_best"].append(tot_base_t-v_best)
    all_train["tot_class"].append(tot_base_t-class_saved)
    all_train["tot_heur"].append(tot_tab_t)
    all_train["tot_no_tab"].append(tot_base_t)
    all_train["train_test"].append("train")
    all_train["Problem"].append(prob)
    all_train["prediction"].append("pos_"+prob if y_pred[id_prob] == 1 else "neg_"+prob)
    id_prob += 1
    
time_saved_cl = sum(y_pred*np.array(train_scores)) - sum(train_times)
print("Train")
print("Time saved with virtual best classifier: ", train_virtual_best, "s")
print("Time saved with random forest: ", time_saved_cl, "s")
print("Time saved by always using the heuristics: ", np.sum(train_scores), "s")
print("\nTotal time with virtual best classifier: ", train_tot_time_base-train_virtual_best, "s")
print("Total time with random forest: ", train_tot_time_base-time_saved_cl, "s")
print("Total time by always using the heuristics: ", train_tot_time_tab, "s")
print("Total time without tabulation: ", train_tot_time_base, "s")

# Save in pickle
if False:
    filename = '/log_' + train_type + "_" + str(random_seed) + "_" + str(split_t_t) + ".pkl"
    with open("folder/"+solver+"/"+train_type+"_"+type_scorer+filename, "wb") as pkl_f:
        pickle.dump([test_problems, train_problems, all_test, all_train], pkl_f)

