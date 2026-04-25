"""
Joseph Winterlich
projectCode.py
"""

# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


def main():
    """
    Thought process
    - read in data (from .txt file, in a folder so I need to use the right path)
    - only keep the features I need (including the runs scored as the "y" variable)
        - check retrosheet.org guide sheet for instructions on how to interpret the data (.txt file)
    - maybe split data for home team and away team, possibly then concatenating them

    - create each of the models
        - maybe create a working (simple) version of each before trying to finetune
    - then, build in performance metrics and comparisons (identify best, worst, etc.)
    """

    # read in data and save as df
    data = pd.read_csv("gl2020_25/gl2025.txt", header = None)
    # print(data.head())


    """
    columns to keep
    4 - visiting team
    7 - home team
    10 - visiting team score
    11 - home team score
    18 - attendance
    
    visiting team:
    22 - at-bats
    23 - hits
    24 - doubles
    25 - triples
    26 - homeruns
    27 - RBI
    31 - walks
    33 - strikeouts
    34 - stolen bases
    39 - number of pitchers used
    
    home team:
    50 - at-bats
    51 - hits
    52 - doubles
    53 - triples
    54 - homeruns
    55 - RBI
    59 - walks
    61 - strikeouts
    62 - stolen bases
    67 - number of pitchers used
    
    161 - do they have the full game info (Y=yes, N=no part of game, 
    D=derived from box score and game story, P=portions of game included)
    For above, probably only include "Y" games and make a note
    
    note: subtracted 1 from each to get indices due to 0-based indexing
    """


    # preprocessing (feature selection, handling missing values, tidying, combining data
    # from home and away teams)

    # indices of columns I want to keep
    # columns_to_keep = [4,7,10,11,18,22,23,24,25,26,27,31,33,34,39,50,51,52,53,54,55,59,61,62,67,161]
    columns_to_keep = [3,6,9,10,17,21,22,23,24,25,26,30,32,33,38,49,50,51,52,53,54,58,60,61,66,160]
    data = data[columns_to_keep]
    print("Number of samples:", len(data))

    # changes column indices to start at 0
    num_columns = len(columns_to_keep)
    # print(num_columns == len(data.columns))
    column_indices = range(num_columns)
    data.columns = column_indices
    # print indices of columns
    # print(data.columns)

    # switches the two columns with number of pitchers, so that it's number of pitchers
    # faced instead of number of pitchers used, because I think that makes more sense
    # to include it that way around
    data[14], data[24] = data[24], data[14]

    # changes column names to describe corresponding stats
    data.columns = ["visiting_team", "home_team", "visiting_team_score", "home_team_score",
                    "attendance", "v_at_bats", "v_hits", "v_doubles", "v_triples", "v_homeruns",
                    "v_rbi", "v_walks", "v_strikeouts", "v_stolen_bases", "v_pitchers_faced",
                    "h_at_bats", "h_hits", "h_doubles", "h_triples", "h_homeruns", "h_rbi",
                    "h_walks", "h_strikeouts", "h_stolen_bases", "h_pitchers_faced", "game_info_note"]
    # print(data.head())
    # print(len(data))

    # only keeps games for which the full info is included
    data = data[data["game_info_note"] == "Y"]
    # print(len(data))

    # test code - code works confirmed
    # print(data[["v_pitchers_faced", "h_pitchers_faced"]].head())


    # splits data into home and away teams. note: also encode home/away with is_home 1/0
    # also adds generic columns so I can drop the irrelevant away/home team stats later (i.e. not using
    # away team stats to predict home team runs)
    # home team
    data_home = data.copy()
    data_home["is_home"] = 1
    data_home["team"] = data["home_team"]
    data_home["runs_scored"] = data["home_team_score"]
    data_home["at_bats"] = data["h_at_bats"]
    data_home["hits"] = data["h_hits"]
    data_home["doubles"] = data["h_doubles"]
    data_home["triples"] = data["h_triples"]
    data_home["homeruns"] = data["h_homeruns"]
    data_home["rbi"] = data["h_rbi"]
    data_home["walks"] = data["h_walks"]
    data_home["strikeouts"] = data["h_strikeouts"]
    data_home["stolen_bases"] = data["h_stolen_bases"]
    data_home["pitchers_faced"] = data["h_pitchers_faced"]


    # visiting team
    data_visiting = data.copy()
    data_visiting["is_home"] = 0
    data_visiting["team"] = data["visiting_team"]
    data_visiting["runs_scored"] = data["visiting_team_score"]
    data_visiting["at_bats"] = data["v_at_bats"]
    data_visiting["hits"] = data["v_hits"]
    data_visiting["doubles"] = data["v_doubles"]
    data_visiting["triples"] = data["v_triples"]
    data_visiting["homeruns"] = data["v_homeruns"]
    data_visiting["rbi"] = data["v_rbi"]
    data_visiting["walks"] = data["v_walks"]
    data_visiting["strikeouts"] = data["v_strikeouts"]
    data_visiting["stolen_bases"] = data["v_stolen_bases"]
    data_visiting["pitchers_faced"] = data["v_pitchers_faced"]


    # some columns are then somewhat useless, so I chose to drop them
    columns_to_drop = ["visiting_team", "home_team", "visiting_team_score", "home_team_score"]
    data_home = data_home.drop(columns_to_drop, axis = 1)
    data_visiting = data_visiting.drop(columns_to_drop, axis = 1)


    # combine home and away data by stacking them
    data_all = pd.concat([data_home, data_visiting], ignore_index = True)
    # note: I think I avoid data leakage because shared info is dropped except for attendance

    # print("Shape before dropping home/away specific columns:", data_all.shape)
    # now, drop home/away specific columns
    columns_to_drop = ["v_at_bats", "v_hits", "v_doubles", "v_triples", "v_homeruns", "v_rbi", "v_walks",
                       "v_strikeouts", "v_stolen_bases", "v_pitchers_faced", "h_at_bats", "h_hits",
                       "h_doubles", "h_triples", "h_homeruns", "h_rbi", "h_walks", "h_strikeouts",
                       "h_stolen_bases", "h_pitchers_faced"]
    data_all = data_all.drop(columns_to_drop, axis = 1)
    # print("Shape after dropping home/away specific columns:", data_all.shape)

    # print("Original columns of combined dataframe:")
    # print(data_all.columns)

    # move team name column to 0th index and is_home to 1st index
    all_columns = data_all.columns.tolist()
    all_columns.insert(0, all_columns.pop(all_columns.index("team")))
    all_columns.insert(1, all_columns.pop(all_columns.index("is_home")))
    data_all = data_all[all_columns]

    # confirming columns of the combined dataframe
    # print("Columns of combined dataframe:")
    # print(data_all.columns)

    # checks dataset for missing values and removes any
    # note: removed two rows that had missing attendance values
    # print("Shape before dropping:", data_all.shape)
    # print("Number of NaN's per column: \n", data_all.isna().sum())
    data_all = data_all.dropna()
    # print("Shape after dropping:", data_all.shape)


    # split features and target variable (runs scored), also drops informational columns
    X = data_all.drop(columns = ["team", "game_info_note", "runs_scored"]).values
    y = data_all["runs_scored"].values

    # confirms no missing values in data
    # print("Number of NaN's per column: \n", np.isnan(X).sum(axis=0))


    # use histogram to determine the distribution of runs_scored
    plt.hist(y, bins=20, color="darkblue")
    plt.xlabel("Runs Scored")
    plt.ylabel("Frequencies")
    plt.title("Distribution of Runs Scored")
    plt.savefig("runs_scored.png")
    plt.show()

    # converts y back to df to use describe() for summary stats
    y_df = pd.Series(y)
    print("Summary stats for runs scored:")
    print(y_df.describe())


    # split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


    # Model 1 - simple linear regression
    print()
    print("Linear regression model")
    linreg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg_model", LinearRegression()),
    ])

    linreg_pipeline.fit(X_train, y_train)
    linreg_predictions = linreg_pipeline.predict(X_test)


    # evaluation of linreg model
    # splitting evaluation by training and testing
    linreg_mae_train = mean_absolute_error(y_train, linreg_pipeline.predict(X_train))
    linreg_mse_train = mean_squared_error(y_train, linreg_pipeline.predict(X_train))
    linreg_r2_score_train = r2_score(y_train, linreg_pipeline.predict(X_train))

    linreg_mae_test = mean_absolute_error(y_test, linreg_predictions)
    linreg_mse_test = mean_squared_error(y_test, linreg_predictions)
    linreg_r2_score_test = r2_score(y_test, linreg_predictions)

    print("linreg training mae:", linreg_mae_train)
    print("linreg testing mae:", linreg_mae_test)
    print("linreg training mse:", linreg_mse_train)
    print("linreg testing mse:", linreg_mse_test)
    print("linreg training r2_score:", linreg_r2_score_train)
    print("linreg testing r2_score:", linreg_r2_score_test)




    # model 2 - Lasso regularisation (because linear regression may be susceptible to
    # errors if features are not independent (which they're probably not in this case))

    # use lasso cv (to find optimal alpha), testing with several powers of 10
    print()
    print("Lasso cv model to find optimal alpha value")

    lasso_cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso_cv_model", LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], cv=5))
    ])

    lasso_cv_pipeline.fit(X_train, y_train)
    print("Optimal alpha value:", lasso_cv_pipeline.named_steps["lasso_cv_model"].alpha_)
    # note: determines alpha=0.001 to be optimal value, so I changed the value for model 1.2 below



    print()
    print("Lasso regularisation model")

    lasso_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso_model", Lasso(alpha = 0.001))
    ])

    lasso_pipeline.fit(X_train, y_train)
    lasso_predictions = lasso_pipeline.predict(X_test)


    # splitting evaluation by training and testing
    lasso_mae_train = mean_absolute_error(y_train, lasso_pipeline.predict(X_train))
    lasso_mse_train = mean_squared_error(y_train, lasso_pipeline.predict(X_train))
    lasso_r2_score_train = r2_score(y_train, lasso_pipeline.predict(X_train))

    lasso_mae_test = mean_absolute_error(y_test, lasso_predictions)
    lasso_mse_test = mean_squared_error(y_test, lasso_predictions)
    lasso_r2_score_test = r2_score(y_test, lasso_predictions)

    print("lasso training mae:", lasso_mae_train)
    print("lasso testing mae:", lasso_mae_test)
    print("lasso training mse:", lasso_mse_train)
    print("lasso testing mse:", lasso_mse_test)
    print("lasso training r2_score:", lasso_r2_score_train)
    print("lasso testing r2_score:", lasso_r2_score_test)



    # model 3 - decision tree regressor
    print()
    print("Decision tree regressor model")
    decision_tree_pipeline = Pipeline([
        ("decision_tree_model", DecisionTreeRegressor(max_depth=10, random_state=42)),
    ])

    decision_tree_pipeline.fit(X_train, y_train)
    decision_tree_predictions = decision_tree_pipeline.predict(X_test)

    # splitting evaluation by training and testing
    decision_tree_mae_train = mean_absolute_error(y_train, decision_tree_pipeline.predict(X_train))
    decision_tree_mse_train = mean_squared_error(y_train, decision_tree_pipeline.predict(X_train))
    decision_tree_r2_score_train = r2_score(y_train, decision_tree_pipeline.predict(X_train))

    decision_tree_mae_test = mean_absolute_error(y_test, decision_tree_predictions)
    decision_tree_mse_test = mean_squared_error(y_test, decision_tree_predictions)
    decision_tree_r2_score_test = r2_score(y_test, decision_tree_predictions)

    print("Decision tree training mae:", decision_tree_mae_train)
    print("Decision tree testing mae:", decision_tree_mae_test)
    print("Decision tree training mse:", decision_tree_mse_train)
    print("Decision tree testing mse:", decision_tree_mse_test)
    print("Decision tree training r2_score:", decision_tree_r2_score_train)
    print("Decision tree testing r2_score:", decision_tree_r2_score_test)



    # model 4 - random forest regressor (no max depth)
    print()
    print("Random forest regressor model")
    random_forest_pipeline = Pipeline([
        ("random_forest_model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])

    random_forest_pipeline.fit(X_train, y_train)
    random_forest_predictions = random_forest_pipeline.predict(X_test)

    # splitting evaluation by training and testing
    random_forest_mae_train = mean_absolute_error(y_train, random_forest_pipeline.predict(X_train))
    random_forest_mse_train = mean_squared_error(y_train, random_forest_pipeline.predict(X_train))
    random_forest_r2_score_train = r2_score(y_train, random_forest_pipeline.predict(X_train))

    random_forest_mae_test = mean_absolute_error(y_test, random_forest_predictions)
    random_forest_mse_test = mean_squared_error(y_test, random_forest_predictions)
    random_forest_r2_score_test = r2_score(y_test, random_forest_predictions)

    print("Random forest training mae:", random_forest_mae_train)
    print("Random forest testing mae:", random_forest_mae_test)
    print("Random forest training mse:", random_forest_mse_train)
    print("Random forest testing mse:", random_forest_mse_test)
    print("Random forest training r2_score:", random_forest_r2_score_train)
    print("Random forest testing r2_score:", random_forest_r2_score_test)




    # model 4.2 - random forest regressor (determining optimal value of max_depth and n_estimators
    # through grid search)
    print()
    print("Tuned random forest regressor model, with optimal max_depth and n_estimators")

    """
    random_forest_tuned_pipeline = Pipeline([
        ("random_forest_model_tuned", RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)),
    ])
    """

    random_forest_tuned_model = RandomForestRegressor(random_state=42)
    # uses grid search to find optimal parameters for tuned random forest model
    parameter_grid = {
        "n_estimators" : [50, 100, 200],
        "max_depth" : [None, 5, 7, 10, 12, 15]
    }

    # uses mae as scoring metric. note: used GridSearchCV documentation to learn parameters
    random_forest_grid_search = GridSearchCV(estimator = random_forest_tuned_model, param_grid=parameter_grid,
                               scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
    random_forest_grid_search.fit(X_train, y_train)

    # seeing results from grid search
    print("Optimal parameters:", random_forest_grid_search.best_params_)
    print()

    # assigns tuned model to the best one found through grid search
    random_forest_tuned_model = random_forest_grid_search.best_estimator_
    random_forest_tuned_predictions = random_forest_tuned_model.predict(X_test)

    """
    random_forest_tuned_pipeline.fit(X_train, y_train)
    random_forest_tuned_predictions = random_forest_tuned_pipeline.predict(X_test)
    """

    # splitting evaluation by training and testing
    random_forest_tuned_mae_train = mean_absolute_error(y_train, random_forest_tuned_model.predict(X_train))
    random_forest_tuned_mse_train = mean_squared_error(y_train, random_forest_tuned_model.predict(X_train))
    random_forest_tuned_r2_score_train = r2_score(y_train, random_forest_tuned_model.predict(X_train))

    random_forest_tuned_mae_test = mean_absolute_error(y_test, random_forest_tuned_predictions)
    random_forest_tuned_mse_test = mean_squared_error(y_test, random_forest_tuned_predictions)
    random_forest_tuned_r2_score_test = r2_score(y_test, random_forest_tuned_predictions)

    print("Tuned random forest training mae:", random_forest_tuned_mae_train)
    print("Tuned random forest testing mae:", random_forest_tuned_mae_test)
    print("Tuned random forest training mse:", random_forest_tuned_mse_train)
    print("Tuned random forest testing mse:", random_forest_tuned_mse_test)
    print("Tuned random forest training r2_score:", random_forest_tuned_r2_score_train)
    print("Tuned random forest testing r2_score:", random_forest_tuned_r2_score_test)



    # model 5 - KNN regressor, use grid_search to determine optimal number of neighbours
    print()
    print("KNN regressor model")

    # pipeline for scaling and KNN regressor model
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn_model", KNeighborsRegressor()),
    ])

    # use grid search to find optimal number of neighbours
    # note: had to include knn_model__ because I got an error
    knn_parameter_grid = {"knn_model__n_neighbors" : [3,5,7,9,11,13,15]}
    knn_grid_search = GridSearchCV(estimator = knn_pipeline, param_grid=knn_parameter_grid,
                                   cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    knn_grid_search.fit(X_train, y_train)


    # see results from grid search
    print("Optimal number of neighbours:", knn_grid_search.best_params_)
    print()

    # assigns knn model to the best one found through grid search
    knn_model = knn_grid_search.best_estimator_
    knn_predictions = knn_model.predict(X_test)

    # splitting evaluation by training and testing
    knn_mae_train = mean_absolute_error(y_train, knn_model.predict(X_train))
    knn_mse_train = mean_squared_error(y_train, knn_model.predict(X_train))
    knn_r2_score_train = r2_score(y_train, knn_model.predict(X_train))

    knn_mae_test = mean_absolute_error(y_test, knn_predictions)
    knn_mse_test = mean_squared_error(y_test, knn_predictions)
    knn_r2_score_test = r2_score(y_test, knn_predictions)

    print("KNN training mae:", knn_mae_train)
    print("KNN testing mae:", knn_mae_test)
    print("KNN training mse:", knn_mse_train)
    print("KNN testing mse:", knn_mse_test)
    print("KNN training r2_score:", knn_r2_score_train)
    print("KNN testing r2_score:", knn_r2_score_test)



if __name__ == "__main__":
    main()
