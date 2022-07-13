import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from tqdm import tqdm


## Use ML to detect whether or not a significant detection occurs between year t* and t*+L, where L is lead time
# y=1 (detection, positive class), y=0 (no detection, negative class)
## train on historical observations of objectives before t*, resampled by the decade

# Build dataset, features and y in columns, rows contain examples (model scenarios)
## objective: ['Rel_SOD_%', 'Upstream_Flood_Volume_taf'], alt = ['less', 'greater']
## t_star must be a decade ending in 1, e.g. 2021, 2031, 2041...

def build_data(objective, t_star=2020, L=30):
    if objective == 'Rel_SOD_%':
        alt = 'less'
    else:
        alt = 'greater'

    # store scenario names
    gcm_list = ['access1-0', 'bcc-csm1-1', 'bcc-csm1-1-m', 'canesm2', 'ccsm4', 'cesm1-bgc', 'cesm1-cam5',
                'cmcc-cm', 'cnrm-cm5', 'csiro-mk3-6-0', 'fgoals-g2', 'fio-esm', 'gfdl-cm3', 'gfdl-esm2g',
                'gfdl-esm2m', 'giss-e2-h-cc', 'giss-e2-r', 'giss-e2-r-cc', 'hadgem2-ao', 'hadgem2-cc',
                'hadgem2-es', 'inmcm4', 'ipsl-cm5a-mr', 'ipsl-cm5b-lr', 'miroc5', 'miroc-esm', 'miroc-esm-chem',
                'mpi-esm-lr', 'mpi-esm-mr', 'mri-cgcm3', 'noresm1-m']
    rcp_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    lulc_list = pd.read_csv('lulc_scenario_names.csv').name.to_list()

    # import p-vals, years of first detection
    p_vals = pd.read_csv('significance_results/nonparametric/' + objective + '/30_year_MA/' + alt + '_pvals_win30.csv',
                         index_col=0, parse_dates=True).drop(['count', 'rel_count'], axis=1)
    first_detection = pd.read_csv('significance_results/nonparametric/' + objective + '/30_year_MA/' + alt +
                                  '_single_total_win30.csv', index_col=1)
    first_detection = first_detection.drop(first_detection.columns[0], axis=1)

    # columns of objective export df (years to grab to resample by decade to calculate features)
    years = np.arange(1961, t_star + 1, 10).tolist()
    cols_mean = [str(s) + '_mean' for s in years]
    cols_std = [str(s) + '_std' for s in years]
    # index of export df (scenario names)
    export_index = list(p_vals)
    # create dataframe of objectives (features) and output y (detection (1) or no detection (0))
    export_df = pd.DataFrame(columns=cols_mean + cols_std, index=export_index)

    # populate export df with data from gcm/rcp/lulc combinations and y
    for gcm in tqdm(gcm_list, desc='Building dataset: '):
        for rcp in rcp_list:
            for lulc in lulc_list:
                try:
                    # read all scenarios
                    scenario = gcm + '_' + rcp + '_r1i1p1'
                    df = pd.read_csv('data/obj_' + scenario + '_' + lulc + '.csv.zip', index_col=0,
                                     parse_dates=True)
                    # add historical data (observations before t*), aggregate by decade (mean and standard deviation)
                    for year in years:
                        export_df.loc[gcm + '_' + rcp + '_' + lulc, str(year) + '_mean'] = \
                            df[str(year - 10) + '-10-01':str(year) + '-10-01'][objective].mean()
                        export_df.loc[gcm + '_' + rcp + '_' + lulc, str(year) + '_std'] = \
                            df[str(year - 10) + '-10-01':str(year) + '-10-01'][objective].std()

                    # add output variable y (check for detection in timeframe t* to t*+L)
                    # y=1 (detection), y=0 (no detection)
                    if np.any(p_vals.loc[str(t_star) + '-10-01':str(t_star + L) + '-10-01',
                              gcm + '_' + rcp + '_' + lulc] <= 0.05):
                        export_df.loc[gcm + '_' + rcp + '_' + lulc, 'y'] = 1
                    else:
                        export_df.loc[gcm + '_' + rcp + '_' + lulc, 'y'] = 0

                    # delete example from dataset if first detection already occurs before t*
                    if first_detection.loc[gcm + '_' + rcp + '_' + lulc, 'Year'] < t_star:
                        export_df = export_df.drop(gcm + '_' + rcp + '_' + lulc, axis=0)

                except FileNotFoundError:
                    pass

    # normalize features
    for col in export_df.columns:
        if col not in ['y']:
            export_df[col] = (export_df[col] - export_df[col].mean()) / export_df[col].std()

    return export_df


# Trains logistic regression classifier. Compare results to dummy classifier
# X= df of features, y= array of targets
def logistic_regrssion(X, y):
    # randomly split up data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # create instance of logistic regression model, fit training set, run predictions (on test set)
    LG_clf = LogisticRegression()
    LG_clf.fit(X_train, y_train)
    LG_predictions = LG_clf.predict(X_test)
    LG_coeff = LG_clf.coef_

    # create instance of dummy classifier (uniform distribution - random guesses)
    dummy_clf = DummyClassifier(strategy='uniform')
    dummy_clf.fit(X_train, y_train)
    dummy_predictions = dummy_clf.predict(X_test)

    return LG_predictions, dummy_predictions, y_test, LG_coeff


# calculates true positive, true negative rate
# y_test are array of true values in test set
def tptns(predictions, y_test):
    y_test = y_test.values
    TP = 0
    TN = 0
    for i in range(len(predictions)):
        if y_test[i] == predictions[i] == 1:
            TP += 1
        elif y_test[i] == predictions[i] == 0:
            TN += 1
    TP_rate = TP / len(y_test[np.where(y_test == 1)])
    TN_rate = TN / len(y_test[np.where(y_test == 0)])

    # m_test = len(y_test)
    # pos_test = len(y_test[np.where(y_test == 1)])
    # neg_test = len(y_test[np.where(y_test == 0)])

    return TP_rate, TN_rate


# Plot coefficients for mean/standard deviation features for fixed t*
def coeff_plot(objective, t_star=2031):
    # years used in features
    years = np.arange(1961, t_star + 1, 10)
    # try different lead times
    L_array = np.array([10, 20, 30, 40])
    # create matrix to store model coefficients
    coeff_mean = np.zeros((len(years), len(L_array)))
    coeff_std = np.zeros((len(years), len(L_array)))

    i = 0  # counter
    for L in L_array:
        export_df = build_data(objective=objective, t_star=t_star, L=L)
        X = export_df.drop(['y'], axis=1)
        y = export_df['y']
        LG_coeff = logistic_regrssion(X, y)[3]

        coeff_mean[:, i] = LG_coeff[0][0:int(len(LG_coeff[0]) / 2)]
        coeff_std[:, i] = LG_coeff[0][int(len(LG_coeff[0]) / 2):int(len(LG_coeff[0]))]
        i += 1

    df_mean = pd.DataFrame(coeff_mean, columns=L_array, index=years)
    df_std = pd.DataFrame(coeff_std, columns=L_array, index=years)

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=[8, 5])
    if objective == 'Rel_SOD_%':
        obj_title = 'Water supply reliability'
    else:
        obj_title = 'Flooding'

    df_mean.plot(ax=axes[0], xlabel='normalized trailing decade mean', legend=False)
    df_std.plot(ax=axes[1], xlabel='normalized trailing decade stdv')
    axes[1].legend(title='lead time, years')
    fig.suptitle(obj_title + ', $t^*= $' + str(t_star))
    fig.supxlabel('features')
    fig.supylabel('parameters')

    plt.tight_layout()
    plt.savefig('significance_results/nonparametric/' + objective + '/additional_materials/' + 'LG_parameters_' +
                str(t_star) + '.png')
    plt.clf()

    return


def heatmap(objective):
    # try t* and L combinations
    t_star_array = np.array([2001, 2011, 2021, 2031, 2041, 2051])
    L_array = np.array([10, 20, 30, 40])

    ## initiate performance matrices
    # size of test set
    m_matrix = np.zeros((len(L_array), len(t_star_array)))
    # f1 scores for logistic regression
    LG_TP_matrix = np.zeros((len(L_array), len(t_star_array)))
    LG_TN_matrix = np.zeros((len(L_array), len(t_star_array)))
    # f1 scores for dummy classifier
    D_TP_matrix = np.zeros((len(L_array), len(t_star_array)))
    D_TN_matrix = np.zeros((len(L_array), len(t_star_array)))

    # run models
    for i in range(len(L_array)):
        for j in range(len(t_star_array)):
            L = L_array[i]
            t_star = t_star_array[j]

            # train model with current t* and L value, update performance matrices
            export_df = build_data(objective, t_star=t_star, L=L)
            X = export_df.drop(['y'], axis=1)
            y = export_df['y']
            LG_predictions, dummy_predictions, y_test, LG_coeff = logistic_regrssion(X, y)

            # get f1 scores for Logistic regression
            LG_TP = tptns(predictions=LG_predictions, y_test=y_test)[0]
            LG_TN = tptns(predictions=LG_predictions, y_test=y_test)[1]

            # get f1 scores for dummy classifier
            D_TP = tptns(predictions=dummy_predictions, y_test=y_test)[0]
            D_TN = tptns(predictions=dummy_predictions, y_test=y_test)[1]

            m_matrix[i, j] = len(y_test)
            LG_TP_matrix[i, j] = LG_TP
            LG_TN_matrix[i, j] = LG_TN
            D_TP_matrix[i, j] = D_TP
            D_TN_matrix[i, j] = D_TN

    ##  draw heatmaps
    ##  heatmap of test set sample size
    fig, ax = plt.subplots()
    ax.set_xlabel('$t^*$')
    ax.set_ylabel('lead time, L (years)')

    im = ax.imshow(m_matrix)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(t_star_array)))
    ax.set_xticklabels([str(x) for x in t_star_array.tolist()])
    ax.set_yticklabels([str(x) for x in L_array.tolist()])
    ax.set_yticks(np.arange(len(L_array)))
    ax.set_title('Size of test set')
    # annotate
    for i in range(len(L_array)):
        for j in range(len(t_star_array)):
            text = ax.text(j, i, m_matrix[i, j],
                           ha="center", va="center", color="w")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('significance_results/nonparametric/' + objective + '/additional_materials/LG_test_size.png')
    plt.clf()

    ## heatmap of TP rates for logistic regression
    fig, ax = plt.subplots()
    ax.set_xlabel('$t^*$')
    ax.set_ylabel('lead time, L (years)')

    im = ax.imshow(LG_TP_matrix, vmin=0, vmax=1)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(t_star_array)))
    ax.set_xticklabels([str(x) for x in t_star_array.tolist()])
    ax.set_yticklabels([str(x) for x in L_array.tolist()])
    ax.set_yticks(np.arange(len(L_array)))
    ax.set_title('Logistic regression true positive rate')
    # annotate
    for i in range(len(L_array)):
        for j in range(len(t_star_array)):
            text = ax.text(j, i, round(LG_TP_matrix[i, j], 2),
                           ha="center", va="center", color="w")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('significance_results/nonparametric/' + objective + '/additional_materials/LG_true_positive.png')
    plt.clf()

    ## heatmap of TN rates for logistic regression
    fig, ax = plt.subplots()
    ax.set_xlabel('$t^*$')
    ax.set_ylabel('lead time, L (years)')

    im = ax.imshow(LG_TN_matrix, vmin=0, vmax=1)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(t_star_array)))
    ax.set_xticklabels([str(x) for x in t_star_array.tolist()])
    ax.set_yticklabels([str(x) for x in L_array.tolist()])
    ax.set_yticks(np.arange(len(L_array)))
    ax.set_title('Logistic regression true negative rate')
    # annotate
    for i in range(len(L_array)):
        for j in range(len(t_star_array)):
            text = ax.text(j, i, round(LG_TN_matrix[i, j], 2),
                           ha="center", va="center", color="w")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('significance_results/nonparametric/' + objective + '/additional_materials/LG_true_negative.png')
    plt.clf()

    ## heatmap of TP rates for dummy classifier
    fig, ax = plt.subplots()
    ax.set_xlabel('$t^*$')
    ax.set_ylabel('lead time, L (years)')

    im = ax.imshow(D_TP_matrix, vmin=0, vmax=1)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(t_star_array)))
    ax.set_xticklabels([str(x) for x in t_star_array.tolist()])
    ax.set_yticklabels([str(x) for x in L_array.tolist()])
    ax.set_yticks(np.arange(len(L_array)))
    ax.set_title('Random classifier true positive rate')
    # annotate
    for i in range(len(L_array)):
        for j in range(len(t_star_array)):
            text = ax.text(j, i, round(D_TP_matrix[i, j], 2),
                           ha="center", va="center", color="w")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('significance_results/nonparametric/' + objective + '/additional_materials/dummy_true_positive.png')
    plt.clf()

    ## heatmap of f1 scores for dummy classifier (negative class)
    fig, ax = plt.subplots()
    ax.set_xlabel('$t^*$')
    ax.set_ylabel('lead time, L (years)')

    im = ax.imshow(D_TN_matrix, vmin=0, vmax=1)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(t_star_array)))
    ax.set_xticklabels([str(x) for x in t_star_array.tolist()])
    ax.set_yticklabels([str(x) for x in L_array.tolist()])
    ax.set_yticks(np.arange(len(L_array)))
    ax.set_title('Random classifier true negative rate')
    # annotate
    for i in range(len(L_array)):
        for j in range(len(t_star_array)):
            text = ax.text(j, i, round(D_TN_matrix[i, j], 2),
                           ha="center", va="center", color="w")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('significance_results/nonparametric/' + objective + '/additional_materials/dummy_true_negative.png')
    plt.clf()

    return


def main():
    objectives = ['Rel_SOD_%', 'Upstream_Flood_Volume_taf']
    for objective in objectives:
        heatmap(objective)
        for t_star in [2001, 2031, 2051]:
            coeff_plot(objective=objective, t_star=t_star)

    return


if __name__ == "__main__":
    main()
