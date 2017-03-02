#-------------------------------------------------------------------------------
# Name:        ecoreg_code.py
# Purpose:     Useful functions for the ECOREG analysis.
#
# Author:      James Sample
#
# Created:     12/12/2016
#-------------------------------------------------------------------------------

def run_pca(df, cols=None):
    """ Applies PCA and generates summary plots.
    
    Args:
        df      Dataframe of features. Must include "country" and "regulated" 
                columns
        cols    Subset of columns to use
        
    Returns:
        Dataframe of PC loadings. Also generates a range of plots.
    """  
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import mpld3

    # Define and markers to use for different categories
    groups_dict = {(u'D', 0):('Germany, unregulated', 'g', 'o'),
                   (u'N', 0):('Norway, unregulated', 'b', 'o'),
                   (u'D', 1):('Germany, regulated', 'g', '^'),
                   (u'N', 1):('Norway, regulated', 'b', '^')}
    
    # Extract cols of interest
    cats = df[['country', 'regulated']]

    if cols:
        df = df[cols].astype(float)

    # Standardise the feature data
    feat_std = StandardScaler().fit_transform(df)

    # Setup PCA. Initially, choose to keep ALL components
    pca = PCA()

    # Fit model
    pca.fit(feat_std)

    # Get explained variances (in %)
    var_exp = 100*pca.explained_variance_ratio_
    cum_exp = np.cumsum(var_exp)

    # Get eigenvalues
    cov_mat = np.cov(feat_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Get number of EVs > 1 (Kaiser-Guttman criterion)
    # and print summary
    n_kgc = (eig_vals > 1).sum()
    print 'Variance explained by first %s PCs (%%):\n' % n_kgc
    print var_exp[:n_kgc]
    print '\nTotal: %.2f%%' % var_exp[:n_kgc].sum()
    
    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    # Explained variance
    axes[0].bar(range(1, len(var_exp)+1), var_exp, 
                align='center', label='Individual components')
    axes[0].plot(range(1, len(cum_exp)+1), cum_exp, 
                 'r-o', label='Cumulative')
    axes[0].set_xlabel('Principal component')
    axes[0].set_ylabel('Variance explained (%)')
    axes[0].legend(loc='center right')
    
    # Eigenvalues
    axes[1].plot(range(1, len(eig_vals)+1), np.sort(eig_vals)[::-1], 
                 'r-o', label='Eigenvalues')
    axes[1].axhline(1, c='k', ls='-', label='Kaiser-Guttman threshold')
    axes[1].set_xlabel('Principal component')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].legend(loc='upper right')  
    
    # PC loadings
    loads = pd.DataFrame(data=pca.components_, 
                         columns=df.columns,
                         index=range(1, pca.components_.shape[0]+1)).T

    # Project into 2 and 3 components
    fig = plt.figure(figsize=(16, 6))
   
    # Plot 2 components
    ax = fig.add_subplot(1, 2, 1)
    
    # Refit the PCA, this time specifying 2 components
    # and transforming the result
    feat_reduced = PCA(n_components=2).fit_transform(feat_std)
    
    # Build df 
    data = pd.DataFrame({'PC1':feat_reduced[:, 0],
                         'PC2':feat_reduced[:, 1],
                         'country':cats['country'],
                         'regulated':cats['regulated']})    

    groups = data.groupby(['country', 'regulated'])
    
    # Plot
    for name, group in groups:
        ax.scatter(group['PC1'], group['PC2'], s=60,
                   label=groups_dict[name][0],
                   c=groups_dict[name][1],
                   marker=groups_dict[name][2])
        
    ax.set_xlabel('First principal component')
    ax.set_ylabel('Second principal component')
    ax.set_title('First two PCA directions')
       
    # Plot 3 components
    ax = fig.add_subplot(1, 2, 2, projection='3d', 
                         elev=-150, azim=135)

    # Refit the PCA, this time specifying 3 components
    # and transforming the result
    feat_reduced = PCA(n_components=3).fit_transform(feat_std)

    # Build df with colours
    data = pd.DataFrame({'PC1':feat_reduced[:, 0],
                         'PC2':feat_reduced[:, 1],
                         'PC3':feat_reduced[:, 2],
                         'country':cats['country'],
                         'regulated':cats['regulated']})   
    
    groups = data.groupby(['country', 'regulated'])
    
    # Plot
    for name, group in groups:
        ax.scatter(group['PC1'], group['PC2'], group['PC3'],
                   label=groups_dict[name][0],
                   c=groups_dict[name][1],
                   marker=groups_dict[name][2],
                   s=60)
        
    ax.set_title('First three PCA directions')
    ax.set_xlabel('First principal component')
    ax.set_ylabel('Second principal component')
    ax.set_zlabel('Third principal component')
    ax.legend(bbox_to_anchor=(0.15, -0.1), frameon=True)
    plt.show()

    return loads

def bayesian_t(df, val_col, grp_col='regulated',
               sig_fac=2, unif_l=0, unif_u=20,
               exp_mn=30, 
               plot_trace=False, plot_ppc=False,
               plot_vars=False, plot_diffs=True,
               steps=2000, mcmc='metropolis'):
    """ Simple Bayesian test for differences between two groups.
    
    Args:
        df         Dataframe. Must have a column containing values
                   and a categorical 'regulated' column that is [0, 1]
                   to define the two groups
        val_col    Name of the values column
        grp_col    Name of the categorical column defining the groups
        sig_fac    Factor applied to std. dev. of pooled data to define
                   prior std. dev. for group means
        unif_l     Lower bound for uniform prior on std. dev. of group
                   means
        unif_u     Upper bound for uniform prior on std. dev. of group
                   means
        exp_mn     Mean of exponential prior for v in Student-T 
                   distribution
        plot_trace Whether to plot the MCMC traces
        plot_ppc   Whether to perform and plot the Posterior Predictive
                   Check 
        plot_vars  Whether to plot posteriors for variables
        plot_diffs Whether to plot posteriors for differences
        steps      Number of steps to take in MCMC chains
        mcmc       Sampler to use: ['metropolis', 'slice', 'nuts']
    
    Returns:
        Creates plots showing the distribution of differences in 
        means and variances, plus optional diagnostics. Returns the 
        MCMC trace
    """
    import numpy as np
    import pymc3 as pm
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt

    # Get overall means and s.d.
    mean_all = df[val_col].mean()
    std_all = df[val_col].std()

    # Group data
    grpd = df.groupby(grp_col)
    
    # Separate groups
    reg_data = grpd.get_group(1)[val_col].values
    ureg_data = grpd.get_group(0)[val_col].values   

    # Setup model
    with pm.Model() as model:
        # Priors for means of Student-T dists
        reg_mean = pm.Normal('regulated_mean', mu=mean_all, sd=std_all*sig_fac)
        ureg_mean = pm.Normal('unregulated_mean', mu=mean_all, sd=std_all*sig_fac)

        # Priors for std. dev. of Student-T dists
        reg_std = pm.Uniform('regulated_std', lower=unif_l, upper=unif_u)
        ureg_std = pm.Uniform('unregulated_std', lower=unif_l, upper=unif_u)

        # Prior for v of Student-T dists
        nu = pm.Exponential('v_minus_one', 1./29.) + 1

        # Define Student-T dists
        # PyMC3 uses precision = 1 / (sd^2) to define dists rather than std. dev.
        reg_lam = reg_std**-2
        ureg_lam = ureg_std**-2

        reg = pm.StudentT('regulated', nu=nu, mu=reg_mean, lam=reg_lam, observed=reg_data)
        ureg = pm.StudentT('unregulated', nu=nu, mu=ureg_mean, lam=ureg_lam, observed=ureg_data)

        # Quantities of interest (difference of means and std. devs.)
        diff_of_means = pm.Deterministic('difference_of_means', reg_mean - ureg_mean)
        diff_of_stds = pm.Deterministic('difference_of_stds', reg_std - ureg_std)
        
        # Run sampler to approximate posterior
        if mcmc == 'metropolis':
            trace = pm.sample(steps, step=pm.Metropolis())
        elif mcmc == 'slice':
            trace = pm.sample(steps, step=pm.Slice())
        elif mcmc == 'nuts':
            trace = pm.sample(steps)
        else:
            raise ValueError("mcmc must be one of ['metropolis', 'slice', 'nuts']")

    # Plot results
    # Raw data
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,4))
    
    for name, grp in grpd:
        sn.distplot(grp[val_col].values, ax=axes[name], kde=False)
        axes[name].set_title('Regulated = %s' % name)        

    # Traces
    if plot_trace:
        pm.traceplot(trace)
    
    # Posteriors for variables
    if plot_vars:
        pm.plot_posterior(trace[1000:],
                          varnames=['regulated_mean', 'unregulated_mean', 
                                    'regulated_std', 'unregulated_std'],
                          alpha=0.3)

    # Posteriors for differences
    if plot_diffs:
        pm.plot_posterior(trace[1000:],
                          varnames=['difference_of_means', 'difference_of_stds'],
                          ref_val=0,
                          alpha=0.3)
        
    # Posterior predictive check
    if plot_ppc:
        ppc = pm.sample_ppc(trace, samples=500, model=model, size=100)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,4))

        sn.distplot([n.mean() for n in ppc['unregulated']], ax=axes[0])
        axes[0].axvline(ureg_data.mean(), c='k')
        axes[0].set(title='Posterior predictive of the mean (unregulated)', 
                    xlabel='Mean', 
                    ylabel='Frequency')

        sn.distplot([n.mean() for n in ppc['regulated']], ax=axes[1])
        axes[1].axvline(reg_data.mean(), c='k')
        axes[1].set(title='Posterior predictive of the mean (regulated)', 
                    xlabel='Mean', 
                    ylabel='Frequency')
    
    return trace

def mlr(df, exp_vars, resp_var, 
        method='ols', 
        fit_intercept=True,
        kcv=3,
        normalize=False):
    """ Performs various types of multiple linear regression.
    
    Args:
        df:            Data frame with features/responses as columns 
                       and samples as rows.
        exp_vars:      List of string specifying explanatory variables.
        resp_var:      String specifying the response variable.
        method:        'ols', 'lasso', 'ridge', 'el-net'.
        fit_intercept: Whether to fit an intercept. Default is True.
        kcv:           Number of "folds" for k-fold cross validation.
                       Default is 3.
        normalize:     Whether to normalise X before regression.
                       Default is False.
    Returns:
        A data frame of parameter estimates with (rather dodgy?)
        2-sigma error bounds and 95% significance
    """
    from sklearn import cross_validation
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.linear_model import LassoCV, ElasticNetCV
    from sklearn.metrics import r2_score
    from sklearn.utils import resample
    import matplotlib.pyplot as plt
    import seaborn as sn
    import pandas as pd
    import numpy as np
    
    # Separate data
    X = df[exp_vars]
    y = df[resp_var]
    
    # Setup model
    if method == 'ols':
        model = LinearRegression(fit_intercept=fit_intercept, 
                                 normalize=normalize)
    elif method == 'lasso':
        model = LassoCV(fit_intercept=fit_intercept, 
                        normalize=normalize, 
                        max_iter=10000,
                        cv=kcv)
    elif method == 'ridge':
        model = RidgeCV(fit_intercept=fit_intercept, 
                        normalize=normalize, 
                        alphas=np.logspace(-10, 10, 21))
    elif method == 'el-net':
        model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                             fit_intercept=fit_intercept, 
                             normalize=normalize,
                             cv=kcv)
    else:
        raise ValueError('"method" parameter must be in ["ols", "lasso", "ridge", "el-net"]')
    
    # k-fold cross validation
    #cv_scores = cross_validation.cross_val_score(model, X, y, cv=kcv, scoring='r2')
    #print 'Mean r2 from %s-fold CV: %.3f\n' % (kcv, cv_scores.mean())
    
    # Train model on full dataset
    model.fit(X, y)
    
    # Get y-hat
    y_pred = model.predict(X)
    
    # r2 based on calibration data
    r2 = r2_score(y, y_pred)
    print 'r2:', r2
    print ''
    
    # Summary of model
    print model
    print ''
    
    if method == 'lasso':
        print 'Lasso alpha:', model.alpha_
        print ''
    elif method == 'ridge':
        print 'Ridge alpha:', model.alpha_
        print ''
    elif method == 'el-net':
        print 'Elastic net alpha:', model.alpha_   
        print 'Elastic net L1 ratio:', model.l1_ratio_ 
        print ''
    else: # OLS
        pass
    
    # Plot
    fig = plt.figure(figsize=(15,15))
    
    # Paired points for each site
    ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
    ax1.plot(range(0, len(X.index)), y, 'ro', label='Observed')
    ax1.plot(range(0, len(X.index)), y_pred, 'b^', label='Modelled')
    
    ax1.set_xticks(range(0, len(X.index)))
    ax1.set_xticklabels(X.index, rotation=90, fontsize=12)
    ax1.set_xlim(0, len(X.index)-1)
    
    ax1.set_xlabel('Site code', fontsize=16)
    ax1.set_ylabel(resp_var)
    ax1.set_title('Points paired for each location', fontsize=20)
    ax1.legend(loc='best', fontsize=16)
    
    # Modelled versus observed
    ax2 = plt.subplot2grid((2,2), (1,0), colspan=1)
    ax2.plot(y, y_pred, 'ro')
    ax2.set_xlabel('Observed', fontsize=16)
    ax2.set_ylabel('Modelled', fontsize=16)
    ax2.set_title('Modelled versus observed', fontsize=20)
    
    # Hist of residuals
    ax3 = plt.subplot2grid((2,2), (1,1), colspan=1)
    sn.distplot(y - y_pred, kde=True, ax=ax3)
    ax3.set_title('Histogram of residuals', fontsize=20)
    
    plt.tight_layout()
    
    # Get param estimates
    params = pd.Series(model.coef_, index=X.columns)

    # Estimate confidence using bootstrap
    # i.e. what is the std. dev. of the estimates for each parameter
    # based on 1000 resamplings
    err = np.std([model.fit(*resample(X, y)).coef_ for i in range(1000)], 
                 axis=0)

    # Build df
    res = pd.DataFrame({'effect':params,
                        'error':2*err})

    # Rough indicator of significance: are the estimated values more than
    # 2 std. devs. from 0 (~95% CI?). NB: this assumnes the "marginal posterior"  
    # is normal, which I haven't tested for and which quite possibly isn't true
    # - use with care! 
    res['signif'] = np.abs(res['effect']) > res['error']
    
    return res

def robust_lin_reg(df, var_map, 
                   steps=2000, mcmc='metropolis',
                   plot_trace=True, plot_vars=True):
    """ Robust Bayesian linear regression.
    
    Args:
        df         Dataframe. Must have a column containing values
                   and a categorical 'regulated' column that is [0, 1]
                   to define the two groups
        val_map    Dict specifying x and y vars: {'x':'expl_var',
                                                  'y':'resp_var'}
        steps      Number of steps to take in MCMC chains
        mcmc       Sampler to use: ['metropolis', 'slice', 'nuts']
        plot_trace Whether to plot the MCMC traces
        plot_vars  Whether to plot posteriors for variables
    
    Returns:
        Creates plots showing the distribution of differences in 
        means and variances, plus optional diagnostics. Returns the 
        MCMC trace
    """
    import pymc3 as pm
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import theano 
    
    # Get cols
    df = df[var_map.values()]
    
    # Swap keys and values
    var_map_rev = dict((v,k) for k,v in var_map.iteritems())
    
    # Convert df columns to x and y
    df.columns = [var_map_rev[i] for i in df.columns]

    with pm.Model() as model:
        # Priors
        nu = pm.Exponential('v_minus_one', 1./29.) + 1
        
        # The patsy string below automatically assumes mu=0 and estimates
        # lam = (1/s.d.**2), so don't need to add these. Do need to add 
        # prior for nu though.
        family = pm.glm.families.StudentT(nu=nu)
        
        # Define model
        pm.glm.glm('y ~ x', df, family=family)
        
        # Find MAP as starting point
        start = pm.find_MAP()

        # Run sampler to approximate posterior
        if mcmc == 'metropolis':
            step = pm.Metropolis()
            trace = pm.sample(steps, step, start=start)
        elif mcmc == 'slice':
            step = pm.Slice()
            trace = pm.sample(steps, step, start=start)
        elif mcmc == 'nuts':
            step = pm.NUTS(scaling=start)
            trace = pm.sample(steps, step)
        else:
            raise ValueError("mcmc must be one of ['metropolis', 'slice', 'nuts']")

    # Traces
    if plot_trace:
        pm.traceplot(trace)
    
    # Posteriors for variables
    if plot_vars:
        pm.plot_posterior(trace[-1000:],
                          varnames=['v_minus_one', 'lam'],
                          alpha=0.3)

        pm.plot_posterior(trace[1000:],
                          varnames=['x', 'Intercept'],
                          ref_val=0,
                          alpha=0.3)
        
    # PPC
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, xlabel=var_map['x'], ylabel=var_map['y'])
    ax.scatter(df.x, df.y, marker='o', label='Data')
    pm.glm.plot_posterior_predictive(trace, samples=50, eval=df.x,
                                     label='PPC', alpha=0.3)
    
    return trace

def plot_lasso_path(df, resp_var, exp_vars):
    """ Plot the lasso path. Both response and explanatory
        variables are standardised first.
    
    Args:
        df:       Dataframe
        resp_var: String. Response variable
        exp_vars: List of strings. Explanatory variables
    
    Returns:
        Dataframe of path and matplotlib figure with tooltip-
        labelled lines. To view this figure in a notebook, use
        mpld3.display(f) on the returned figure object, f.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mpld3
    from mpld3 import plugins
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import lasso_path

    # Standardise the feature data and response
    feat_std = StandardScaler().fit_transform(df[[resp_var,] + exp_vars])
    
    # Calculate lasso path
    alphas, coefs, _ = lasso_path(feat_std[:, 1:],         # X
                                  feat_std[:, 0],          # y
                                  eps=1e-3,                # Path length
                                  fit_intercept=False)     # Already centred

    # -Log(alphas) is easier for display
    neg_log_alphas = -np.log10(alphas)

    # Build df of results
    res_df = pd.DataFrame(data=coefs.T, index=alphas, columns=exp_vars)
    
    # Plot
    fig, ax = plt.subplots()

    for coef, name in zip(coefs, exp_vars):
        line = ax.plot(neg_log_alphas, coef, label=name)
        plugins.connect(fig, plugins.LineLabelTooltip(line[0], label=name))

    plt.xlabel('-Log(alpha)')
    plt.ylabel('Coefficients')
    plt.title('Lasso paths')
    plt.legend(loc='best', title='', ncol=3)    
    
    return res_df, fig

def best_lasso(df, resp_var, exp_vars, kcv=3, cv_path=False, 
               hists=False):
    """ Find the best lasso model through cross-validation.
    
    Args:
        df:       Dataframe
        resp_var: String. Response variable
        exp_vars: List of strings. Explanatory variables
        kcv:      Number of cross-validation folds
        cv_path:  Whether to plot the path of cross-validation
                  scores
        hists:    Whether to plot histograms of coefficient
                  estimates based on bootstrapping
    
    Returns:
        Dataframe of coefficients for best model and histograms
        of coefficient variability based on bootstrap resampling.
    """
    import seaborn as sn
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV
    from sklearn.utils import resample

    # Standardise the feature data and response
    feat_std = StandardScaler().fit_transform(df[[resp_var,] + exp_vars])

    model = LassoCV(fit_intercept=False, 
                    normalize=False, 
                    max_iter=10000,
                    cv=kcv,
                    eps=1e-3)

    # Train model on full dataset
    model.fit(feat_std[:, 1:], feat_std[:, 0])

    print model

    # Get param estimates
    params = pd.DataFrame(pd.Series(model.coef_, index=exp_vars))
    
    if cv_path:
        # Display results
        m_log_alphas = -np.log10(model.alphas_)

        plt.figure()
        plt.plot(m_log_alphas, model.mse_path_, ':')
        plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
                 label='Average across the folds', linewidth=2)
        plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                    label='alpha: CV estimate')

        plt.legend()

        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean square error')
        plt.axis('tight')

        plt.show()

    if hists:
        # Estimate confidence using bootstrap
        # i.e. what is the std. dev. of the estimates for each parameter
        # based on 1000 resamplings
        err = np.array([model.fit(*resample(feat_std[:, 1:], 
                                            feat_std[:, 0])).coef_ for i in range(1000)])
        err_df = pd.DataFrame(data=err, columns=exp_vars)

        # Melt for plotting with seaborn
        err_df = pd.melt(err_df)
        g = sn.FacetGrid(err_df, col="variable", col_wrap=4)
        g = g.map(plt.hist, "value", bins=20)

        # Vertical line at 0
        g.map(sn.plt.axvline, x=0, c='k', lw=2)
    
    return params