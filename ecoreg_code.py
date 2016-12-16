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