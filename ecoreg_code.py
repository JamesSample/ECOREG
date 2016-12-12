#-------------------------------------------------------------------------------
# Name:        ecoreg_code.py
# Purpose:     Useful functions for the ECOREG analysis.
#
# Author:      James Sample
#
# Created:     12/12/2016
#-------------------------------------------------------------------------------

def run_pca(df, cols=None, c=None):
    """ Applies PCA and generates summary plots.
    
    Args:
        df      Dataframe of features
        cols    Subset of columns to use
        c       Categorical variable for colouring points
        
    Returns:
        Dataframe of PC loadings. Also generates a range of plots.
    """  
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Define colours to use for different categories
    col_dict = {'D':'g', 'N':'b', 0:'g', 1:'b'}
    
    # Extract cols of interest
    if c:
        colours = df[[c]]

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

    print 'Variance explained by first 3 PCs (%):\n'
    print var_exp[:3]
    print '\nTotal: %.2f%%' % var_exp[:3].sum()
    
    # Get eigenvalues
    cov_mat = np.cov(feat_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
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
    axes[1].plot(range(1, len(cum_exp)+1), eig_vals, 
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
    
    # Build df with colours
    data = pd.DataFrame({'PC1':feat_reduced[:, 0],
                         'PC2':feat_reduced[:, 1],
                         c:colours[c]})    
    groups = data.groupby(c)

    # Plot
    for name, group in groups:
        ax.scatter(group['PC1'], group['PC2'], s=40,
                   label=name, c=col_dict[name])
    ax.set_xlabel('First principal component')
    ax.set_ylabel('Second principal component')
    ax.set_title('First two PCA directions')
    ax.legend(title=c, loc='best')
   
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
                         c:colours[c]})    
    groups = data.groupby(c)
    
    # Plot
    for name, group in groups:
        ax.scatter(group['PC1'], group['PC2'], group['PC3'],
                   label=name, c=col_dict[name], s=40)
    ax.set_title('First three PCA directions')
    ax.set_xlabel('First principal component')
    ax.set_ylabel('Second principal component')
    ax.set_zlabel('Third principal component')
    ax.legend(title=c, loc='best')
    plt.show()

    return loads