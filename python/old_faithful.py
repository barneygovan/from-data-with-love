#!/usr/bin/env python

import argparse

import faithful.bayesfmm as fmm
from faithful import ida, clustering, em
from stats import diagnostics, utils

def __run_clustering(data, output_dir):
    """k-means clustering code"""
    [km2,] = clustering.plot_kmeans(data, filename='%s/faithful_kmeans_k2.png' % output_dir)
    clustering.plot_kmeans(data, ks=(6,), filename='%s/faithful_kmeans_k6.png' % output_dir)
    # train and test data
    data_train, data_test = utils.split_data(data, train_split=0.8)
    [kmeans_model_2, kmeans_model_6] = clustering.plot_kmeans(data_train, ks=(2,6), suppress_output=True)
    clustering.kmeans_predict(data_test, kmeans_model_2,
                              filename='{0}/faithful_kmeans_k2_predict.png'.format(output_dir))
    clustering.kmeans_predict(data_test, kmeans_model_6,
                              filename='{0}/faithful_kmeans_k6_predict.png'.format(output_dir))

    return km2

def __run_em(data, output_dir, km2):
    theta, sigma, pi = em.gaussian_em_2(data, max_reps=100)
    em.plot_against_kmeans(data, theta, km2,
                           filename='{0}/faithful_em_kmeans.png'.format(output_dir))
    em.draw_contour_plots(theta[-1], sigma[-1], pi[-1],
                          filename='{0}/faithful_em_model.png'.format(output_dir))

def __run_bayesfmm(data, iterations, save_diagnostics, output_dir, burnin, km2):
    gaussian_fmm = fmm.GaussianFiniteMixtureModel()
    pi, theta, sigma = gaussian_fmm.run(data, k=2, iterations=iterations)

    outfilename = None
    if save_diagnostics:
        outfilename = '{0}/diagnostics_{1}_{2}.png'.format(output_dir, '{0}', '{1}')

    #generate diagnostics
    diagnostics.stationarity_plot(pi, ylab=r'\pi',
                                  filename=(outfilename.format('stationarity','pi')
                                    if outfilename is not None else None))
    diagnostics.stationarity_plot(theta, ylab=r'\theta',
                                  filename=(outfilename.format('stationarity','theta')
                                    if outfilename is not None else None))
    diagnostics.stationarity_plot(sigma, ylab=r'\Sigma',
                                 filename=(outfilename.format('stationarity','sigma')
                                    if outfilename is not None else None))

    #determine model parameters
    pi_mean, theta_mean, sigma_mean = fmm.calculate_posterior_means(pi, theta, sigma, burnin)
    # print pi_mean, theta_mean, sigma_mean
    em.plot_against_kmeans(data, theta, km2,
                           filename='{0}/faithful_bayesfmm_kmeans.png'.format(output_dir))
    fmm.draw_contour_plots(pi_mean, theta_mean, sigma_mean,
                           filename='{0}/faithful_bayesfmm_model.png'.format(output_dir))


def main(filename, iterations, save_diagnostics, output_dir, burnin):
    """ Run all the Old Faithful code """
    data = []
    with open(filename,'rb') as csvfile:
        #skip header
        _ = csvfile.next()
        for line in csvfile:
            eruption_time, waiting_time = line.split(',')
            data.append([float(eruption_time), float(waiting_time)])

    #generate ida images
    ida.scatter_plot(data, '{0}/faithful_ida_scatter.png'.format(output_dir))
    ida.histogram(data, '{0}/faithful_ida_hist.png'.format(output_dir))
    ida.linear_regression(data, '{0}/faithful_ida_regression.png'.format(output_dir))

    #clustering
    km2 = __run_clustering(data, output_dir)

    #expectation-maximization
    __run_em(data, output_dir, km2)

    #build bayes fmm model
    __run_bayesfmm(data, iterations, save_diagnostics, output_dir, burnin, km2)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('filename', metavar='datafile')
    cmdline_parser.add_argument('--iterations', action='store',
                                dest='iterations', type=int, default=500)
    cmdline_parser.add_argument('--save_diagnostics', action='store_true', default=False)
    cmdline_parser.add_argument('--output_dir', action='store', default='.')
    cmdline_parser.add_argument('--burnin', action='store', type=int, default=0)

    parsed_args = cmdline_parser.parse_args()

    main(parsed_args.filename, parsed_args.iterations,
         parsed_args.save_diagnostics, parsed_args.output_dir, parsed_args.burnin)

