"""
Bandpower binning and analytic error estimates.
"""

import numpy as np
from scipy.stats import binned_statistic as binnedstat

__all__ = ["bin1D", "bin2D", "bin_spectrum", "knox_errors"]


def bin_spectrum(cl, bin_edges):
    """Bin a spectrum defined at every multipole into bandpowers.

    Parameters
    ----------
    cl : ndarray
        Spectrum of shape (lmax+1,), starting at ell=0.
    bin_edges : ndarray
        Bin edges of the form low <= ell < high (the final edge is
        inclusive).

    Returns
    -------
    cents : ndarray
        Bin centers.
    binned : ndarray
        Unweighted mean of cl in each bin.
    """
    cl = np.asarray(cl, dtype=float)
    ells = np.arange(cl.size, dtype=float)
    return bin1D(np.asarray(bin_edges, dtype=float)).bin(ells, cl)


def knox_errors(cents, bin_edges, cl_xx, cl_yy, cl_xy, fsky):
    """Compute Knox-formula bandpower errors for a cross-spectrum.

    Parameters
    ----------
    cents : ndarray
        Bin centers.
    bin_edges : ndarray
        Bin edges.
    cl_xx, cl_yy, cl_xy : ndarray
        Binned auto- and cross-spectra (including noise).
    fsky : float
        Effective sky fraction.

    Returns
    -------
    sigma : ndarray
        1-sigma errors on the binned cross-spectrum.
    """
    dl = np.diff(bin_edges)
    nmodes = (2.0 * cents + 1.0) * dl * fsky
    var = (cl_xx * cl_yy + cl_xy**2) / nmodes
    return np.sqrt(var)


class bin1D:
    """
    * Takes data defined on x0 and produces values binned on x.
    * Assumes x0 is linearly spaced and continuous in a domain?
    * Assumes x is continuous in a subdomain of x0.
    * Should handle NaNs correctly.
    """

    def __init__(self, bin_edges):
        self.update_bin_edges(bin_edges)

    def update_bin_edges(self, bin_edges):
        self.bin_edges = bin_edges
        self.numbins = len(bin_edges) - 1
        self.cents = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        self.bin_edges_min = self.bin_edges.min()
        self.bin_edges_max = self.bin_edges.max()

    def bin(self, ix, iy, stat=np.nanmean):
        x = ix.copy()
        y = iy.copy()
        # this just prevents an annoying warning (which is otherwise informative) everytime
        # all the values outside the bin_edges are nans
        y[x < self.bin_edges_min] = 0
        y[x > self.bin_edges_max] = 0

        # pretty sure this treats nans in y correctly, but should double-check!
        bin_means = binnedstat(x, y, bins=self.bin_edges, statistic=stat)[0]

        return self.cents, bin_means


class bin2D:
    def __init__(self, modrmap, bin_edges):
        self.centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        self.cents = self.centers  # backwards compatibility
        self.digitized = np.digitize(modrmap.reshape(-1), bin_edges, right=True)
        self.bin_edges = bin_edges
        self.modrmap = modrmap

    def bin(self, data2d, weights=None, err=False, get_count=False, mask_nan=False):
        if weights is None:
            if mask_nan:
                keep = ~np.isnan(data2d.reshape(-1))
            else:
                keep = np.ones((data2d.size,), dtype=bool)
            count = np.bincount(self.digitized[keep])[1:-1]
            res = np.bincount(self.digitized[keep], (data2d).reshape(-1)[keep])[1:-1] / count
            if err:
                meanmap = self.modrmap.copy().reshape(-1) * 0
                for i in range(self.centers.size):
                    meanmap[self.digitized == i] = res[i]
                std = np.sqrt(
                    np.bincount(
                        self.digitized[keep],
                        ((data2d - meanmap.reshape(self.modrmap.shape)) ** 2.0).reshape(-1)[keep],
                    )[1:-1]
                    / (count - 1)
                    / count
                )
        else:
            count = np.bincount(self.digitized, weights.reshape(-1))[1:-1]
            res = np.bincount(self.digitized, (data2d * weights).reshape(-1))[1:-1] / count
        if get_count:
            assert not (err)  # need to make more general
            return self.centers, res, count
        if err:
            assert not (get_count)
            return self.centers, res, std
        return self.centers, res
