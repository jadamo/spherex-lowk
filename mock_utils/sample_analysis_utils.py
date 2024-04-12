import numpy as np
from scipy.integrate import quad
from scipy.spatial import cKDTree


def GSMF(sm):
    logsm = np.log10(sm)
    V     = 256                             #Survey volume in Mpc3
    Phi, bin_edges = np.histogram(logsm ,bins= 30) #Unnormalized histogram and bin edges
    dM    = bin_edges[1] - bin_edges[0]                 #Bin size
    Max   = bin_edges[0:-1] + dM/2.               #Mass axis
    Phi   = Phi / V / dM                    #Normalize to volume and bin size
    return Max, Phi



def compute_pair_counts(points, bins):
    """Compute the pair counts for a set of points using a KDTree."""
    tree = cKDTree(points)
    return tree.count_neighbors(tree, bins, cumulative=False)

def xyz_to_xi(x_subset, y_subset, z_subset):
    '''
    From 2 methods. 
    
    '''
    # Define the bin edges for separation r
    bins = np.logspace(0.01, 2, 100)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Compute DD(r) for the observed data subset
    observed_points = np.vstack((x_subset, y_subset, z_subset)).T
    DD = compute_pair_counts(observed_points, bins)

    # Generate a random sample of points in the same volume
    random_points = np.vstack((
        np.random.uniform(min(x_subset), max(x_subset), len(x_subset)),
        np.random.uniform(min(y_subset), max(y_subset), len(x_subset)),
        np.random.uniform(min(z_subset), max(z_subset), len(x_subset))
    )).T

    # Compute RR(r) for the random sample
    RR = compute_pair_counts(random_points, bins)

    # Compute the two-point correlation function
    xi = DD / RR - 1

    # Remove the last value of xi to match the size of bin_centers
    xi = xi[:-1]
    
    # Compute DR(r) for the observed data subset and the random sample
    DR = cKDTree(observed_points).count_neighbors(cKDTree(random_points), bins)

    # Compute the two-point correlation function using Landy-Szalay estimator
    xi_Landy_Szalay = (DD - 2*DR + RR) / RR

    xi_Landy_Szalay = xi_Landy_Szalay[:-1]
    
    return bin_centers, xi, xi_Landy_Szalay

# adds gaussian noise to 
def add_gaussian_noise(ra, dec, redshift, sigma_ra, sigma_dec, sigma_z):
    
    # For now, assume that there's a constant redshift-independent ra, dec, and z noise
    ra_noisy, dec_noisy, redshift_noisy = np.zeros(len(ra)), np.zeros(len(dec)), np.zeros(len(redshift))
    for i in range(len(ra)):
        ra_noisy[i] = np.random.default_rng().normal(ra[i], sigma_ra)
        dec_noisy[i] = np.random.default_rng().normal(dec[i], sigma_dec)
        redshift_noisy[i] = np.random.default_rng().normal(redshift[i], sigma_z)

    return ra_noisy, dec_noisy, redshift_noisy

# Define the comoving distance integral
def integrand(z):
    # Define cosmological parameters for a standard Lambda-CDM model
    H0 = 70.0  # Hubble constant at z=0 in km/s/Mpc
    Om0 = 0.3  # Matter density parameter
    c = 299792.458  # Speed of light in km/s
    
    Ez = (Om0 * (1 + z)**3 + (1 - Om0))**0.5
    return c / (H0 * Ez)

def radec_to_cartesian(ra, dec, redshift):

    # Compute comoving distance for each redshift value
    comoving_distances = np.array([quad(integrand, 0, z)[0] for z in redshift])

    # Convert RA, Dec, and comoving distance to Cartesian coordinates
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    x = comoving_distances * np.cos(dec_rad) * np.cos(ra_rad)
    y = comoving_distances * np.cos(dec_rad) * np.sin(ra_rad)
    z = comoving_distances * np.sin(dec_rad)

    # For computation time considerations, use a subset of the data
    subset_indices = np.random.choice(len(x), size=len(x), replace=False)
    x_subset = x[subset_indices]
    y_subset = y[subset_indices]
    z_subset = z[subset_indices]
    
    return x_subset, y_subset, z_subset