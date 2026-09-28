import numpy as np

def day_length(lat_deg, doy):
    """Astronomical day length (sunrise to sunset) in hours.

    Sunrise and sunset are defined as the moments the sun's centre is 0.833°
    below the horizon, which accounts for atmospheric refraction (~0.567°)
    and the solar semi-diameter (~0.267°). This is the standard definition
    used by almanacs and weather services.

    The solar declination is computed with Spencer's (1971) Fourier series,
    accurate to about 0.035° (~0.0006 rad). The resulting day length is
    typically within a few minutes of ephemeris-based values at mid-latitudes;
    errors grow near the polar circles, where day length changes rapidly.

    Parameters
    ----------
    lat_deg : float or array_like
        Latitude in degrees, positive north. Longitude is not needed:
        day length depends only on latitude and date.
    doy : int or array_like
        Day of year, 1 (1 January) to 365. Leap years are ignored; the
        error from this is well below one minute.
        `lat_deg` and `doy` are broadcast against each other with NumPy
        rules, e.g. lat[:, None] and doy[None, :] give a (lat, doy) grid.

    Returns
    -------
    float or ndarray
        Day length in hours, in [0, 24]. Returns 24 during polar day and
        0 during polar night (the hour-angle cosine is clipped to [-1, 1]).
        Night length is 24 minus this value.

    References
    ----------
    Spencer, J. W. (1971). Fourier series representation of the position
    of the sun. Search, 2(5), 172.

    Examples
    --------
    >>> round(float(day_length(47.0, 172)), 2)   # Neuchâtel, 21 June
    15.76
    >>> float(day_length(80.0, 172))             # polar day
    24.0
    """
    g = 2*np.pi/365*(np.asarray(doy) - 1)
    decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g) - 0.006758*np.cos(2*g)
            + 0.000907*np.sin(2*g) - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
    phi = np.deg2rad(np.asarray(lat_deg))
    cos_w0 = (np.sin(np.deg2rad(-0.833)) - np.sin(phi)*np.sin(decl)) / (np.cos(phi)*np.cos(decl))
    return 2*np.rad2deg(np.arccos(np.clip(cos_w0, -1, 1)))/15
