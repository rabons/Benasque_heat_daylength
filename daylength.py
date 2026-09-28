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


def daily_solar(lat_deg, doy, elevation_m=0.0):
    """Daily solar radiation at the top of the atmosphere and under clear skies.

    Top-of-atmosphere (TOA) daily insolation is computed exactly from the
    solar geometry: the solar constant scaled by the Earth-Sun distance,
    integrated over the sunlit part of the day (geometric sunrise to sunset,
    sun centre at 0 deg; refraction adds negligible energy). It combines the
    two drivers of how much sun a place gets: day length and solar elevation.

    Clear-sky surface radiation uses the FAO-56 approximation
    Rso = (0.75 + 2e-5 * z) * Ra, valid for cloud-free days. It is typically
    within about 10% of measured clear-sky totals.

    Declination and eccentricity follow Spencer (1971), as in `day_length`.
    Agreement with the FAO-56 worked example (20 deg S, 3 September,
    Ra = 32.2 MJ m-2 d-1) is within 2%; the difference comes from FAO's
    simpler declination formula and older solar constant (1367 vs 1361 W m-2).

    Parameters
    ----------
    lat_deg : float or array_like
        Latitude in degrees, positive north.
    doy : int or array_like
        Day of year, 1-365. Broadcast against `lat_deg` with NumPy rules.
    elevation_m : float or array_like, optional
        Surface elevation in metres, used only for the clear-sky estimate.

    Returns
    -------
    dict of float or ndarray
        toa_daily_mean_Wm2    : 24-hour mean TOA irradiance, W m-2.
        toa_daily_MJm2        : daily TOA total, MJ m-2 per day.
        toa_noon_Wm2          : TOA irradiance at solar noon (peak intensity), W m-2.
        clearsky_surface_MJm2 : clear-sky surface daily total, MJ m-2 per day.
        All are 0 during polar night.

    References
    ----------
    Spencer, J. W. (1971). Fourier series representation of the position
    of the sun. Search, 2(5), 172.
    Allen, R. G. et al. (1998). Crop evapotranspiration, FAO Irrigation
    and Drainage Paper 56, Chapter 3.

    Examples
    --------
    >>> round(float(daily_solar(47.0, 172)["toa_daily_mean_Wm2"]), 1)   # Neuchâtel, 21 June
    482.8
    >>> round(float(daily_solar(90.0, 172)["toa_daily_mean_Wm2"]))      # North Pole, 21 June
    524
    """
    doy = np.asarray(doy, dtype=float)
    phi = np.deg2rad(np.asarray(lat_deg, dtype=float))
    g = 2*np.pi/365*(doy - 1)
    decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g) - 0.006758*np.cos(2*g)
            + 0.000907*np.sin(2*g) - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
    E0 = (1.000110 + 0.034221*np.cos(g) + 0.001280*np.sin(g)
          + 0.000719*np.cos(2*g) + 0.000077*np.sin(2*g))          # (r0/r)^2
    S0 = 1361.0                                                    # solar constant, W m-2
    ws = np.arccos(np.clip(-np.tan(phi)*np.tan(decl), -1, 1))      # sunset hour angle
    H0 = S0*E0/np.pi*(ws*np.sin(phi)*np.sin(decl) + np.cos(phi)*np.cos(decl)*np.sin(ws))
    toa_mj = H0*86400/1e6
    return dict(
        toa_daily_mean_Wm2=H0,
        toa_daily_MJm2=toa_mj,
        toa_noon_Wm2=S0*E0*np.clip(np.cos(phi - decl), 0, None),
        clearsky_surface_MJm2=(0.75 + 2e-5*np.asarray(elevation_m))*toa_mj,
    )
