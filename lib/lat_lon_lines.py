"""
File to create lattidue and longitude lines for moll and ortho projections
https://github.com/pb-aj/un-spot-able
"""


import numpy as np

def get_moll_latitude_lines(dlat=30, npts=1000, niter=100,unseen_deg=None):
    res = []

    # Work in degrees for the range, then convert to radians for math
    try:
        latlines_deg = np.arange(-90, 90, dlat)[1:]
    except:
        latlines_deg = np.array([])

    if not unseen_deg is None:
        latlines_deg = np.append(latlines_deg, unseen_deg)
    
    for lat_deg in latlines_deg:
        lat_rad = np.deg2rad(lat_deg)
        theta = lat_rad
        # Newton's method to find theta (must stay in radians)
        for n in range(niter):
            den = 2 + 2 * np.cos(2 * theta)
            if np.abs(den) < 1e-6:
                break
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat_rad)) / den
        
        # Scale output to degrees: x spans ±180, y spans ±90
        x = np.linspace(-180, 180, npts)
        y = np.ones(npts) * 90 * np.sin(theta)
        
        # Clip to the elliptical boundary
        a, b = 90, 180
        y[(y / a) ** 2 + (x / b) ** 2 > 1.00001] = np.nan
        res.append((x, y))
    return res


def get_moll_longitude_lines(dlon=30, npts=1000, niter=100):
    res = []
    # Longitude range in degrees
    lonlines_deg = np.arange(-180, 180+dlon, dlon)
    
    for lon_deg in lonlines_deg:
        # Generate latitude points in degrees, convert to radians for math
        lat_deg = np.linspace(-90, 90, npts)
        lat_rad = np.deg2rad(lat_deg)
        theta = np.array(lat_rad)
        
        for n in range(niter):
            den = 2 + 2 * np.cos(2 * theta)
            den[den < 1e-6] = 1e-6
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat_rad)) / den
        
        # Standard Mollweide x = (2*sqrt(2)/pi) * lon * cos(theta)
        # To scale x to ±180 when lon is ±180:
        # x_deg = lon_deg * cos(theta)
        x = lon_deg * np.cos(theta)
        y = 90 * np.sin(theta)
        
        res.append((x, y))
    return res


def get_ortho_latitude_lines(inc=90, obl=0, fproj=0, dlat=30, npts=1000,unseen_deg=None):
    # Convert inputs to radians for math
    inc_rad = np.deg2rad(inc)
    obl_rad = np.deg2rad(obl)
    
    ci = np.cos(inc_rad)
    si = np.sin(inc_rad)
    co = np.cos(obl_rad)
    so = np.sin(obl_rad)

    res = []

    try:
        # Latitude lines from -90 to 90
        latlines_deg = np.arange(-90, 90, dlat)[1:]
    except:
        latlines_deg = np.array([])

    if not unseen_deg is None:
        latlines_deg = np.append(latlines_deg, unseen_deg)
    
    for lat_deg in latlines_deg:
        lat_rad = np.deg2rad(lat_deg)

        # Ellipse parameters (unit scale)
        y0 = np.sin(lat_rad) * si
        a = np.cos(lat_rad)
        b = a * ci
        
        x = np.linspace(-a, a, npts)
        # Avoid sqrt of tiny negatives
        term = np.maximum(0, 1 - (x / a) ** 2)
        y1 = y0 - b * np.sqrt(term)
        y2 = y0 + b * np.sqrt(term)

        # Mask lines on the backside
        if si != 0:
            if inc_rad > np.pi / 2:
                y1[y1 < y1[np.argmax(x ** 2 + y1 ** 2)]] = np.nan
                y2[y2 < y2[np.argmax(x ** 2 + y2 ** 2)]] = np.nan
            else:
                y1[y1 > y1[np.argmax(x ** 2 + y1 ** 2)]] = np.nan
                y2[y2 > y2[np.argmax(x ** 2 + y2 ** 2)]] = np.nan

        # Scale to degrees (Radius 90) and rotate
        scale = 90
        for y in (y1, y2):
            xr = (-x * co - (y * (1 - fproj)) * so) * scale
            yr = (-x * so + (y * (1 - fproj)) * co) * scale
            res.append((xr, yr))

    return res[::2]


def get_ortho_longitude_lines(inc=90, obl=0, fproj=0, theta=0, dlon=30, npts=1000):
    # Convert inputs to radians
    inc_rad = np.deg2rad(inc)
    obl_rad = np.deg2rad(obl)
    theta_rad = np.deg2rad(theta)
    
    ci = np.cos(inc_rad)
    si = np.sin(inc_rad)
    co = np.cos(obl_rad)
    so = np.sin(obl_rad)

    # Equator-on check in degrees
    equator_on = (inc > 88) and (inc < 92)

    res = []
    if equator_on:
        offsets_deg = np.arange(-90, 90+dlon, dlon)
    else:
        offsets_deg = np.arange(0, 360+dlon, dlon)

    for offset_deg in offsets_deg:
        offset_rad = np.deg2rad(offset_deg)
        
        if equator_on:
            sgns = [1]
            bsgn = 1 if np.cos(theta_rad + offset_rad) >= 0 else -1
        else:
            bsgn = 1
            sgns = [1, -1] if np.cos(theta_rad + offset_rad) >= 0 else [-1, 1]

        for lon_rad, sgn in zip([0, np.pi], sgns):
            # 3D points on unit sphere
            y = np.linspace(-1, 1, npts)
            b = bsgn * np.sin(lon_rad - theta_rad - offset_rad)
            x = b * np.sqrt(np.maximum(0, 1 - y ** 2))
            z = sgn * np.sqrt(np.abs(1 - x ** 2 - y ** 2))

            if not equator_on:
                # Manual Rotation matrix for Inc (Rotation around X axis)
                angle = np.pi / 2 - inc_rad
                # x remains x
                y_new = y * np.cos(angle) - z * np.sin(angle)
                z_new = y * np.sin(angle) + z * np.cos(angle)
                x, y, z = x, y_new, z_new

                # Mask lines on the backside
                if si != 0:
                    if inc_rad < np.pi / 2:
                        y[:np.argmax(x ** 2 + y ** 2) + 1] = np.nan
                    else:
                        y[np.argmax(x ** 2 + y ** 2):] = np.nan

            # Scale to degrees (Radius 90) and apply obliquity
            scale = 90
            xr = (-x * co - (y * (1 - fproj)) * so) * scale
            yr = (-x * so + (y * (1 - fproj)) * co) * scale
            res.append((xr, yr))

    return res
