"""Geometry shared by the scripts: equirectangular panorama <-> rays <-> perspective faces.

Panorama frame: x right, y down, z forward (the centre of the picture). A pixel (u, v) of a W x H panorama looks along
    lon = (u / W - 0.5) * 360 deg,  lat = (0.5 - v / H) * 180 deg,   ray = (cos lat sin lon, -sin lat, cos lat cos lon)
A face is a pinhole camera with the same centre, turned by yaw (to the right) and pitch (up).
"""
import cv2
import numpy as np


def face_rotation(yaw_deg, pitch_deg):
    """R with x_face = R @ x_pano (COLMAP's cam_from_rig rotation)."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])     # pano <- yawed
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])     # yawed <- pitched up (y is down)
    return (Ry @ Rx).T


def rays_to_lonlat(d):
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(d[..., 0], d[..., 2])), np.degrees(-np.arcsin(np.clip(d[..., 1], -1, 1)))


def lonlat_to_rays(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)


def face_lonlat(R, size, focal):
    """Longitude and latitude (deg) seen by every pixel centre of a face."""
    c = (np.arange(size) + 0.5 - size / 2) / focal
    x, y = np.meshgrid(c, c)
    return rays_to_lonlat(np.stack([x, y, np.ones_like(x)], -1) @ R)          # rows of (ray @ R) = R.T @ ray


def face_maps(R, size, focal, W, H):
    """cv2.remap maps: where in the W x H panorama each face pixel comes from."""
    lon, lat = face_lonlat(R, size, focal)
    return ((lon / 360 + 0.5) * W - 0.5).astype(np.float32), ((0.5 - lat / 180) * H - 0.5).astype(np.float32)


def sample(pano, mapx, mapy, interp=cv2.INTER_CUBIC):
    """Remap with wrap-around at the left / right border of the panorama."""
    pad = 8
    wide = cv2.copyMakeBorder(pano, 0, 0, pad, pad, cv2.BORDER_WRAP)
    return cv2.remap(wide, mapx + pad, mapy, interp, borderMode=cv2.BORDER_REPLICATE)


def pixel_lonlat(xy, R, size, focal):
    """Face pixel coordinates (N, 2; COLMAP convention, pixel centre at +0.5) -> lon, lat in the panorama."""
    d = np.c_[(xy - size / 2) / focal, np.ones(len(xy))] @ R
    return rays_to_lonlat(d)
