import numpy as np

def altaz_frame(lon,lat):
    '''rotation matrix from RaDec to AltAz (in cartesian coords)
    '''
    x=[-np.cos(lon)*np.sin(lat),-np.sin(lon)*np.sin(lat),np.cos(lat)]
    y=[-np.sin(lon),np.cos(lon),0]
    z=[np.cos(lon)*np.cos(lat),np.sin(lon)*np.cos(lat),np.sin(lat)]
    return np.array([x,y,z])


def get_pass_pars(p1,scale=1,rep=2):
    ep1=EarthLocation(p1['longitude'],p1['latitude'],p1['altitude']*1000*scale)
    #x,y,z=eloc2cart(ep1)
    #sky=SkyCoord(x*u.m,y*u.m,z*u.m,representation_type='cartesian',obstime=datetime.fromtimestamp(p1['epoch']), location=kos)
    edif=eloc2cart(ep1)-eloc2cart(kos)
    dis=np.sqrt((edif**2).sum())
    if rep==0: return edif/dis
    mat1=altaz_frame(np.deg2rad(qth[1]),np.deg2rad(qth[0]))
    if rep==1: return toangle(edif/dis)
    #return dis,sky.altaz.alt.value,sky.altaz.az.value
    return toangle(mat1@(edif/dis)),dis/1000,datetime.fromtimestamp(p1['epoch']).isoformat()