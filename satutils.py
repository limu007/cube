import numpy as np

def altaz_frame(lon,lat):
    '''rotation matrix from RaDec to AltAz (in cartesian coords)
    '''
    x=[-np.cos(lon)*np.sin(lat),-np.sin(lon)*np.sin(lat),np.cos(lat)]
    y=[-np.sin(lon),np.cos(lon),0]
    z=[np.cos(lon)*np.cos(lat),np.sin(lon)*np.cos(lat),np.sin(lat)]
    return np.array([x,y,z])

dsin=lambda x:np.sin(np.deg2rad(x))
dcos=lambda x:np.cos(np.deg2rad(x))
dist_cir=lambda a,b:dsin(a[0])*dsin(b[0])+dcos(a[0])*dcos(b[0])*dcos(b[1]-a[1]) #great circle

eloc2cart=lambda k: np.array([k.x.value,k.y.value,k.z.value])
toangle=lambda p:[np.rad2deg(np.arcsin(p[2])),np.rad2deg(np.arctan2(p[1],p[0]))]
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

norm=lambda v:np.sqrt(np.dot(v,v))
def triad(s,m):
    t1=s
    t2=np.cross(s,m) #vektorový súčin
    t2/=norm(t2)
    t3=np.cross(t1,t2)
    return np.array([t1,t2,t3])

