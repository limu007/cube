import numpy as np
from datetime import datetime

def parse_log(ifile,magid='magMPU',debug=0):
    dall=open(ifile,'br').read()
    mall=dall.decode('u8')
    mall=mall.split('\n')
    isel=[i for i in range(len(mall)) if mall[i][2:8]==magid]
    print('found %i magn. entries'%len(isel))
    gpslst=[]

    for i in isel:
        p=mall[i-6][20:]
        try:
            dm=datetime.strptime(p[:p.rfind(" ")],"%Y-%m-%d %H:%M:%S")
            gpslst.append(dm)
        except:
            gpslst.append(None)
        
    obc=np.array([mall[i][15:].strip().split(',') for i in isel]) #magMPU
    lode=np.array([mall[i-1][15:].strip().split(',') for i in isel]) #magMMC
    temps=[mall[i-3][mall[i-3].rfind(':')+1].strip().split(',') for i in isel]
    temps=np.array([a for a in temps if len(a)==5])
    obc=obc.astype(int)
    lode=lode.astype(int)
    temps=temps.astype(int)
    restime=[mall[i+4][mall[i+4].find(':')+1:].strip() for i in isel]
    restime=[int(a[:a.find(' ')]) for a in restime]
    hsel=[i for i in range(len(mall)) if mall[i].find('edi hk')>=0]
    #if debug>=1: print("%i edi hk ")
    ehklst=[]
    for i in hsel:
        p=mall[i+1]
        if debug>1: print(p) 
        if not p.find('UTC')>0:
            ehklst.append(None)
        p=p[p.find('UTC')+4:].strip()
        dm=datetime.strptime(p[:p.rfind(")")],"%d.%m.%Y %H:%M:%S")
        ehklst.append(dm)
    if debug==2:
        return mall,isel,hsel,ehklst
    print('found %i edi entries'%len([e for e in ehklst if e!=None]))
    d0=ehklst[0]
    qlat=[(e-d0).seconds for e in ehklst if e!=None]
    #qhash={hsel[j+1]:qlat[j] for j in range(len(hsel)-1)}
    ahsel=np.array(hsel)
    hpos=[sum(ahsel<i-6) for i in isel]
    if hpos[-1]>=len(qlat): hpos[-1]=len(qlat)-1
    time=np.array([(qlat[h],(gpslst[i]-d0).seconds,restime[i]) for i,h in enumerate(hpos)])
    print("start time ",d0)
    return time,obc,lode,isel,temps,d0

def count_ints(lmin=1,perc=[40,60]):
    #how many intervals above/below median
    a4,a6=np.percentile(y,perc)
    s4=np.where(y<a4)[0]
    s6=np.where(y>a6)[0]
    return sum(s4[1:]-s4[:-1]>1),sum(s6[1:]-s6[:-1]>1)

#--------------------------------------

def get_batter(ph,y,ndiv=10,nsig=2,niter=2,rep=0):
    lab=(ph*ndiv).astype(int)
    rmad=[]
    stad=[]
    cts=[]
    for i in range(ndiv):
        dsel=y[lab==i]
        if len(dsel)<2: 
            rmad.append(0)
            stad.append(0)
            continue
        for j in range((niter-1) if len(dsel)<5 else niter):
            mean,sdev=np.mean(dsel),np.std(dsel)
            dsel=dsel[abs(dsel-mean)<nsig*sdev]
        cts.append([sum(lab==i),len(dsel)])
        rmad.append(np.mean(dsel))
        stad.append(np.std(dsel))
    if rep==2: return np.array(stad)/y.std(),np.array(cts)
    if rep==1: return np.array(rmad),np.array(stad)
    return np.array(stad)/y.std()

def get_batter2(ph,y,ndiv=10,nsig=2,niter=2,rep=0):
    iph=(ph*ndiv).astype(int)
    qph=ph*ndiv-iph
    wei=[]
    val=[]
    for i in range(ndiv):
        wei.append([])
        val.append([])
    for i in range(len(ph)):
        if qph[i]<1-1e-4:
            val[iph[i]].append(y[i])
            wei[iph[i]].append(1-qph[i])
        if qph[i]>1e-4:
            val[(iph[i]+1)%ndiv].append(y[i])
            wei[(iph[i]+1)%ndiv].append(qph[i])
    wei=[np.array(w)/np.sum(w) for w in wei]
    prof=[np.sum(wei[i]*val[i]) for i in range(ndiv)]
    sprd=[np.sqrt(np.sum(wei[i]*(val[i]-prof[i])**2)) for i in range(ndiv)]
    if rep==2:
        return sprd,prof
    return sprd,prof

def get_scatter_orig(ph,y,ndiv=10,rep=0,nsig=0,niter=0):
    from scipy import ndimage as nd
    lab=(ph*ndiv).astype(int)
    if rep==1: return nd.mean(y,lab,np.arange(0,ndiv))
    stad=nd.standard_deviation(y,lab,np.arange(0,ndiv))
    stad[np.isnan(stad)]=0
    return stad/y.std()
pers=np.r_[5:20:0.1]


get_scatter=get_batter2

nsig=1.5
niter=2
def phase_mean(pers,t,y,ndiv=10,rep=0,mode='mean'):
    mep,qep=[],[]
    cep=[]
    for per in pers:
        ph=t/per-(t/per).astype(int)
        des,cnt=get_scatter(ph,y,ndiv=ndiv,nsig=nsig,niter=niter,rep=2)[:2]
        #des=des[des>0]
        if mode=='medi':
            mep.append(np.median(des))
            qep.append(np.median(abs(des-mep[-1])))
        else:
            mep.append(np.mean(des))
            qep.append(np.sqrt(np.mean((des-mep[-1])**2)))
        if rep==3: cep.append(sum(cnt[:,0]))
        elif rep==2: cep.append(sum(cnt[:,1])/sum(cnt[:,0]))
    if rep>=2: return np.array(mep),np.array(qep),np.array(cep)
    return np.array(mep),np.array(qep)

#--------------------------------------

chi_reject=10
chi_smooth=3

def chifun(res):
    #res=y-np.polyval(idx,t)-myspline(pts)(ph)
    rsel=abs(res)<np.percentile(abs(res),100-chi_reject)
    return sum(res[rsel]**2)

def spscan(t,y,perx=np.r_[20:80:0.5],ndiv=10,mean=None,doplot=False,detrend=[]):
    
    from scipy import interpolate as ip
    from scipy import optimize as op
    spuni,quni=[],[]
    tdif=t-t.mean()
    for per in perx:
        ph=t/per-(t/per).astype(int)
        if len(spuni)==0:
            if not np.iterable(mean):
                if len(detrend)>0: 
                    des,mean=get_scatter(ph,y-np.polyval(detrend,tdif),ndiv=ndiv)
                    spini=list(detrend)+list(mean)
                else:
                    des,mean=get_scatter(ph,y,ndiv=ndiv)
                    spini=mean
            else:
                spini=mean
            #myspline=lambda pts:ip.UnivariateSpline(np.r_[:1:1j*(ndiv+1)], list(pts)+[pts[0]], s=chi_smooth)
            def myspline(pts):
                spfun=ip.UnivariateSpline(np.r_[:1:1j*(ndiv+1)], list(pts)+[pts[0]], s=chi_smooth)
                spfun.set_smoothing_factor(chi_smooth)
                return spfun
        else:
            spini=np.mean(spuni[-3:],0)
        if len(detrend)>0:
            qmin=lambda pts:chifun(y-np.polyval(pts[:2],tdif)-myspline(pts[2:])(ph)) 
        else:
            qmin=lambda pts:chifun(y-myspline(pts)(ph))
        spbest=op.fmin(qmin,spini,disp=0)

        spuni.append(spbest)
        quni.append(qmin(spuni[-1]))
    if doplot:
        pl.plot(perx,quni)
    return spuni,quni

def correct_trend(t,y,pers,ndiv=10):
    from scipy import interpolate as ip

    outx=spscan(t,y,pers,ndiv=ndiv)
    spbest=outx[0][np.argmin(outx[1])]
    per=pers[np.argmin(outx[1])]
    ph=t/per-(t/per).astype(int)
    bspline=ip.UnivariateSpline(np.r_[:1:1j*(ndiv+1)], list(spbest)+[spbest[0]], s=chi_smooth)
    idx2=np.polyfit(t-t.mean(),y-bspline(ph),1)
    y2=y-np.polyval(idx2,t-t.mean())
    outx2=spscan(t,y2,pers,ndiv=ndiv)
    print(np.min(outx2[1])/np.min(outx[1]))
    return outx2[0],outx2[1],idx2

#--------------------------------------
def fit_min(perx,y,w,ndeg=6,halfwin=30,bigwin=60,doplot=False):
    pmin=perx[np.argmin(y)]
    print(pmin)
    if doplot: pl.plot(perx,y)
    sel=abs(perx-pmin)<halfwin
    id5=np.polyfit(perx[sel],y[sel],ndeg,w=w[sel])
    if doplot: pl.plot(perx[sel],np.polyval(id5,perx[sel]))
    rt=np.roots(np.polyder(id5))
    if sum(abs(rt.imag)<1e-10)==0: return rt
    pos=rt[abs(rt.imag)<1e-10].real
    if sum(abs(pos-pmin)<bigwin)==0:
        return pos
    pos=pos[abs(pos-pmin)<bigwin]
    isel=np.argmin(np.polyval(id5,pos))
    return pos[isel]

allper=[]
def find_me_period(fname,perx,st=4,doplot=False,halfwin=30,debug=0,src='lo'):
    time,obc,lo,qq=parse_log(ldir+fname)
    tsprd=time[st:].std(0)
    if tsprd[0]>4*tsprd[0]:
        print('GPS timing failed')
    t=time[st:,2]
    print("max time %i s"%time[-1,0])
    
    allrep=[]
    allqep=[]
    for i in range(3):
        if src=='lo': y=lo[st:,i]
        else: y=obc[st:,i]
        if sum(count_ints(y))<4: 
            print("channel %i not periodic"%(i+1))
            continue
        rep,qep=phase_mean(perx,t,y)
    
        #print(phasx[np.argmin(rep)])
        allrep.append(rep)
        allqep.append(qep)
    allrep=np.array(allrep).T
    allqep=np.array(allqep).T
    if debug==1: return allrep,allqep
    chansel=(allrep.max(0)-allrep.min(0))>allqep.mean(0)
    if sum(chansel)==0:
        print("no good minimum")
        isel=np.argmin(allrep.min(0))
        y=allrep[:,isel]
        w=1/allqep[:,isel]
        #return allrep,allqep
    else:
        print("selected %i channels"%sum(chansel))
        y=allrep[:,chansel].mean(1)
        w=1/allqep[:,chansel].mean(1)

    return fit_min(perx,y,w,halfwin=halfwin,doplot=doplot)

#--------------simulator------------

def make_curve(size=120,per=60,leng=200,ndeg=6,fac=3,noise=0.2):
    x=np.random.randint(0,leng,size=size)
    x=np.sort(x)
    phs=np.random.rand(ndeg)
    amp=np.random.rand(ndeg)
    ysyn=np.sin(x*2*np.pi/per)*phs[0]+np.cos(x*2*np.pi/per)*np.sqrt(1-phs[0]**2)
    for i in range(1,ndeg):
        ysyn=np.random.uniform()*fac*ysyn+np.sin(x*2*np.pi/per*i)*phs[i]+np.cos(x*2*np.pi/per*i)*np.sqrt(1-phs[i]**2)
    return x,ysyn+np.random.randn(len(x))*noise

#---------------------------------------------------------------------

def get_sat_pos(nw=None,tle=None,npt=200,delta=30,rep=0):
    import subprocess as sp
    from datetime import datetime,timedelta
    if nw==None: nw=datetime.now()
    #ff=zoo.stdin
    ff=open("/tmp/code","w")
    if tle!=None: ff.write('o '+tle.replace('\n','|')+'\n')
    for t in range(npt):
        nw=nw+timedelta(0,delta)
        ff.write("s %.1f\n"%nw.timestamp())
    ff.write("q\n")
    ff.close()
    #zoo.stdin.flush()
    out=sp.check_output(["python2","/home/munz/bin/predalpha.py","/tmp/code"],10)
    out=out.decode('u8').split('\n')
    import numpy as np
    if rep==1: 
        return np.array([a.split()[1:] for a in out[:-1]]).astype(float).T
    xlon,xlat=np.array([a.split()[1:] for a in out[:-1]]).astype(float).T[:2]
    return xlon,xlat

def plot_sun(mok):
    from matplotlib import pyplot as pl
    nn=mok[mok.columns[1]]
    ann=np.array(nn)
    ssel=ann>ann[0]
    ssel[:-1][ann[1:]-ann[:-1]>10]=False
    ssel[-1]=False
    nn=nn[ssel]
    pl.figure(figsize=(15,5))
    IRbeg=list(mok.columns).index('ssIRRad(X+)')
    fn=mok.columns[IRbeg]
    pl.plot(nn-ann[0],np.array(mok[fn])[ssel])
    pl.plot(nn-ann[0],np.array(mok[fn.replace('+','-')])[ssel])
    pl.xlabel('sec')
    pl.legend(['X+','X-'])
    pl.figure(figsize=(15,5))
    fn2=fn.replace('X','Y')
    pl.plot(nn-ann[0],np.array(mok[fn2])[ssel])
    pl.plot(nn-ann[0],np.array(mok[fn2.replace('+','-')])[ssel])
    pl.xlabel('sec')
    fn3=mok.columns[IRbeg+4]
    pl.plot(nn-ann[0],np.array(mok[fn3])[ssel])
    pl.legend(['Y+','Y-','Z-'])

    
def get_mag_dir(date="2022-07-02",lat=50,lon=15,alt=550):
    import numpy as np
    import urllib3
    #magmap={}
    fields=['total-intensity',
     'declination',
     'inclination',
     'north-intensity',
     'east-intensity',
     'vertical-intensity',
     'horizontal-intensity']
    import xml.etree.ElementTree as ET
    http = urllib3.PoolManager()
    #for lat in np.r_[-80:90:10]:
    path=f'http://geomag.bgs.ac.uk/web_service/GMModels/wmm/2020/?latitude={lat}&longitude={lon}&altitude={alt}&date={date}&format=xml'
    out=http.request('GET',path)
    try:
        et=ET.fromstring(out.data)
        table=et.find('field-value')
    except:
        print('failed to parse [%s]'%path)
        print(out.data)
        return
    #pos=f'{lat}_{lon}'
    return dict([(dd.tag,float(dd.text)) for dd in table])

######################################################s
# rotation framework revisited
quatern=lambda mat:np.array([mat[1][2]-mat[2][1],mat[2][0]-mat[0][2],mat[0][1]-mat[1][0]])
quat4=lambda mat:np.sqrt(1+np.trace(rot1))/2

andrcross=lambda p: np.array([[0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]])
andrcross4=lambda p: np.array([[0,p[2],-p[1],p[0]],[-p[2],0,p[0],p[1]],[p[1],-p[0],0,p[2]],[-p[0],-p[1],-p[2],0]]) #for rotation
recost = lambda q,q4: np.eye(3)*(q4**2-np.sum(q**2))+q[:,np.newaxis]*q[np.newaxis,:]*2-andrcross(q)*2*q4
def get_prot(a,b):
    '''reconstruct rotation from angle a to b
    '''
    norm=lambda v:np.sqrt(np.dot(v,v))
    import math as m
    ang1=np.arccos(a.dot(b))
    c1=np.cross(a,b)
    c1/=norm(c1)
    q=c1*m.sin(ang1/2)
    q4=m.cos(ang1/2)
    mat=recost(q,-q4)
    return mat

tle= """GRBALPHA                
1 47959U 21022AD  21136.80882950  .00001202  00000-0  87027-4 0  9997
2 47959  97.5614  39.6019 0023282  60.8418 299.5182 15.05651497  7118"""
tle_vzlu="""VZLUSAT-2
1 51085U 22002DF  22038.96554158  .00002307  00000-0  13979-3 0  9997
2 51085  97.5035 108.1989 0015559 156.2113 203.9840 15.11467589  3837"""
qth = (78.223,15.6582861,19)
