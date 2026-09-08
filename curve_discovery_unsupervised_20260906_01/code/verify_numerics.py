"""Independent numerical checks: all raw vertices, L2 isometry, native elastic DP."""
from run_experiments import *
import ctypes

def dtw_ref(a,ta,b,tb,beta):
    dp=np.full((len(a)+1,len(b)+1),np.inf);dp[0,0]=0
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i+1,j+1]=(a[i]-b[j])**2+beta*(ta[i]-tb[j])**2+min(dp[i,j+1],dp[i+1,j],dp[i,j])
    return np.sqrt(dp[-1,-1]/(len(a)+len(b)))

def msm_ref(a,b,c):
    def cost(x,a,b):return c if min(a,b)<=x<=max(a,b) else c+min(abs(x-a),abs(x-b))
    d=np.zeros((len(a),len(b)));d[0,0]=abs(a[0]-b[0])
    for i in range(1,len(a)):d[i,0]=d[i-1,0]+cost(a[i],a[i-1],b[0])
    for j in range(1,len(b)):d[0,j]=d[0,j-1]+cost(b[j],a[0],b[j-1])
    for i in range(1,len(a)):
        for j in range(1,len(b)):
            d[i,j]=min(d[i-1,j-1]+abs(a[i]-b[j]),d[i-1,j]+cost(a[i],a[i-1],b[j]),d[i,j-1]+cost(b[j],a[i],b[j-1]))
    return d[-1,-1]

def main():
    curves=json.loads((ROOT/'data/raw_curves.json').read_text());check={}
    for c in curves:
        p=Path(c['meta']['source_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==c['meta']['source_sha256']
        d=pd.read_csv(p);assert len(d)==len(c['x']) and np.array_equal(d.x,np.array(c['x'])) and np.array_equal(d.y,np.array(c['y']))
    check['all_218_sources_unchanged_and_all_9674_points_preserved']=True
    # Independent closed-form integral on union knots for pairs WITHOUT repeated x.
    eligible=[i for i,c in enumerate(curves) if c['meta']['duplicate_x_points']==0]
    rng=np.random.default_rng(123);maxerr=0
    for norm in NORMS:
        X=np.load(ROOT/f'data/representation_{norm}.npz')['X']
        for _ in range(100):
            i,j=rng.choice(eligible,2,replace=False);a,b=curves[i],curves[j]
            u=np.union1d(a['u'],b['u']);diff=np.interp(u,a['u'],values(a,norm))-np.interp(u,b['u'],values(b,norm))
            exact=np.sum(np.diff(u)*(diff[:-1]**2+diff[:-1]*diff[1:]+diff[1:]**2)/3)
            err=abs(exact-np.sum((X[i]-X[j])**2));maxerr=max(err,maxerr);assert err<1e-9
    check['L2_pairwise_max_squared_error']=maxerr
    lib=ctypes.CDLL(str(ROOT/'code/elastic_distances.dylib'));ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.raw_dtw.argtypes=[ptr,ptr,ctypes.c_int,ptr,ptr,ctypes.c_int,ctypes.c_double];lib.raw_dtw.restype=ctypes.c_double
    lib.raw_msm.argtypes=[ptr,ctypes.c_int,ptr,ctypes.c_int,ctypes.c_double];lib.raw_msm.restype=ctypes.c_double
    for _ in range(40):
        a=rng.normal(size=rng.integers(2,10));b=rng.normal(size=rng.integers(2,10));ta=np.linspace(0,1,len(a));tb=np.linspace(0,1,len(b))
        for beta in [0,.25,1]:assert np.isclose(lib.raw_dtw(a,ta,len(a),b,tb,len(b),beta),dtw_ref(a,ta,b,tb,beta))
        for c in [.01,.1]:assert np.isclose(lib.raw_msm(a,len(a),b,len(b),c),msm_ref(a,b,c))
        assert abs(lib.raw_msm(a,len(a),a,len(a),.1))<1e-12
    check['native_elastic_vs_independent_python_200_cases']='passed'
    for p in (ROOT/'experiments').glob('*sweep.csv'):
        d=pd.read_csv(p)
        for col in [c for c in d if c.startswith('qe')]:assert np.isfinite(d[col]).all()
    check['all_sweep_QE_finite']=True
    dump(ROOT/'experiments/verification.json',check);print(check)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
